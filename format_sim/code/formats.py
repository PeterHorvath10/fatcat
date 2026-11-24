"""Parameter formats at byte level."""
import numpy as np
import torch

BLOCK = 16
E2M1 = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6], dtype=torch.float32)
E2M1_MAG = E2M1[:8]
E2M1_MID = (E2M1_MAG[1:] + E2M1_MAG[:-1]) / 2


def _e2m1_encode(x):
    """x (already divided by its scale) -> nibble 0..15 (bit 3 = sign), nearest value, saturating at 6."""
    mag = torch.bucketize(x.abs().clamp(max=6.0), E2M1_MID.to(x.device))
    return (mag + 8 * (x < 0)).to(torch.uint8)


def _e2m1_decode(n):
    return E2M1.to(n.device)[n.long()]


def _nibbles(r):
    """16 bytes -> 32 nibbles, low nibble first."""
    return torch.stack([r & 15, r >> 4], dim=-1).reshape(-1)


def _e4m3_decode(b):
    return b.contiguous().view(torch.float8_e4m3fn).to(torch.float32)


def _e4m3_encode(x):
    return x.clamp(-448, 448).to(torch.float8_e4m3fn).view(torch.uint8)


def _bf16_bytes(r):
    return r.contiguous().view(torch.bfloat16).to(torch.float32)


class TensorState:
    """regions: {name: nbytes}. clean(): float32 [out, in]. fault(region, block, r): (flat idx, float32 values)."""
    regions = {}

    def nblocks(self, region):
        return -(-self.regions[region] // BLOCK)


class BF16(TensorState):
    def __init__(self, W):
        self.W = W.to(torch.bfloat16)
        self.N = W.numel()
        self.regions = {"codes": 2 * self.N}

    def clean(self):
        return self.W.float()

    def fault(self, region, block, r):
        idx = 8 * block + torch.arange(8, device=r.device)
        keep = idx < self.N
        return idx[keep], _bf16_bytes(r)[keep]


class FP8(TensorState):
    def __init__(self, W):
        W = W.float()
        self.shape = W.shape
        self.scale = (W.abs().max() / 448.0).clamp(min=1e-12)
        self.codes = _e4m3_encode(W / self.scale).reshape(-1)
        self.N = W.numel()
        self.regions = {"codes": self.N, "tscale": 4}

    def clean(self):
        return (_e4m3_decode(self.codes) * self.scale).reshape(self.shape)

    def fault(self, region, block, r):
        if region == "codes":
            idx = 16 * block + torch.arange(16, device=r.device)
            keep = idx < self.N
            return idx[keep], (_e4m3_decode(r) * self.scale)[keep]
        new = r[:4].contiguous().view(torch.float32)
        return torch.arange(self.N, device=r.device), _e4m3_decode(self.codes) * new


class INT8(TensorState):
    def __init__(self, W):
        W = W.float()
        self.shape = W.shape
        o, i = W.shape
        self.scale = (W.abs().amax(1) / 127.0).clamp(min=1e-12).to(torch.bfloat16).float()
        self.codes = torch.round(W / self.scale[:, None]).clamp(-128, 127).to(torch.int8)
        self.N = W.numel()
        self.regions = {"codes": self.N, "scales": 2 * o}

    def clean(self):
        return self.codes.float() * self.scale[:, None]

    def fault(self, region, block, r):
        o, i = self.shape
        if region == "codes":
            idx = 16 * block + torch.arange(16, device=r.device)
            keep = idx < self.N
            idx = idx[keep]
            return idx, r.view(torch.int8).float()[keep] * self.scale[idx // i]
        rows = 8 * block + torch.arange(8, device=r.device)
        keep = rows < o
        rows, new = rows[keep], _bf16_bytes(r)[keep]
        idx = (rows[:, None] * i + torch.arange(i, device=r.device)[None]).reshape(-1)
        return idx, (self.codes[rows].float() * new[:, None]).reshape(-1)


class INT4(TensorState):
    G = 128

    def __init__(self, W):
        W = W.float()
        self.shape = W.shape
        o, i = W.shape
        assert i % self.G == 0
        Wg = W.reshape(o, i // self.G, self.G)
        self.scale = (Wg.abs().amax(2) / 7.0).clamp(min=1e-12).to(torch.bfloat16).float()
        self.codes = torch.round(Wg / self.scale[:, :, None]).clamp(-8, 7).to(torch.int8).reshape(-1)
        self.N = W.numel()
        self.regions = {"codes": self.N // 2, "scales": 2 * self.scale.numel()}

    def clean(self):
        return (self.codes.float() * self.scale.reshape(-1).repeat_interleave(self.G)).reshape(self.shape)

    def fault(self, region, block, r):
        if region == "codes":
            idx = 32 * block + torch.arange(32, device=r.device)
            n = _nibbles(r).to(torch.int16)
            vals = torch.where(n < 8, n, n - 16).float()
            keep = idx < self.N
            idx = idx[keep]
            return idx, vals[keep] * self.scale.reshape(-1)[idx // self.G]
        g = 8 * block + torch.arange(8, device=r.device)
        keep = g < self.scale.numel()
        g, new = g[keep], _bf16_bytes(r)[keep]
        idx = (g[:, None] * self.G + torch.arange(self.G, device=r.device)[None]).reshape(-1)
        return idx, self.codes[idx].float() * new.repeat_interleave(self.G)


class NVFP4(TensorState):
    B = 16

    def __init__(self, W):
        W = W.float()
        self.shape = W.shape
        o, i = W.shape
        assert i % self.B == 0
        self.gscale = (448.0 * 6.0 / W.abs().max().clamp(min=1e-12)).float()
        Wb = W.reshape(-1, self.B)
        s = (self.gscale * Wb.abs().amax(1) / 6.0).clamp(max=448.0)
        self.bscale = _e4m3_encode(s)
        s = _e4m3_decode(self.bscale)
        x = torch.where(s[:, None] > 0, Wb * self.gscale / s[:, None], torch.zeros_like(Wb))
        self.codes = _e2m1_encode(x).reshape(-1)
        self.N = W.numel()
        self.regions = {"codes": self.N // 2, "bscale": self.N // self.B, "gscale": 4}

    def _deq(self, codes, bscale_per_weight, g):
        return _e2m1_decode(codes) * bscale_per_weight / g

    def clean(self):
        return self._deq(self.codes, _e4m3_decode(self.bscale).repeat_interleave(self.B), self.gscale).reshape(self.shape)

    def fault(self, region, block, r):
        dev = r.device
        if region == "codes":
            idx = 32 * block + torch.arange(32, device=dev)
            keep = idx < self.N
            idx = idx[keep]
            s = _e4m3_decode(self.bscale[idx // self.B])
            return idx, self._deq(_nibbles(r)[keep], s, self.gscale)
        if region == "bscale":
            b = 16 * block + torch.arange(16, device=dev)
            keep = b < self.bscale.numel()
            b, new = b[keep], _e4m3_decode(r)[keep]
            idx = (b[:, None] * self.B + torch.arange(self.B, device=dev)[None]).reshape(-1)
            return idx, self._deq(self.codes[idx], new.repeat_interleave(self.B), self.gscale)
        new = r[:4].contiguous().view(torch.float32)
        return torch.arange(self.N, device=dev), self._deq(self.codes, _e4m3_decode(self.bscale).repeat_interleave(self.B), new)


class MXFP4(TensorState):
    B = 32

    def __init__(self, W):
        W = W.float()
        self.shape = W.shape
        o, i = W.shape
        assert i % self.B == 0
        Wb = W.reshape(-1, self.B)
        m = Wb.abs().amax(1)
        e = torch.where(m > 0, torch.floor(torch.log2(m.clamp(min=1e-38))) - 2, torch.full_like(m, -127.0)).clamp(-127, 127)
        self.scales = (e + 127).to(torch.uint8)
        self.codes = _e2m1_encode(Wb / torch.pow(2.0, e)[:, None]).reshape(-1)
        self.N = W.numel()
        self.regions = {"codes": self.N // 2, "scales": self.N // self.B}

    @staticmethod
    def _e8m0(b):
        v = torch.pow(torch.tensor(2.0, device=b.device), b.float() - 127.0)
        return torch.where(b == 255, torch.full_like(v, float("nan")), v)

    def clean(self):
        return (_e2m1_decode(self.codes) * self._e8m0(self.scales).repeat_interleave(self.B)).reshape(self.shape)

    def fault(self, region, block, r):
        dev = r.device
        if region == "codes":
            idx = 32 * block + torch.arange(32, device=dev)
            keep = idx < self.N
            idx = idx[keep]
            return idx, _e2m1_decode(_nibbles(r)[keep]) * self._e8m0(self.scales[idx // self.B])
        b = 16 * block + torch.arange(16, device=dev)
        keep = b < self.scales.numel()
        b, new = b[keep], self._e8m0(r)[keep]
        idx = (b[:, None] * self.B + torch.arange(self.B, device=dev)[None]).reshape(-1)
        return idx, _e2m1_decode(self.codes[idx]) * new.repeat_interleave(self.B)


class GGUF(TensorState):
    """Raw bytes of one GGUF tensor: [rows, row_bytes], each row a sequence of quantization blocks."""

    def __init__(self, data, qtype, device, row_perm=None):
        import gguf
        self.row_perm = row_perm
        self.gguf = gguf
        self.qtype = qtype
        self.wpb, self.bs = gguf.GGML_QUANT_SIZES[qtype]
        self.raw = np.ascontiguousarray(data).reshape(-1, self.bs)
        self.rows = data.shape[0]
        self.device = device
        self.N = self.raw.shape[0] * self.wpb
        self.regions = {"data": self.raw.size}

    SCALE_BYTES = {"Q4_0": (0, 2), "Q4_1": (0, 4), "Q5_0": (0, 2), "Q5_1": (0, 4), "Q8_0": (0, 2),
                   "Q4_K": (0, 16), "Q5_K": (0, 16), "Q6_K": (192, 210)}

    def scale_block(self, gblock):
        """The 16-byte block that covers the first scale byte of quantization block `gblock`."""
        return (gblock * self.bs + self.SCALE_BYTES[self.qtype.name][0]) // BLOCK

    def clean(self):
        w = self.gguf.quants.dequantize(self.raw, self.qtype)
        w = torch.from_numpy(np.ascontiguousarray(w)).to(self.device).reshape(self.rows, -1)
        if self.row_perm is not None:
            out = torch.empty_like(w)
            out[self.row_perm] = w
            w = out
        return w

    def fault(self, region, block, r):
        off = BLOCK * block
        end = min(off + BLOCK, self.raw.size)
        b0, b1 = off // self.bs, (end - 1) // self.bs
        blk = self.raw[b0:b1 + 1].copy()
        blk.reshape(-1)[off - b0 * self.bs:end - b0 * self.bs] = r.cpu().numpy()[:end - off]
        w = self.gguf.quants.dequantize(blk, self.qtype).reshape(-1)
        idx = b0 * self.wpb + torch.arange(w.size, device=self.device)
        lo, hi = self.SCALE_BYTES[self.qtype.name]
        self.hit_scale = any(lo <= (o % self.bs) < hi for o in range(off, end))
        if self.row_perm is not None:
            cols = self.N // self.rows
            idx = self.row_perm[idx // cols] * cols + idx % cols
        return idx, torch.from_numpy(np.ascontiguousarray(w)).to(self.device)


SYNTH = {"bf16": BF16, "fp8": FP8, "int8": INT8, "int4": INT4, "nvfp4": NVFP4, "mxfp4": MXFP4}
