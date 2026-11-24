"""vLLM worker extension that hashes the live model weights and reports changes.

Loaded via ``vllm serve ... --worker-extension-cls jetson.thor.weight_check_ext.WeightCheckWorkerExtension``
(jetson/thor/serve_vllm.py does this when vllm.weight_check = true).
Its methods run inside the worker process (``self`` is the vLLM Worker), where
``self.model_runner.model`` is the model whose parameters live in GPU memory.

Call flow (via AsyncLLM.collective_rpc from the API-server process):
  * ``wc_baseline()`` once  -> snapshot every parameter: a change-sensitive hash
    plus a CPU copy for later element-level diffing.
  * ``wc_check(attempt)``   -> re-hash every parameter; for any that differ from
    the baseline, diff against the CPU copy and return which tensor changed
    (layer number + weight type like ``down_proj``) and the changed values.
  * ``wc_baseline_from_file(files)`` -> like wc_baseline, but the reference is
    built from the checkpoint on disk (each file's bytes verified against a
    known sha256), so a corrupted load can never become the reference.
  * ``wc_zero()``           -> zero every parameter in place (before a reload,
    so lost reload writes show up as zeros: ``zero_elems`` in wc_check).
Every in-place change is undone by the next ``reload_weights``.
"""

from __future__ import annotations

import hashlib
import os
import re

import torch

from common import config


_LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.")
_SCALAR_SUFFIXES = ("weight", "bias", "weight_scale", "scale", "qweight", "qzeros")

# int dtype of the same element size, for exact bitwise comparison (NaN-safe).
_INT_VIEW = {
    torch.bfloat16: torch.int16,
    torch.float16: torch.int16,
    torch.float32: torch.int32,
    torch.float64: torch.int64,
    torch.int8: torch.int8,
    torch.uint8: torch.uint8,
    torch.int16: torch.int16,
    torch.int32: torch.int32,
    torch.int64: torch.int64,
}
for _fp8 in ("float8_e4m3fn", "float8_e5m2"):
    if hasattr(torch, _fp8):
        _INT_VIEW[getattr(torch, _fp8)] = torch.uint8


_dump: tuple | None = None


def _dump_settings() -> tuple:
    """(dump dir or "", max changed elements per dump) from config/thor.toml, read once per worker."""
    global _dump
    if _dump is None:
        cfg = config.load("thor.toml")
        directory = cfg.get("weight_check.dump_dir", default="")
        _dump = (directory, cfg.get("weight_check.dump_max_elems", int) if directory else 0)
    return _dump


def _hash_tensor(t: torch.Tensor) -> tuple:
    """Change-sensitive hash of a tensor's raw bytes (two int64 reductions).

    h0 = sum(bytes) catches every single-byte change; h1 = sum(byte * pos) makes
    multi-flip cancellation essentially impossible. Both fit in int64 for the
    tensor sizes here (<= a few GiB). Computed on-device, no host transfer.
    """
    b = t.detach().contiguous().flatten().view(torch.uint8).to(torch.int64)
    n = b.numel()
    if n == 0:
        return (0, 0, 0)
    idx = torch.arange(1, n + 1, device=b.device, dtype=torch.int64) % 1009
    h0 = int(b.sum().item())
    h1 = int((b * idx).sum().item())
    return (h0, h1, int(n))


def _parse_name(name: str) -> tuple:
    """Return (layer_number_or_None, weight_type) e.g. (13, 'down_proj')."""
    m = _LAYER_RE.search(name)
    layer = int(m.group(1)) if m else None
    parts = name.split(".")
    leaf = parts[-2] if len(parts) >= 2 and parts[-1] in _SCALAR_SUFFIXES else parts[-1]
    return layer, leaf


def _fmt_value(v: torch.Tensor):
    """Scalar tensor element -> python value plus its raw hex bit pattern."""
    dt = v.dtype
    if dt.is_floating_point:
        bits = int(v.view(_INT_VIEW[dt]).item()) & ((1 << (v.element_size() * 8)) - 1)
        return float(v.float().item()), f"0x{bits:0{v.element_size() * 2}x}"
    iv = int(v.item())
    return iv, f"0x{iv & ((1 << (v.element_size() * 8)) - 1):0{v.element_size() * 2}x}"


def _diff_tensor(name: str, base_cpu, cur_t: torch.Tensor, max_elems: int) -> dict:
    """Element-level diff of ``cur_t`` (live, on GPU) against ``base_cpu`` (CPU
    baseline copy). Returns a record with the changed-element count and up to
    ``max_elems`` (index, expected->actual, raw bits) entries. ``base_cpu`` may
    be ``None`` when the baseline stored only a hash (tensor too large to copy);
    then the change is reported without element detail.
    """
    layer, wtype = _parse_name(name)
    rec = {
        "name": name,
        "layer": layer,
        "type": wtype,
        "dtype": str(cur_t.dtype).replace("torch.", ""),
        "shape": list(cur_t.shape),
    }
    if base_cpu is None:
        rec.update({"changed_elems": None, "elems": [],
                    "note": "hash_only_baseline (tensor too large for element diff)"})
        return rec
    base = base_cpu.flatten()
    cur = cur_t.to("cpu").flatten()
    if base.numel() != cur.numel():
        rec.update({"changed_elems": None, "elems": [],
                    "note": f"size_changed {base.numel()}->{cur.numel()}"})
        return rec
    iv = _INT_VIEW.get(cur_t.dtype, torch.int64)
    mask = base.view(iv) != cur.view(iv)
    flat = torch.nonzero(mask, as_tuple=False).flatten()
    cnt = int(flat.numel())
    # Changed elements that read back as all-zero bits: after wc_zero() these
    # are words the reload never wrote (lost writes); the rest is garbage.
    zero_cnt = int((mask & (cur.view(iv) == 0)).sum().item())
    elems = []
    for i in flat[:max_elems].tolist():
        exp_v, exp_bits = _fmt_value(base[i])
        act_v, act_bits = _fmt_value(cur[i])
        elems.append({
            "flat_index": int(i),
            "expected": exp_v, "expected_bits": exp_bits,
            "actual": act_v, "actual_bits": act_bits,
        })
    rec.update({"changed_elems": cnt, "zero_elems": zero_cnt, "elems": elems})
    # weight_check.dump_dir: save every changed index with its expected/actual bits (.npz), for
    # changes up to weight_check.dump_max_elems elements (partial wipes; full wipes are skipped).
    dump_dir, dump_max = _dump_settings()
    if dump_dir and 0 < cnt <= dump_max:
        import time
        import numpy as np
        os.makedirs(dump_dir, exist_ok=True)
        path = os.path.join(dump_dir, f"{time.strftime('%Y%m%dT%H%M%S')}_{name}.npz")
        np.savez_compressed(
            path, shape=np.array(cur_t.shape), stride=np.array(cur_t.stride()),
            index=flat.numpy().astype(np.int64),
            expected=base.view(iv)[flat].numpy(), actual=cur.view(iv)[flat].numpy())
        rec["dump"] = path
    return rec


class WeightCheckWorkerExtension:
    """Mixed into the vLLM Worker; methods invoked via collective_rpc."""

    def _wc_model(self):
        model = getattr(self.model_runner, "model", None)
        if model is None:
            raise RuntimeError("weight-check: worker has no model_runner.model")
        return model

    def wc_zero(self) -> dict:
        """Overwrite every parameter with zeros in place (same storage), so a
        following reload_weights that loses a write leaves a visible zero
        instead of an identical old value. Returns how many elements are still
        non-zero afterwards (expected 0)."""
        model = self._wc_model()
        nonzero = 0
        total = 0
        with torch.no_grad():
            for _, p in model.named_parameters():
                p.detach().zero_()
            torch.cuda.synchronize()
            for _, p in model.named_parameters():
                nonzero += int(torch.count_nonzero(p.detach()).item())
                total += p.numel()
        return {"zeroed_elems": int(total), "nonzero_after": nonzero}

    @staticmethod
    def _read_verified(path: str, sha256: str, tries: int = 3) -> bytes:
        """Read ``path`` from the drive (page cache dropped first) until its
        sha256 matches; a glitched read is simply retried."""
        seen = []
        for _ in range(tries):
            fd = os.open(path, os.O_RDONLY)
            try:
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                with os.fdopen(os.dup(fd), "rb") as fh:
                    data = fh.read()
            finally:
                os.close(fd)
            digest = hashlib.sha256(data).hexdigest()
            if digest == sha256:
                return data
            seen.append(digest)
        raise RuntimeError(f"{path}: sha256 mismatch on {tries} reads: {seen}")

    def wc_baseline_from_file(self, files: list) -> dict:
        """Reference = the checkpoint on disk, not whatever a reload produced.

        ``files`` is [{"path": ..., "sha256": ...}, ...]. Every parameter's
        expected value is rebuilt from the checkpoint tensors (fused params via
        the model's packed_modules_mapping, concatenated on dim 0), then hashed
        on the device exactly like wc_baseline.
        """
        from safetensors.torch import load as load_safetensors

        model = self._wc_model()
        tensors = {}
        for f in files:
            tensors.update(load_safetensors(self._read_verified(f["path"], f["sha256"])))
        packed = getattr(model, "packed_modules_mapping", {}) or {}

        def expected(name: str) -> torch.Tensor:
            if name in tensors:
                return tensors[name]
            for fused, parts in packed.items():
                token = f".{fused}."
                if token in name:
                    return torch.cat(
                        [tensors[name.replace(token, f".{part}.")] for part in parts], 0
                    )
            raise KeyError(f"no checkpoint tensor for parameter {name}")

        hashes, copies, total_bytes = {}, {}, 0
        for name, p in model.named_parameters():
            exp = expected(name).contiguous()
            if tuple(exp.shape) != tuple(p.shape) or exp.dtype != p.dtype:
                raise ValueError(
                    f"{name}: checkpoint {tuple(exp.shape)} {exp.dtype} vs "
                    f"loaded {tuple(p.shape)} {p.dtype}"
                )
            hashes[name] = _hash_tensor(exp.to(p.device))
            copies[name] = exp
            total_bytes += exp.numel() * exp.element_size()
        torch.cuda.empty_cache()
        self._wc_hash, self._wc_base_cpu = hashes, copies
        return {
            "baselined": True,
            "source": "file",
            "tensors": len(hashes),
            "bytes": int(total_bytes),
        }

    def wc_baseline(self) -> dict:
        """Snapshot every parameter (hash + CPU copy). Idempotent."""
        model = self._wc_model()
        self._wc_hash = {}
        self._wc_base_cpu = {}
        total_bytes = 0
        for name, p in model.named_parameters():
            t = p.detach()
            self._wc_hash[name] = _hash_tensor(t)
            self._wc_base_cpu[name] = t.to("cpu", copy=True)
            total_bytes += t.numel() * t.element_size()
        return {
            "baselined": True,
            "tensors": len(self._wc_hash),
            "bytes": int(total_bytes),
        }

    def wc_check(self, attempt: int = -1, max_tensors: int = 256,
                 max_elems: int = 256) -> dict:
        """Re-hash all params; report those differing from the baseline.

        Baseline stays fixed, so each check reports the full set of parameters
        currently corrupted vs the pristine load (cumulative), with the exact
        changed elements (expected vs actual value) for each.
        """
        if not getattr(self, "_wc_hash", None):
            # Lazy baseline: first call snapshots the (clean) loaded weights.
            summary = self.wc_baseline()
            summary.update({"attempt": attempt, "changed": [], "changed_count": 0})
            return summary

        model = self._wc_model()
        changed = []
        for name, p in model.named_parameters():
            t = p.detach()
            if _hash_tensor(t) == self._wc_hash.get(name):
                continue
            changed.append(_diff_tensor(name, self._wc_base_cpu.get(name), t, max_elems))
            if len(changed) >= max_tensors:
                break
        return {
            "attempt": attempt,
            "baselined": False,
            "changed_count": len(changed),
            "changed": changed,
        }
