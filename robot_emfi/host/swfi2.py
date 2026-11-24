#!/usr/bin/env python3
"""Software fault injection with the DRAM fault model: where does one garbled 16-byte block hurt?

The memory tests showed that EM pulses corrupt DRAM reads, and that the AES-XTS memory encryption
turns every faulted access into one aligned 16-byte block of random bytes (8 fp16 or 4 fp32
values). This simulation plants such a block in the weights, the activations or the key/value
cache of the SmolVLA policy, in one stage of one inference, and compares the action chunk with the
fault-free one. It runs the stock LeRobot policy in fp16 (the TensorRT engines' precision) on the
local GPU, split as the deployed engines are (onnx_inits.py lists their weights):
  vision   one camera: SigLIP encoder + connector                                 98.2 M fp16
  prefill  VLM layers 0-14, layer 15's K/V projections, and the expert's K/V
           projections of the prefix for its 8 cross-attention layers             149.7 M fp16
  expert   one denoising step: the action expert (96.6 M fp16) and the fp32 action / time
           projectors
The key/value cache is what the prefill engine hands to the expert: the VLM's K/V for the 8
self-attention layers and the expert-projected K/V for the 8 cross-attention layers.
Sites:
  w-vision / w-prefill / w-expert   a weight block, read for one camera / the prefill / one step
  w-expert-all                      a weight block of the expert in all 10 steps (persistent)
  a-vision / a-prefill / a-expert   a block of the input activation one GEMM reads in that stage
  kv-write                          a block of the KV cache, read by all 10 steps
  kv-read                           a block of the KV cache in one step's read only
  kv-burst, kv-colbit               bit flips in 32 consecutive KV values / in one K/V column (not in
                                    the paper; in the default plan so that --seed 0 draws the paper's faults)
Weights are drawn in proportion to their size, activations to the bytes the GEMM reads. A 2-D
weight's block is 8 consecutive inputs of one output (layout "row", the [out, in] order of the
ONNX graphs) or, with layout "col", 8 consecutive outputs of one input; half of the faults each.
Inputs are the 8 reference observations from the 27 Oct rollouts (../data/traces).

usage (simulation environment): swfi2.py OUT.npz [--faults N] [--plan site,...] [--check]
"""

import argparse
import collections
import os
import re
import time
from contextlib import contextmanager, nullcontext
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
import torch
from torch import nn

from site_config import SMOLVLA_CKPT, TRACES
CKPT = Path(SMOLVLA_CKPT)
HERE = Path(__file__).resolve().parent
FIXTURE = HERE / "out/swfi2/fixture.npz"
TASK = "pick up the red cube and place it in the transparent box"
OBS = [("20261027_190527", 0), ("20261027_190527", 418), ("20261027_191033_seed1", 247),
       ("20261027_191155_seed2", 57), ("20261027_191155_seed2", 513), ("20261027_191301_seed3", 323),
       ("20261027_191352_seed4", 152), ("20261027_191352_seed4", 589)]     # (run, request step)
SITES = ("w-vision", "w-prefill", "w-expert", "w-expert-all", "a-vision", "a-prefill", "a-expert",
         "kv-write", "kv-read", "kv-burst", "kv-colbit")
BLOCK = 16
BURST = 32
LIM = [(-120.4, 120.4), (-105, 105), (-97.9, 97.9), (-103.9, 103.9), (-180, 180), (0, 100)]   # move_analyze.py
DEV = "cuda"


class Pass(nn.Module):
    """Stands in for an expert cross-layer K/V projection in the denoising steps: the prefill
    engine already projected the prefix, so the step reads the projected cache as it is."""

    def __init__(self, dtype):
        super().__init__()
        self.register_buffer("weight", torch.empty(0, dtype=dtype))

    def forward(self, x):
        return x


def kind_of(name):
    """Short type and layer of a parameter or module name."""
    m = re.search(r"layers\.(\d+)\.", name)
    parts = [p for p in name.split(".") if p not in ("weight", "bias")]
    return parts[-1] + (".bias" if name.endswith(".bias") else ""), int(m.group(1)) if m else -1


class Chain:
    """The policy split at the engine boundaries. self.stage says which stage runs: ("vision", cam),
    ("prefill", None) or ("expert", step); a planted activation fault waits for its stage."""

    def __init__(self, policy, post):
        from lerobot.policies.smolvla.modeling_smolvla import make_att_2d_masks
        self.mask2d = make_att_2d_masks
        self.policy, self.m, self.post = policy, policy.model, post
        self.vle = self.m.vlm_with_expert
        every = self.vle.self_attn_every_n_layers
        self.cross = [l for l in range(self.vle.num_vlm_layers) if every > 0 and l % every]
        self.hd = self.vle.vlm.config.text_config.head_dim
        ex = self.vle.lm_expert.layers
        self.kv_proj = {l: (ex[l].self_attn.k_proj, ex[l].self_attn.v_proj) for l in self.cross}
        self.stage, self.calls, self.record, self.act = None, collections.Counter(), None, None
        self.pools = self._weight_pools()
        for name, mod in self.vle.named_modules():
            if isinstance(mod, (nn.Linear, nn.Conv2d)):
                mod.register_forward_pre_hook(self._pre(name))
        for name in ("state_proj", "action_in_proj", "action_out_proj", "action_time_mlp_in", "action_time_mlp_out"):
            getattr(self.m, name).register_forward_pre_hook(self._pre(name))
        for l in self.cross:
            sa = ex[l].self_attn
            sa.k_proj, sa.v_proj = Pass(sa.k_proj.weight.dtype), Pass(sa.v_proj.weight.dtype)

    def _weight_pools(self):
        """Weights each engine reads, as the ONNX graphs hold them (dead VLM layer-15 parts and the
        unused final norm are not in the prefill graph)."""
        vlm = self.vle.vlm.model
        n = self.vle.num_vlm_layers
        pools = {"vision": [], "prefill": [], "expert": []}
        for name, p in vlm.vision_model.named_parameters(prefix="vision_model"):
            pools["vision"].append((name, p))
        for name, p in vlm.connector.named_parameters(prefix="connector"):
            pools["vision"].append((name, p))
        for name, p in vlm.text_model.layers.named_parameters(prefix="text.layers"):
            kind, layer = kind_of(name)
            if layer < n - 1 or kind in ("input_layernorm", "k_proj", "v_proj"):
                pools["prefill"].append((name, p))
        for l in self.cross:
            for proj, mod in zip(("x_k_proj", "x_v_proj"), self.kv_proj[l]):
                for pn, p in mod.named_parameters():
                    pools["prefill"].append((f"expert.layers.{l}.{proj}.{pn}", p))
        pools["prefill"] += [(f"state_proj.{pn}", p) for pn, p in self.m.state_proj.named_parameters()]
        cross_kv = {id(p) for l in self.cross for mod in self.kv_proj[l] for p in mod.parameters()}
        for name, p in self.vle.lm_expert.named_parameters(prefix="expert"):
            if id(p) not in cross_kv:
                pools["expert"].append((name, p))
        for mod in ("action_in_proj", "action_time_mlp_in", "action_time_mlp_out", "action_out_proj"):
            pools["expert"] += [(f"{mod}.{pn}", p) for pn, p in getattr(self.m, mod).named_parameters()]
        return {k: (v, np.array([p.numel() * p.element_size() for _, p in v], float)) for k, v in pools.items()}

    # ---- stages ------------------------------------------------------------------------------
    @contextmanager
    def at(self, stage, idx=None):
        prev, self.stage, self.calls = self.stage, (stage, idx), collections.Counter()
        try:
            yield
        finally:
            self.stage = prev

    def _pre(self, name):
        def hook(mod, args):
            if self.stage is None:
                return None
            k = self.calls[name]
            self.calls[name] += 1
            if self.record is not None:
                self.record.append((self.stage, name, k, args[0].numel() * args[0].element_size()))
            f = self.act
            if f is None or f["stage"] != self.stage or f["name"] != name or f["call"] != k:
                return None
            x = args[0].contiguous().clone()
            f["info"] = block_flat(x, f["rng"])
            return (x,) + tuple(args[1:])
        return hook

    def vision(self, img, cam):
        with self.at("vision", cam):
            return self.vle.embed_image(img)

    def prefill(self, embs, inp):
        images, masks, lt, lm, state = inp
        with self.at("prefill"):
            it = iter(embs)
            self.vle.embed_image = lambda img: next(it)
            try:
                pe, pad, att = self.m.embed_prefix(images, masks, lt, lm, state=state)
            finally:
                del self.vle.embed_image
            _, past = self.vle.forward(attention_mask=self.mask2d(pad, att), position_ids=torch.cumsum(pad, dim=1) - 1,
                                       past_key_values=None, inputs_embeds=[pe, None], use_cache=True,
                                       fill_kv_cache=True)
            for l in self.cross:
                k, v = past[l]["key_states"], past[l]["value_states"]
                kp, vp = self.kv_proj[l]
                past[l] = {"key_states": kp(k.to(kp.weight.dtype).view(*k.shape[:2], -1)).view(*k.shape[:2], -1, self.hd),
                           "value_states": vp(v.to(vp.weight.dtype).view(*v.shape[:2], -1)).view(*v.shape[:2], -1, self.hd)}
        return past, pad

    def denoise(self, past, pad, noise, start=0, x0=None, xs=None, step_past=None, step_ctx=None):
        """The 10 steps from `start` (x0 = the clean x_t entering it). step_past {step: cache} swaps
        the cache for one step; step_ctx {step: context factory} wraps one step."""
        n = self.m.config.num_steps
        dt = -1.0 / n
        x_t = noise if start == 0 else x0
        for step in range(start, n):
            if xs is not None:
                xs.append(x_t)
            tt = torch.tensor(1.0 + step * dt, dtype=torch.float32, device=DEV).expand(x_t.shape[0])
            p = (step_past or {}).get(step, past)
            ctx = (step_ctx or {}).get(step, nullcontext)
            with self.at("expert", step), ctx():
                v_t = self.m.denoise_step(prefix_pad_masks=pad, past_key_values=p, x_t=x_t, timestep=tt)
            x_t = x_t + dt * v_t
        return x_t

    def chunk(self, x):
        return self.post(x[:, :, :6]).detach().to(torch.float64).cpu().numpy()[0]

    def kv_list(self, past):
        return [past[l][k] for l in sorted(past) for k in ("key_states", "value_states")]

    # ---- faults -------------------------------------------------------------------------------
    @contextmanager
    def weight(self, spec):
        """Plants spec's block in its weight for the duration of the context."""
        w = spec["view"]()
        old = w.clone()
        w.copy_(spec["new"])
        try:
            yield
        finally:
            w.copy_(old)

    def weight_spec(self, stage, rng):
        params, sizes = self.pools[stage]
        name, p = params[int(rng.choice(len(params), p=sizes / sizes.sum()))]
        per = BLOCK // p.element_size()
        if p.dim() == 2 and "embedding" not in name and rng.random() < 0.5:
            layout = "col"
            j, o0 = int(rng.integers(p.shape[1])), int(rng.integers(p.shape[0] // per)) * per
            view = lambda: p.data[o0:o0 + per, j]
            off = o0 * p.shape[1] + j
        else:
            layout = "row" if p.dim() == 2 else "flat"
            off = int(rng.integers(p.numel() // per)) * per
            view = lambda: p.data.view(-1)[off:off + per]
        return {"view": view, "new": rand_block(rng, p.dtype), "target": name, "layout": layout, "offset": off,
                "old": view().float().cpu().numpy(), "isz": p.element_size()}


def rand_block(rng, dtype):
    return torch.frombuffer(bytearray(rng.bytes(BLOCK)), dtype=dtype).to(DEV)


def block_flat(x, rng):
    """One aligned block of contiguous x replaced with random bytes, in place."""
    per = BLOCK // x.element_size()
    b = int(rng.integers(x.numel() // per))
    flat = x.view(-1)
    old = flat[b * per:(b + 1) * per].float().cpu().numpy()
    flat[b * per:(b + 1) * per] = rand_block(rng, x.dtype)
    return b * per, old, flat[b * per:(b + 1) * per].float().cpu().numpy(), x.element_size()


def kv_fault(t, rng, model):
    """Copy of KV tensor t ([1, 177, heads, dim] fp16) with a block, burst or column-bit fault."""
    y = t.contiguous().clone()
    if model == "block":
        return y, block_flat(y, rng)
    a = y.cpu().numpy().reshape(y.shape[1], -1)                       # [tokens, features]
    u = a.view(np.uint16)
    if model == "burst":
        i = int(rng.integers(a.size - BURST + 1))
        old = a.reshape(-1)[i:i + BURST].astype(np.float32)
        u.reshape(-1)[i:i + BURST] ^= (np.uint16(1) << rng.integers(16, size=BURST).astype(np.uint16))
        new = a.reshape(-1)[i:i + BURST].astype(np.float32)
    else:                                                             # colbit
        i = int(rng.integers(a.shape[1]))
        old = a[:, i].astype(np.float32)
        u[:, i] ^= np.uint16(1 << int(rng.integers(16)))
        new = a[:, i].astype(np.float32)
    return torch.from_numpy(a.reshape(y.shape)).to(DEV), (i, old, new, 2)


def outcome(c, clean):
    """The EM campaigns' classes (move_analyze.py) on a chunk in degrees."""
    if not np.isfinite(c).all():
        return "nan", float("nan")
    d = float(np.abs(c - clean).max())
    if d == 0:
        return "unchanged", d
    if d < 10:
        return "small", d
    inside = all(LIM[j][0] - 0.5 <= c[:, j].min() and c[:, j].max() <= LIM[j][1] + 0.5 for j in range(6))
    return ("silent" if inside and np.abs(np.diff(c, axis=0)).max() <= 10 else "erratic"), d


def kv_diff(kv, clean):
    """Per KV tensor: tokens and features that differ from the clean cache, largest finite change."""
    tok, feat, mx, nonfin = [], [], [], 0
    for a, b in zip(kv, clean):
        a, b = a[0].reshape(a.shape[1], -1), b[0].reshape(b.shape[1], -1)
        ne = a != b
        tok.append(int(ne.any(1).sum()))
        feat.append(int(ne.any(0).sum()))
        d = (a.float() - b.float()).abs()
        d = d[torch.isfinite(d)]
        mx.append(float(d.max()) if d.numel() else 0.0)
        nonfin += int((~torch.isfinite(a)).sum())
    return tok, feat, mx, nonfin


def load():
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    policy = SmolVLAPolicy.from_pretrained(str(CKPT)).to(DEV).eval()
    pre, post = make_pre_post_processors(policy.config, pretrained_path=str(CKPT))
    return policy, pre, post


def inputs(policy, pre, top, side, state):
    from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS
    img = lambda a: torch.from_numpy(np.ascontiguousarray(a)).permute(2, 0, 1).float().div(255)
    b = pre({"observation.images.camera1": img(top), "observation.images.camera2": img(side),
             "observation.state": torch.from_numpy(np.asarray(state, np.float32)), "task": TASK})
    images, masks = policy.prepare_images(b)
    return images, masks, b[OBS_LANGUAGE_TOKENS], b[OBS_LANGUAGE_ATTENTION_MASK], policy.prepare_state(b)


def check(policy, pre, post):
    """Inputs and fp32 chunk against the bundle's fixture, fp16 against fp32, the split chain against
    the stock sampler, determinism, and the weight pools against the ONNX graphs."""
    fx = np.load(FIXTURE)
    images, masks, lt, lm, state = inputs(policy, pre, fx["raw"][0], fx["raw"][1], fx["state"])
    pix = torch.stack([i[0] for i in images]).cpu().numpy()
    print(f"pixels vs fixture: max |diff| {np.abs(pix - fx['pixel_values']).max():.2e}; "
          f"tokens equal {np.array_equal(lt.cpu().numpy(), fx['lang_tokens'])}; "
          f"state max |diff| {np.abs(state.cpu().numpy() - fx['model_state']).max():.2e}")
    fimg = [torch.from_numpy(fx["pixel_values"][c:c + 1]).to(DEV) for c in range(2)]
    fin = (fimg, [torch.ones(1, dtype=torch.bool, device=DEV)] * 2, torch.from_numpy(fx["lang_tokens"]).to(DEV),
           torch.from_numpy(fx["lang_masks"]).to(DEV), torch.from_numpy(fx["model_state"]).to(DEV))
    noise = torch.from_numpy(fx["noise"]).to(DEV)
    with torch.no_grad():
        a32 = policy.model.sample_actions(*fin, noise=noise)
    print(f"fp32 vs fixture action_pred (normalized): max |diff| {(a32.cpu() - torch.from_numpy(fx['action_pred'])).abs().max():.2e}")
    c32 = post(a32[:, :, :6]).double().cpu().numpy()[0]
    policy.model.vlm_with_expert.to(torch.float16)
    with torch.no_grad():
        a16 = policy.model.sample_actions(*fin, noise=noise)
    c16 = post(a16[:, :, :6]).double().cpu().numpy()[0]
    print(f"fp16 vs fp32 chunk: max |diff| {np.abs(c16 - c32).max():.3f} deg")
    chain = Chain(policy, post)
    with torch.no_grad():
        runs = []
        for _ in range(2):
            embs = [chain.vision(fimg[c], c) for c in range(2)]
            past, pad = chain.prefill(embs, fin)
            runs.append(chain.chunk(chain.denoise(past, pad, noise)))
    print(f"split chain vs stock fp16 sampler identical: {np.array_equal(runs[0], c16)}; "
          f"chain deterministic: {np.array_equal(runs[0], runs[1])}")
    for k, (params, sizes) in chain.pools.items():
        n = sum(p.numel() for _, p in params)
        print(f"pool {k}: {len(params)} tensors, {n / 1e6:.2f} M elements, {sizes.sum() / 1e6:.1f} MB")
    return chain


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out", nargs="?")
    ap.add_argument("--faults", type=int, default=150, help="per observation and site")
    ap.add_argument("--plan", default=",".join(SITES))
    ap.add_argument("--obs", type=int, default=len(OBS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    plan = a.plan.split(",")
    if set(plan) - set(SITES):
        ap.error(f"unknown sites {set(plan) - set(SITES)}")
    t0 = time.perf_counter()
    policy, pre, post = load()
    print(f"policy loaded in {time.perf_counter() - t0:.0f} s", flush=True)
    torch.set_grad_enabled(False)
    if a.check:
        check(policy, pre, post)
        return
    policy.model.vlm_with_expert.to(torch.float16)
    chain = Chain(policy, post)
    rng = np.random.default_rng(a.seed)
    cols = collections.defaultdict(list)
    devs, kvt, kvf, kvm, cleans = [], [], [], [], []
    act_pool = None
    for oi, (run, step) in enumerate(OBS[:a.obs]):
        d = np.load(TRACES / run / "trace.npz")
        i = int(np.flatnonzero(d["req_step"] == step)[0])
        inp = inputs(policy, pre, d["req_top"][i], d["req_side"][i], d["req_state"][i])
        noise = torch.from_numpy(rng.standard_normal((1, 50, 32)).astype(np.float32)).to(DEV)
        chain.record = [] if act_pool is None else None
        embs = [chain.vision(inp[0][c], c) for c in range(2)]
        past, pad = chain.prefill(embs, inp)
        xs = []
        clean = chain.chunk(chain.denoise(past, pad, noise, xs=xs))
        again = chain.chunk(chain.denoise(*chain.prefill([chain.vision(inp[0][c], c) for c in range(2)], inp), noise))
        if not np.array_equal(clean, again):
            raise RuntimeError("clean chain is not deterministic")
        cleans.append(clean)
        if act_pool is None:
            act_pool = {}
            dead = {f"vlm.model.text_model.layers.{chain.vle.num_vlm_layers - 1}.{m}"
                    for m in ("self_attn.q_proj", "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")}
            for st in ("vision", "prefill", "expert"):
                want = {"vision": ("vision", 0), "prefill": ("prefill", None), "expert": ("expert", 0)}[st]
                rec = [(n, k, b) for s, n, k, b in chain.record if s == want and n not in dead]
                act_pool[st] = (rec, np.array([b for _, _, b in rec], float))
                print(f"activation pool {st}: {len(rec)} GEMM inputs, {act_pool[st][1].sum() / 1e6:.1f} MB", flush=True)
            chain.record = None
        kv_clean = chain.kv_list(past)
        for site in plan:
            for _ in range(a.faults):
                idx, info, kvd = -1, {}, None
                if site in ("w-vision", "w-prefill", "w-expert", "w-expert-all"):
                    st = site[2:].replace("-all", "")
                    spec = chain.weight_spec(st, rng)
                    info = {"target": spec["target"], "layout": spec["layout"], "offset": spec["offset"],
                            "old": spec["old"], "new": spec["new"].float().cpu().numpy(), "isz": spec["isz"]}
                    if site == "w-vision":
                        idx = int(rng.integers(2))
                        with chain.weight(spec):
                            e = chain.vision(inp[0][idx], idx)
                        p2, _ = chain.prefill([e if c == idx else embs[c] for c in range(2)], inp)
                        x = chain.denoise(p2, pad, noise)
                        kvd = kv_diff(chain.kv_list(p2), kv_clean)
                    elif site == "w-prefill":
                        with chain.weight(spec):
                            p2, _ = chain.prefill(embs, inp)
                        x = chain.denoise(p2, pad, noise)
                        kvd = kv_diff(chain.kv_list(p2), kv_clean)
                    elif site == "w-expert":
                        idx = int(rng.integers(10))
                        x = chain.denoise(past, pad, noise, start=idx, x0=xs[idx],
                                          step_ctx={idx: lambda spec=spec: chain.weight(spec)})
                    else:
                        with chain.weight(spec):
                            x = chain.denoise(past, pad, noise)
                elif site in ("a-vision", "a-prefill", "a-expert"):
                    st = site[2:]
                    rec, sizes = act_pool[st]
                    name, call, _ = rec[int(rng.choice(len(rec), p=sizes / sizes.sum()))]
                    idx = int(rng.integers(2 if st == "vision" else 10)) if st != "prefill" else -1
                    chain.act = {"stage": (st, None if st == "prefill" else idx), "name": name, "call": call, "rng": rng}
                    try:
                        if st == "vision":
                            e = chain.vision(inp[0][idx], idx)
                            p2, _ = chain.prefill([e if c == idx else embs[c] for c in range(2)], inp)
                            x = chain.denoise(p2, pad, noise)
                            kvd = kv_diff(chain.kv_list(p2), kv_clean)
                        elif st == "prefill":
                            p2, _ = chain.prefill(embs, inp)
                            x = chain.denoise(p2, pad, noise)
                            kvd = kv_diff(chain.kv_list(p2), kv_clean)
                        else:
                            x = chain.denoise(past, pad, noise, start=idx, x0=xs[idx])
                        off, old, new, isz = chain.act["info"]
                    finally:
                        chain.act = None
                    info = {"target": name, "layout": "act", "offset": off, "old": old, "new": new, "isz": isz}
                else:
                    t = int(rng.integers(len(kv_clean)))
                    model = {"kv-burst": "burst", "kv-colbit": "colbit"}.get(site, "block")
                    y, (off, old, new, isz) = kv_fault(kv_clean[t], rng, model)
                    l, kk = t // 2, ("key_states", "value_states")[t % 2]
                    p2 = dict(past)
                    p2[l] = {**past[l], kk: y}
                    if site == "kv-read":
                        idx = int(rng.integers(10))
                        x = chain.denoise(past, pad, noise, start=idx, x0=xs[idx], step_past={idx: p2})
                    else:
                        x = chain.denoise(p2, pad, noise)
                    info = {"target": f"kv.layers.{l}.{'kv'[t % 2]}", "layout": model, "offset": off,
                            "old": old, "new": new, "isz": isz}
                c = chain.chunk(x)
                cls, dmax = outcome(c, clean)
                kind, layer = kind_of(info["target"])
                with np.errstate(invalid="ignore"):              # random bytes include signalling NaNs
                    new = np.asarray(info["new"], np.float64)
                fin = new[np.isfinite(new)]
                cols["obs"].append(oi)
                cols["site"].append(site)
                cols["idx"].append(idx)
                cols["target"].append(info["target"])
                cols["kind"].append(kind)
                cols["layer"].append(layer)
                cols["layout"].append(info["layout"])
                cols["offset"].append(info["offset"])
                cols["isz"].append(info["isz"])
                cols["new_nonfinite"].append(int((~np.isfinite(new)).sum()))
                cols["new_max"].append(float(np.abs(fin).max()) if fin.size else 0.0)
                cols["old_max"].append(float(np.nanmax(np.abs(info["old"]))))
                cols["outcome"].append(cls)
                cols["dmax"].append(dmax)
                devs.append((c - clean).astype(np.float32))
                tok, feat, mx, nonfin = kvd if kvd else ([0] * 32, [0] * 32, [0.0] * 32, 0)
                kvt.append(tok)
                kvf.append(feat)
                kvm.append(mx)
                cols["kv_nonfinite"].append(nonfin)
        print(f"obs {oi} ({run} step {step}) done, {len(cols['obs'])} faults, {time.perf_counter() - t0:.0f} s", flush=True)
        if a.out:
            np.savez(a.out, devs=np.array(devs), kv_tok=np.array(kvt, np.int16), kv_feat=np.array(kvf, np.int16),
                     kv_max=np.array(kvm, np.float32), clean=np.array(cleans),
                     **{k: np.array(v) for k, v in cols.items()})
    print(f"done: {len(cols['obs'])} faults -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
