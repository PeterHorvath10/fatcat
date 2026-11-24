#!/usr/bin/env python3
"""Block faults in the deployed TensorRT chain: the block simulation of Section 6.3, on the Jetson.

host/swfi2.py plants one garbled 16-byte block per inference in a PyTorch copy of the policy. This
plants the same fault in the deployed TensorRT engines, through swfi.py's split of the chain (same
engines and arithmetic as the rollout), and classifies the plan as swfi2.py does:
  w-vision      a block of the vision engine's weight memory while it encodes one camera
  w-prefill     a block of the prefill engines' weight memory during the prefill
  w-expert      a block of the expert engines' weight memory during one denoising step
  w-expert-all  the same in all 10 steps, as after a fault during loading
  kv-write      a block of one KV-cache tensor the prefill hands over, read by all 10 steps
  kv-read       the same block in one step's read only
swfi2.py's activation sites (a-*) lie inside the engines and are not reachable here. Weight blocks
are drawn in proportion to each engine's weight memory (found as in wmem.py) and lie in TensorRT's
own layout of the weights, where swfi2.py draws half row and half column blocks. wmem.weight_map
names the prefill and expert weights. Each fault runs on one observation and is compared with the
fault-free plan from the same inputs and noise. The arm is not used.

Observations: the 8 reference observations (wmem.OBS), logged during rollouts.

usage (AOT venv on the Jetson, from JETSON_DIR):
  trt_blocksim.py OUT.npz [--faults N] [--plan w-vision,...] [--seed S]
  Table 6 (TRT): --faults 500 --seed 1, i.e. 8 x 500 = 4,000 faults per source
"""

import argparse
import collections
import re
import time

import numpy as np

import swfi
import wmem
from wmem import BLOCK, d2h, h2d

SITES = ("w-vision", "w-prefill", "w-expert", "w-expert-all", "kv-write", "kv-read")
ENGINES = {"vision": ("vision",), "prefill": ("prefill", "state_proj"), "expert": tuple(wmem.EXPERT)}
NAMED = {"prefill": "smolvlm_expert_prefill.onnx", "state_proj": "state_projector.onnx", **wmem.EXPERT}
CHECK_EVERY = 25                         # faults between fault-free re-runs that prove the restore


def kind_of(name, engine=""):
    """swfi2.py's short type and layer of a weight; the expert's K/V projections in the prefill are x_*."""
    if not name:
        return "", -1
    m = re.search(r"layers\.(\d+)\.", name)
    parts = [p for p in name.split(".") if p not in ("weight", "bias")]
    kind = parts[-1] + (".bias" if name.endswith(".bias") else "")
    if engine == "prefill" and "lm_expert" in name and kind.split(".")[0] in ("k_proj", "v_proj"):
        kind = "x_" + kind
    return kind, int(m.group(1)) if m else -1


def itemsize_at(wmap, ri, off):
    for r, s, e, *_, isz in wmap:
        if r == ri and s <= off < e:
            return isz
    return 2                             # unnamed bytes: the engines are fp16


def plant(ptr, new):
    old = d2h(ptr, BLOCK)
    h2d(ptr, new)
    return old


def restore(ptr, old):
    h2d(ptr, old)
    if not np.array_equal(d2h(ptr, BLOCK), old):
        raise RuntimeError(f"the block at {ptr:#x} was not restored")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--faults", type=int, default=250, help="per observation and source")
    ap.add_argument("--plan", default=",".join(SITES))
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    plan = a.plan.split(",")
    if set(plan) - set(SITES):
        ap.error(f"unknown sources {set(plan) - set(SITES)}")

    rec = wmem.install_recorder(swfi.AOT_CACHE)
    obs = wmem.load_obs()
    t0 = time.perf_counter()
    chain = swfi.Chain(swfi.load_backend())
    want = {e for es in ENGINES.values() for e in es}
    regions = [(tag, ptr, size) for ptr, (size, tag) in rec.live.items() if tag in want]
    stage_of = {e: st for st, es in ENGINES.items() for e in es}
    pools = {}
    for st in ENGINES:
        idx = [ri for ri, (tag, _, _) in enumerate(regions) if stage_of[tag] == st]
        pools[st] = (idx, np.array([regions[ri][2] for ri in idx], float))
    print(f"engines loaded in {time.perf_counter() - t0:.0f} s; RSS {swfi.check_rss()} MB; weight memory "
          + ", ".join(f"{st} {s.sum() / 1e6:.1f} MB in {len(i)} allocations" for st, (i, s) in pools.items()), flush=True)

    t = time.perf_counter()
    wmap, missing = wmem.weight_map(regions, swfi.BUNDLE, NAMED)
    for st in ("prefill", "expert"):
        idx, sizes = pools[st]
        cov = sum(e - s for r, s, e, *_ in wmap if r in idx)
        print(f"weight map {st}: {cov / sizes.sum():.2%} of the bytes named", flush=True)
    print(f"weight map: {len(wmap)} entries, {len(missing)} initializers not found, {time.perf_counter() - t:.0f} s; "
          f"RSS {swfi.check_rss()} MB", flush=True)
    for m in missing[:12]:
        print("   not found:", m, flush=True)

    rng = np.random.default_rng(a.seed)
    cols = collections.defaultdict(list)
    devs, cleans = [], []
    n = 0
    for oi, (run, step, top, side, state) in enumerate(obs):
        noise = rng.standard_normal((1, chain.b.chunk_size, 32)).astype(np.float32)
        pix, lang, s = chain.inputs(top, side, state)
        embs = chain.vision(pix)
        past, pad = chain.prefill(embs, lang, s)
        clean = chain.chunk(chain.denoise(past, pad, noise))

        def full():
            p_, m_ = chain.prefill(chain.vision(pix), lang, s)
            return chain.chunk(chain.denoise(p_, m_, noise))

        if not np.array_equal(full(), clean):
            raise RuntimeError("the fault-free chain is not deterministic")
        cleans.append(clean)
        kv_names = sorted(past)
        for site in plan:
            for _ in range(a.faults):
                new = rng.bytes(BLOCK)
                idx, ri, off, name, layout, isz = -1, -1, -1, "", "", 2
                if site.startswith("w-"):
                    st = {"w-vision": "vision", "w-prefill": "prefill"}.get(site, "expert")
                    pidx, sizes = pools[st]
                    ri = pidx[int(rng.choice(len(pidx), p=sizes / sizes.sum()))]
                    off = int(rng.integers(regions[ri][2] // BLOCK)) * BLOCK
                    ptr = regions[ri][1] + off
                    name, layout, _ = wmem.locate(wmap, ri, off)
                    if name and "." not in name:              # a projector engine's own "weight" / "bias"
                        name = f"{regions[ri][0]}.{name}"
                    isz = itemsize_at(wmap, ri, off)
                    if site == "w-vision":
                        idx = int(rng.integers(2))
                        old = plant(ptr, new)
                        try:
                            e = chain.p.vision(pix[idx])
                        finally:
                            restore(ptr, old)
                        p2, pad2 = chain.prefill([e if c == idx else embs[c] for c in range(2)], lang, s)
                        x = chain.denoise(p2, pad2, noise)
                    elif site == "w-prefill":
                        old = plant(ptr, new)
                        try:
                            p2, pad2 = chain.prefill(embs, lang, s)
                        finally:
                            restore(ptr, old)
                        x = chain.denoise(p2, pad2, noise)
                    elif site == "w-expert":
                        idx = int(rng.integers(chain.b.num_steps))
                        box = {}

                        def hook(step_, hand, arr, at=idx, ptr=ptr, box=box):
                            if step_ == at and hand == "xt":
                                box["old"] = plant(ptr, new)
                            elif step_ == at and hand == "vel":
                                restore(ptr, box.pop("old"))
                            return arr
                        try:
                            x = chain.denoise(past, pad, noise, hook)
                        finally:
                            if "old" in box:
                                restore(ptr, box["old"])
                    else:
                        old = plant(ptr, new)
                        try:
                            x = chain.denoise(past, pad, noise)
                        finally:
                            restore(ptr, old)
                else:
                    name = kv_names[int(rng.integers(len(kv_names)))]
                    y = past[name].copy()
                    flat = y.reshape(-1).view(np.uint8)
                    off = int(rng.integers(flat.size // BLOCK)) * BLOCK
                    flat[off:off + BLOCK] = np.frombuffer(new, np.uint8)
                    layout, isz = "block", y.itemsize
                    p2 = dict(past)
                    if site == "kv-write":
                        p2[name] = y
                        x = chain.denoise(p2, pad, noise)
                    else:
                        idx = int(rng.integers(chain.b.num_steps))

                        def hook(step_, hand, arr, at=idx, p2=p2, name=name, y=y):
                            if step_ == at and hand == "expin":
                                p2[name] = y          # the decode call of this step unpacks p2 after this hook
                            elif step_ == at and hand == "expout":
                                p2[name] = past[name]
                            return arr
                        x = chain.denoise(p2, pad, noise, hook)
                c = chain.chunk(x)
                cls, dmax = wmem.outcome(c, clean)
                kind, layer = kind_of(name, regions[ri][0] if ri >= 0 else "")
                with np.errstate(invalid="ignore"):
                    vals = np.frombuffer(new, {2: np.float16, 4: np.float32}[isz]).astype(np.float64)
                for k, v in (("obs", oi), ("site", site), ("idx", idx), ("target", name), ("kind", kind),
                             ("layer", layer), ("layout", layout), ("region", ri), ("offset", off), ("isz", isz),
                             ("new_nonfinite", int((~np.isfinite(vals)).sum())), ("outcome", cls), ("dmax", dmax)):
                    cols[k].append(v)
                cols["new"].append(np.frombuffer(new, np.uint8))
                devs.append((c - clean).astype(np.float32))
                n += 1
                if n % CHECK_EVERY == 0:
                    if not np.array_equal(full(), clean):
                        raise RuntimeError(f"after fault {n}: the fault-free plan changed, the engines are not clean")
                    swfi.check_rss()
        cnt = collections.Counter(o for o, ob in zip(cols["outcome"], cols["obs"]) if ob == oi)
        print(f"obs {oi} ({run} step {step}) done: {dict(cnt)}; {time.perf_counter() - t0:.0f} s, "
              f"RSS {swfi.check_rss()} MB", flush=True)
        np.savez(a.out, clean=np.array(cleans), devs=np.array(devs), obs_ids=np.array([f"{r}:{s}" for r, s, *_ in obs]),
                 regions=np.array([f"{t}:{sz}" for t, _, sz in regions]),
                 **{k: np.array(v) for k, v in cols.items()})
    print(f"done: {n} faults -> {a.out}, {time.perf_counter() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
