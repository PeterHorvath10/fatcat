#!/usr/bin/env python3
"""Worker side of the load campaigns (Section 6.6): fault the expert weights while they load, decide
whether the faulted model is *erratic*, and only then let the arm move.

Two roles:

  --capture GOLDEN.npz
      Clean load (no pulse). Run the eight reference observations through a fault-free backend and save
      their action chunks as the golden reference. Run once before a campaign.

  (wrapper, spawned by arm_load_fi.py as the rollout's TensorRT worker)
      Wraps smolvla_212_v3_rollout_async.py's worker exactly as fi_worker.py does, but:
        * a trt.Runtime shim prints one "ARMLOAD {decode_ready}" line and idles for FI_LOAD_IDLE_MS
          right before the decode engine deserializes, so the host (arm_load_run.py) can place a
          ChipSHOUTER pulse into that host->device weight copy;
        * the same shim installs wmem's recording allocator, so after load the two decode buffers are read
          back and diffed 16 bytes at a time against a clean golden load (FI_LOAD_DECODE_GOLDEN, saved by a
          run with FI_LOAD_CAPTURE_DECODE=1); every changed block is named by weight/layer in the verdict;
        * validate_fixture is forced to PASS, so a faulted load is NOT rejected (the output check is
          out of the path in all load campaigns);
        * an erratic load's faulted blocks (all of them, golden and faulted bytes) are saved to FI_LOAD_FAULT,
          and FI_LOAD_TEST_PLANT=that.npz plants the same fault again into a clean load (no HV);
        * after load the eight reference observations are run through the (possibly faulted) backend and
          each plan is classified against the golden one (wmem.outcome: nan/erratic/small/...). The worst
          class is the verdict, written to FI_LOAD_VERDICT and emitted as "ARMLOAD {verdict}".
      The verdict is the arm gate: arm_load_fi.py drops every motor command until the verdict is
      "erratic". nan loads make the rollout abort (no motion); small/clean loads are relaunched.

No fixture/output check gates the load here; the plan classification only *labels* it. The arm moving is
the separate, deliberate consequence of an erratic label.
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

import swfi
import wmem

TASK = "pick up the red cube and place it in the transparent box"
HERE = Path(__file__).resolve().parent
GOLDEN_NPZ = Path(os.environ.get("FI_LOAD_GOLDEN", HERE / "arm_golden.npz"))
DECODE_GOLDEN = Path(os.environ.get("FI_LOAD_DECODE_GOLDEN", HERE / "arm_decode_golden.npz"))
CAPTURE_DECODE = os.environ.get("FI_LOAD_CAPTURE_DECODE") == "1"   # clean run: save the decode bytes instead
VERDICT_JSON = Path(os.environ.get("FI_LOAD_VERDICT", HERE / "arm_verdict.json"))
IDLE_S = float(os.environ.get("FI_LOAD_IDLE_MS", "300")) / 1e3
DROP_NAN = os.environ.get("FI_LOAD_NAN") == "drop"   # rollout: keep the previous finite plan on a NaN chunk
FAULT_NPZ = os.environ.get("FI_LOAD_FAULT")         # erratic load: every faulted block is saved here
DECODE_ENGINE = swfi.AOT_CACHE / "decode.engine"
RANK = {"unchanged": 0, "small": 1, "silent": 2, "erratic": 3, "nan": 4}
MAX_BLOCKS = 16                                   # blocks listed per verdict (all are counted)


def emit(d):
    sys.stdout.write("ARMLOAD " + json.dumps(d) + "\n")
    sys.stdout.flush()


def kind(name):
    import re
    m = re.search(r"\.(q|k|v|o|down|gate|up)_proj", name or "")
    return m.group(1) if m else (name or "")


def nonfinite(block16):
    """(count, max|finite value|) of the 16-byte block read as fp16, as the decode weights are stored."""
    v = np.frombuffer(block16.tobytes(), np.float16)
    fin = np.isfinite(v)
    return int((~fin).sum()), (float(np.abs(v[fin]).max()) if fin.any() else float("inf"))


def save_decode_golden(regions):
    """Clean load: the decode buffers' bytes and where each weight lies in them (byte-verified against
    the ONNX initializers, wmem.weight_map)."""
    mems = [wmem.d2h(p, s) for p, s in regions]
    wmap, missing = wmem.weight_map([("decode", p, s) for p, s in regions], swfi.BUNDLE)
    np.savez(DECODE_GOLDEN, sizes=np.array([s for _, s in regions]), wmap=np.array(wmap, object),
             **{f"r{i}": m for i, m in enumerate(mems)})
    covered = sum(w[2] - w[1] for w in wmap) / max(1, sum(s for _, s in regions))
    return {"golden_saved": True, "wmap_entries": len(wmap), "covered": covered}


_gold = {}
_fault = {}                                       # every changed block of this load (label() lists 16)


def label(regions):
    """Which 16-byte blocks of the decode weights this load got wrong, against the clean golden load,
    named by weight and layer."""
    if not _gold:
        z = np.load(DECODE_GOLDEN, allow_pickle=True)
        _gold.update(wmap=list(z["wmap"]), mems=[z[f"r{i}"] for i in range(len(z["sizes"]))])
    blocks, n_bad = [], 0
    _fault.update(region=[], offset=[], old=[], new=[], weight=[], where=[])
    for ri, (p, s) in enumerate(regions):
        cur = wmem.d2h(p, s)
        gold = _gold["mems"][ri] if ri < len(_gold["mems"]) else None
        if gold is None or cur.shape != gold.shape:
            blocks.append({"region": ri, "resized": [None if gold is None else int(gold.size), int(cur.size)]})
            continue
        offs = sorted({int(o) // wmem.BLOCK * wmem.BLOCK for o in np.nonzero(cur != gold)[0]})
        n_bad += len(offs)
        for off in offs:
            _fault["region"].append(ri)
            _fault["offset"].append(off)
            _fault["old"].append(gold[off:off + wmem.BLOCK])
            _fault["new"].append(cur[off:off + wmem.BLOCK])
            name, _, where = wmem.locate(_gold["wmap"], ri, off)
            _fault["weight"].append(name)
            _fault["where"].append(where)
        for off in offs[:MAX_BLOCKS]:
            name, layout, where = wmem.locate(_gold["wmap"], ri, off)
            nf, mx = nonfinite(cur[off:off + wmem.BLOCK])
            blocks.append({"region": ri, "offset": off, "weight": name, "where": where, "nan": nf, "maxabs": mx,
                           "old": gold[off:off + wmem.BLOCK].tolist(), "new": cur[off:off + wmem.BLOCK].tolist()})
    return {"n_bad_blocks": n_bad, "kinds": sorted({kind(b["weight"]) for b in blocks if b.get("weight")}),
            "o_proj_hit": any(".o_proj" in (b.get("weight") or "") for b in blocks), "blocks": blocks}


def save_fault(path, verdict):
    """The faulted blocks of this load (offset, golden and faulted bytes, weight), so the exact fault can
    be planted again later (FI_LOAD_TEST_PLANT=path.npz) without a pulse."""
    f = _fault
    np.savez(path, region=np.array(f["region"], np.int64), offset=np.array(f["offset"], np.int64),
             old=np.array(f["old"], np.uint8).reshape(-1, wmem.BLOCK),
             new=np.array(f["new"], np.uint8).reshape(-1, wmem.BLOCK),
             weight=np.array(f["weight"]), where=np.array(f["where"]), verdict=json.dumps(verdict))
    return len(f["offset"])


def plant(spec, rec):
    """No-HV test: write a fault into the decode weights right after their copy. spec is either
    "region:offset:hex" (one block) or the path of a save_fault() file (all its blocks)."""
    regs = [(p, s) for p, (s, tag) in rec.live.items() if tag == "decode"]
    if spec.endswith(".npz"):
        z = np.load(spec)
        for ri, off, new in zip(z["region"], z["offset"], z["new"]):
            wmem.h2d(regs[int(ri)][0] + int(off), new.tobytes())
        return f"{spec} ({len(z['offset'])} blocks)"
    ri, off, hx = spec.split(":")
    wmem.h2d(regs[int(ri)][0] + int(off), bytes.fromhex(hx))
    return spec


def ref_observations(chunk_size, action_width=32):
    """The eight wmem reference observations as Observation objects with a fixed per-index noise draw,
    so the golden capture and every faulted run denoise from identical noise."""
    from bench.obs import Observation
    obs = []
    for i, (run, step, top, side, state) in enumerate(wmem.load_obs()):
        noise = np.random.default_rng(1000 + i).standard_normal((1, chunk_size, action_width)).astype(np.float32)
        obs.append((run, step, Observation(index=i, images=[top, side], state=np.asarray(state, np.float32),
                                            task=TASK, noise=noise)))
    return obs


def plans(backend):
    cs = int(backend.bundle.info["chunk_size"])
    out = []
    for run, step, o in ref_observations(cs):
        chunk = np.asarray(backend.infer(o).chunk, np.float64)
        if chunk.ndim == 3:                                            # (1, chunk_size, action_dim)
            chunk = chunk[0]
        out.append(chunk[:, :6])                                       # (chunk_size, 6) degrees
    return np.array(out)


def classify(backend):
    """Worst wmem.outcome of the eight reference plans against the golden ones."""
    gold = np.load(GOLDEN_NPZ)["chunks"]
    cur = plans(backend)
    outs = [wmem.outcome(c, g) for c, g in zip(cur, gold)]
    labels = [o for o, _ in outs]
    worst = max(labels, key=lambda o: RANK.get(o, 0))
    dmax = max((d for _, d in outs if np.isfinite(d)), default=float("nan"))
    nan_any = any(not np.isfinite(c).all() for c in cur)
    return {"kind": "verdict", "outcome": worst, "per_obs": labels, "max_dev_deg": float(dmax),
            "nan": bool(nan_any), "t": time.time()}


# ---- role 1: clean golden capture -----------------------------------------------------------------
def capture(path):
    backend = swfi.load_backend(chain="host")          # clean load, parity passes normally
    cur = plans(backend)
    np.savez(path, chunks=cur)
    emit({"kind": "golden", "path": str(path), "n": len(cur),
          "finite": bool(np.isfinite(cur).all()), "t": time.time()})


# ---- role 2: rollout worker wrapper with a load-time pulse -----------------------------------------
def install_patches():
    """A trt.Runtime that idles before the decode deserialize (so the host can pulse the copy), and a
    validate_fixture forced to PASS so the faulted load is kept, not rejected."""
    sys.path.insert(0, str(swfi.REPO))                     # bench.* lives in the deployed repo
    sys.path.insert(0, str(swfi.REPO / "bench/aot/smolvla"))
    from runtime_memory import install_lean_tokenizer
    install_lean_tokenizer()
    import tensorrt as trt
    real_runtime = trt.Runtime
    decode_size = DECODE_ENGINE.stat().st_size
    fired = {"done": False}
    rec = wmem.Recorder(trt)                              # recording allocator (wmem.py):
    names = {p.stat().st_size: p.stem for p in swfi.AOT_CACHE.glob("*.engine")}   # where decode's weights live

    class Runtime:
        def __init__(self, logger):
            self.rt = real_runtime(logger)
            self.rt.gpu_allocator = rec.alloc

        def deserialize_cuda_engine(self, blob):
            rec.tag = names.get(len(blob), f"{len(blob)} B")
            try:
                if not fired["done"] and len(blob) == decode_size:
                    fired["done"] = True
                    emit({"kind": "decode_ready", "t": time.time()})  # host fires during the idle below
                    time.sleep(IDLE_S)
                    t0 = time.perf_counter()
                    eng = self.rt.deserialize_cuda_engine(blob)       # the decode weight copy
                    fired["deser_ms"] = (time.perf_counter() - t0) * 1e3
                    spec = os.environ.get("FI_LOAD_TEST_PLANT")       # no-HV test: hex block or fault file
                    if spec:
                        fired["planted"] = plant(spec, rec)
                    return eng
                return self.rt.deserialize_cuda_engine(blob)
            finally:
                rec.tag = None

        def __getattr__(self, k):
            return getattr(self.rt, k)

    trt.Runtime = Runtime

    # TrtSplitSmolVLABackend.load() imports validate_fixture from bench.vendor.smolvla_trt inside the
    # function, so it must be replaced there (replacing it on trt_split_smolvla had no effect). With the
    # graph chain, load() then also compares the device chain with the fixture's stored actions against
    # these thresholds (_load_device), so they are made permissive too: a faulted load is kept, not rejected.
    from bench.vendor import smolvla_trt
    smolvla_trt.validate_fixture = lambda *a, **k: {"status": "PASS", "threshold": -2.0,
                                                    "max_pct_range_threshold": 1e12,
                                                    "note": "load campaign: output check bypassed"}
    from bench.backends import trt_split_smolvla as tss

    # --nan-drop: while the rollout runs (not during the classification above), a NaN/Inf chunk is not
    # passed on -- the worker script would abort on it -- but replaced by the previous finite chunk shifted
    # to the new source step, so the arm keeps following the previous plan (as fi_worker.py FI_V3_NAN=drop).
    from types import SimpleNamespace
    orig_infer = tss.TrtSplitSmolVLABackend.infer
    serving = {"on": False, "dropped": 0}
    last_good = {}

    def infer(self, obs):
        result = orig_infer(self, obs)
        if not serving["on"]:
            return result
        chunk = np.array(result.chunk, dtype=np.float32, copy=True)
        if np.isfinite(chunk).all():
            last_good.update(source_step=int(obs.index), chunk=chunk)
            return result
        if DROP_NAN and last_good and int(obs.index) >= last_good["source_step"]:
            serving["dropped"] += 1
            shift = int(obs.index) - last_good["source_step"]
            keep = last_good["chunk"][np.minimum(np.arange(len(chunk)) + shift, len(chunk) - 1)]
            print(f"[arm_load_worker] NaN chunk at source {obs.index} dropped (#{serving['dropped']}): previous "
                  f"plan (source {last_good['source_step']}, shifted {shift}) kept", flush=True)
            return SimpleNamespace(chunk=keep, timings=getattr(result, "timings", {}))
        return result

    tss.TrtSplitSmolVLABackend.infer = infer
    orig_load = tss.TrtSplitSmolVLABackend.load

    def load(self):
        orig_load(self)                                               # pulse lands in here; parity bypassed
        regions = [(p, s) for p, (s, tag) in rec.live.items() if tag == "decode"]   # decode#0, decode#1
        try:                                                          # which weights the load got wrong
            lab = save_decode_golden(regions) if CAPTURE_DECODE else label(regions)
        except Exception as exc:
            lab = {"label_error": f"{type(exc).__name__}: {exc}"[:300]}
        try:
            v = classify(self)
        except Exception as exc:                                     # never block the worker on classify
            v = {"kind": "verdict", "outcome": "unknown", "error": f"{type(exc).__name__}: {exc}"[:300],
                 "t": time.time()}
        v.update(lab)
        v["decode_regions_mb"] = [round(s / 1e6, 2) for _, s in regions]
        v["decode_deser_ms"] = fired.get("deser_ms")                 # how long the decode copy took here
        if fired.get("planted"):
            v["test_planted"] = fired["planted"]
        v["nan_drop"] = DROP_NAN
        if FAULT_NPZ and v.get("outcome") == "erratic" and _fault.get("offset"):
            try:                                                      # keep the faulted weights
                v["fault_saved"] = {"path": FAULT_NPZ, "blocks": save_fault(FAULT_NPZ, v)}
            except Exception as exc:
                v["fault_saved"] = {"error": f"{type(exc).__name__}: {exc}"[:300]}
        serving["on"] = True                                         # the rollout's own requests from here on
        try:
            VERDICT_JSON.write_text(json.dumps(v))
        except OSError:
            pass
        emit(v)

    tss.TrtSplitSmolVLABackend.load = load


def main():
    argv = sys.argv[1:]
    if argv and argv[0] == "--capture":
        capture(argv[1] if len(argv) > 1 else GOLDEN_NPZ)
        return

    # wrapper mode: argv is [SCRIPT, --worker, --fd, N, ...] (rewritten by arm_load_fi.py's popen hook)
    import runpy
    install_patches()
    script = argv[0]
    sys.argv = argv
    runpy.run_path(script, run_name="__main__")


if __name__ == "__main__":
    main()
