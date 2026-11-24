#!/usr/bin/env python3
"""TensorRT worker wrapper for EMFI on a live rollout: the worker script runs unchanged,
and after each reply has been sent (so the rollout's timing is untouched) it

  1. waits --gap ms (a quiet GPU gap that marks the end of the real inference on the probe),
  2. recomputes the same observation (same images, state, noise): the fault-free reference,
  3. runs a fixed canary observation and compares it with the harness golden chunk
     (catches persistent corruption, which step 2 alone would hide),

then prints one line "FIV3 {json}" to stdout. Errors (NaN chunk, CUDA failure) are
reported the same way before the worker script sends them to the parent.

FI_V3_NAN=drop: a NaN/Inf chunk is not passed on (the worker script would exit on it); the
rollout gets the previous finite chunk shifted to the new source step instead, so the arm keeps
following the previous plan. The FIV3 line still reports the real (NaN) chunk.

usage (spawned by fi_v3.py): fi_worker.py SCRIPT --worker --fd N --seed S
"""

import json
import os
import runpy
import sys
import time
from multiprocessing.connection import Connection
from types import SimpleNamespace
from pathlib import Path

import numpy as np

from site_config import CANARY_RUN, CANARY_STATE, GOLDEN
from site_config import VLA_REPO as REPO
TASK = "pick up the red cube and place it in the transparent box"
GAP_S = float(os.environ.get("FI_VERIFY_GAP_MS", "30")) / 1e3
FIRST_USED = 4  # stale = 4 in every v3 merge: chunk[4] is the first action executed
DROP_NAN = os.environ.get("FI_V3_NAN") == "drop"

sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "bench/aot/smolvla"))
from runtime_memory import install_lean_tokenizer  # noqa: E402

install_lean_tokenizer()
from bench.backends.trt_split_smolvla import TrtSplitSmolVLABackend  # noqa: E402
from bench.obs import Observation  # noqa: E402

orig_infer = TrtSplitSmolVLABackend.infer
orig_send = Connection.send
pending = {}
counter = {"n": 0}
canary = {}
last_good = {}


def emit(line):
    sys.stdout.write("FIV3 " + json.dumps(line) + "\n")
    sys.stdout.flush()


def load_frames(run):
    import cv2

    return [cv2.cvtColor(cv2.imread(str(Path(run) / f"first_obs_{c}.png")), cv2.COLOR_BGR2RGB) for c in ("top", "side")]


def canary_obs():
    if "obs" not in canary:
        images = load_frames(CANARY_RUN)
        noise = np.random.default_rng(0).standard_normal((1, 50, 32)).astype(np.float32)
        canary["obs"] = Observation(index=0, images=images, state=np.asarray(CANARY_STATE, np.float32),
                                    task=TASK, noise=noise)
        canary["golden"] = np.load(GOLDEN)
    return canary["obs"]


def infer(self, obs):
    pending.clear()
    pending.update(backend=self, obs=obs, source_step=int(obs.index), t0=time.perf_counter())
    result = orig_infer(self, obs)
    chunk = np.array(result.chunk, dtype=np.float32, copy=True)
    pending["chunk"] = chunk
    pending["infer_ms"] = (time.perf_counter() - pending["t0"]) * 1e3
    if np.isfinite(chunk).all():
        last_good.update(source_step=int(obs.index), chunk=chunk)
    elif DROP_NAN and last_good and int(obs.index) >= last_good["source_step"]:
        shift = int(obs.index) - last_good["source_step"]
        keep = last_good["chunk"][np.minimum(np.arange(len(chunk)) + shift, len(chunk) - 1)]
        print(f"[fi_worker] NaN chunk at source {obs.index} dropped: previous plan (source "
              f"{last_good['source_step']}, shifted {shift}) kept", flush=True)
        return SimpleNamespace(chunk=keep)
    return result


def verify(kind):
    n = counter["n"]
    counter["n"] += 1
    line = {"n": n, "kind": kind, "source_step": pending.get("source_step"), "t_wall": time.time(),
            "infer_ms": pending.get("infer_ms")}
    chunk = pending.get("chunk")
    if chunk is None:                       # infer() itself raised (CUDA / TensorRT error)
        line["error"] = "infer raised"
        emit(line)
        return
    time.sleep(GAP_S)
    clean = np.asarray(orig_infer(pending["backend"], pending["obs"]).chunk, dtype=np.float32)
    can = np.asarray(orig_infer(pending["backend"], canary_obs()).chunk, dtype=np.float32)
    diff = np.abs(chunk - clean)
    finite = bool(np.isfinite(chunk).all())
    big = np.nan_to_num(diff, nan=1e9, posinf=1e9)
    line.update(
        nan=not finite,
        max_diff=float(big.max()),
        n_diff=int((big > 1e-3).sum()),
        used_max_diff=float(big[FIRST_USED:].max()),
        first_used_diff=float(big[FIRST_USED].max()),
        clean_finite=bool(np.isfinite(clean).all()),
        canary_max_diff=float(np.nan_to_num(np.abs(can - canary["golden"]), nan=1e9).max()),
    )
    if line["max_diff"] > 0:
        line["chunk"] = np.nan_to_num(chunk, nan=1e9, posinf=1e9, neginf=-1e9).round(4).tolist()
        line["clean"] = clean.round(4).tolist()
    emit(line)


def safe_verify(kind):
    try:
        verify(kind)
    except Exception as exc:                # e.g. the CUDA context is gone: report, never raise
        emit({"n": counter["n"], "kind": kind, "source_step": pending.get("source_step"),
              "t_wall": time.time(), "verify_error": f"{type(exc).__name__}: {exc}"[:500]})
    finally:
        pending.clear()


def send(self, msg):
    kind = msg.get("type") if isinstance(msg, dict) else None
    if kind == "error":                     # report before the worker script exits
        safe_verify("error")
    orig_send(self, msg)
    if kind == "result" and "chunk" in pending:
        safe_verify("result")


TrtSplitSmolVLABackend.infer = infer
Connection.send = send

script = sys.argv[1]
sys.argv = sys.argv[1:]
runpy.run_path(script, run_name="__main__")
