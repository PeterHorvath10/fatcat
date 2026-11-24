#!/usr/bin/env python3
"""The canary's fault-free chunk, which fi_worker.py compares every canary run with.

During a live campaign, fi_worker.py runs a fixed canary observation (site_config.CANARY_RUN frames,
CANARY_STATE, noise seed 0) after every reply and compares its chunk with this golden one, which catches
corruption that stays in GPU memory. The chunk depends on the TensorRT engines built on this Jetson, so
capture it once after building them, with no pulses. Runs the rollout's device chain (graph) twice and
checks that it is deterministic.

usage (AOT venv on the Jetson): capture_golden.py [OUT.npy]     (default: site_config.GOLDEN)
"""
import sys
from pathlib import Path

import numpy as np

import swfi
from site_config import CANARY_RUN, CANARY_STATE, GOLDEN


def main():
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else GOLDEN
    backend = swfi.load_backend(chain="graph")
    import cv2
    from bench.obs import Observation

    images = [cv2.cvtColor(cv2.imread(str(CANARY_RUN / f"first_obs_{c}.png")), cv2.COLOR_BGR2RGB)
              for c in ("top", "side")]
    noise = np.random.default_rng(0).standard_normal((1, 50, 32)).astype(np.float32)
    obs = Observation(index=0, images=images, state=np.asarray(CANARY_STATE, np.float32), task=swfi.TASK,
                      noise=noise)
    chunk = np.asarray(backend.infer(obs).chunk, dtype=np.float32)
    if not np.array_equal(chunk, np.asarray(backend.infer(obs).chunk, dtype=np.float32)):
        raise RuntimeError("the device chain is not deterministic")
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, chunk)
    print(f"golden canary chunk {chunk.shape} -> {out}")


if __name__ == "__main__":
    main()
