#!/usr/bin/env python3
"""Cut the 8 reference observations out of the logged rollouts, for ../data/traces.

The simulation (swfi2.py) and the Jetson-side classifications (trt_blocksim.py, arm_load_worker.py, via wmem.py)
read the camera images and joint state of 8 inference requests from logged rollouts (swfi2.OBS: run,
request step). A rollout's trace.npz holds every request of a 20 s episode (~12 MB); this keeps only
the requests in OBS, in the same format (req_step, req_top, req_side, req_state), one file per run.

usage: extract_observations.py ROLLOUT_LOGS OUT_DIR   (ROLLOUT_LOGS/RUN/trace.npz -> OUT_DIR/RUN/trace.npz)
"""
import ast
import sys
from pathlib import Path

import numpy as np

KEYS = ("req_step", "req_top", "req_side", "req_state")


def obs_list():
    """swfi2.OBS, read from the source so this runs without torch or LeRobot."""
    src = (Path(__file__).resolve().parent / "swfi2.py").read_text()
    start = src.index("OBS = [")
    end = src.index("]", start) + 1
    return ast.literal_eval(src[start + len("OBS = "):end])


def main():
    logs, out = Path(sys.argv[1]), Path(sys.argv[2])
    obs = obs_list()
    for run in dict.fromkeys(r for r, _ in obs):
        steps = [s for r, s in obs if r == run]
        d = np.load(logs / run / "trace.npz")
        # the first request of a step, as the loaders take it (step 0 is logged twice: start-up and first re-plan)
        keep = [int(np.flatnonzero(d["req_step"] == s)[0]) for s in steps]
        (out / run).mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out / run / "trace.npz", **{k: d[k][keep] for k in KEYS})
        print(f"{run}: requests {[int(s) for s in d['req_step'][keep]]}")


if __name__ == "__main__":
    main()
