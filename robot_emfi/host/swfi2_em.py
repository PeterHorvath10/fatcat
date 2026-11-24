#!/usr/bin/env python3
"""EM pulses against the block-fault simulation, per stage: silent share (silent / (silent + small)),
median silent plan change, NaN share of changed inferences and plans changed by >= 100 deg. EM pulses
of all ChipSHOUTER policy runs at the given voltages, placed in the inference by their probe position
(pos_analyze.load); simulated weight and activation blocks from swfi2.py.
usage (from host/): swfi2_em.py out/swfi2/swfi2.npz out/v3fi/RUN [...] [--vmin 300 --vmax 400] [--moving]
"""
import argparse
import collections
import json

import numpy as np

from pos_analyze import load

STAGES = [(1.3, 61.0, "vision encoders", "vision", None), (61.0, 63.25, "prefill layers 0-3", "prefill", (0, 4)),
          (63.25, 65.5, "prefill layers 4-7", "prefill", (4, 8)), (65.5, 67.75, "prefill layers 8-11", "prefill", (8, 12)),
          (67.75, 70.0, "prefill layers 12-15", "prefill", (12, 16))]


def moved(prefix):
    """True if the run's console shows the arm moving (stationary runs drop the Goal_Position writes)."""
    for ext in (".console", ".stdout"):
        try:
            if "ARM MOVES" in open(prefix + ext, errors="replace").read():
                return True
        except OSError:
            pass
    return False


def row(cls, dm):
    """(changed, NaN share of changed, silent share, silent median, >= 100 deg plans) of one group."""
    c = collections.Counter(cls)
    changed = len(cls) - c["OK"] - c["recompute hit"] - c["unchanged"]
    sil = dm[cls == "silent"]
    big = int(((dm >= 100) & np.isin(cls, ("silent", "erratic"))).sum())
    nan = c["NAN"] + c["nan"]
    return (changed, nan / max(changed, 1), len(sil) / max(len(sil) + c["small"], 1),
            np.median(sil) if len(sil) else float("nan"), big)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--vmin", type=float, default=300)
    ap.add_argument("--vmax", type=float, default=400)
    ap.add_argument("--moving", action="store_true",
                    help="only runs in which the arm moved (fi_v3.py printed ARM MOVES to the console)")
    a = ap.parse_args()
    shots = []
    for prefix in a.runs:
        rows = [json.loads(line) for line in open(prefix + ".jsonl")]
        if not any(r.get("triggered") for r in rows):
            continue
        if a.moving and not moved(prefix):
            continue
        try:
            shots += [r for r in load(prefix)[1] if a.vmin <= (r.get("voltage") or 0) <= a.vmax]
        except TypeError:                   # runs without a probe end (trigger checks)
            continue
    d = dict(np.load(a.npz))
    print(f"{len(shots)} EM pulses at {a.vmin:.0f}-{a.vmax:.0f} V; canary failures: "
          f"{sum(bool(r.get('canary_max_diff')) for r in shots)}")
    print(f"{'stage':<22s} {'source':<12s} {'n':>6s} {'changed':>7s} {'NaN/ch':>6s} {'share':>5s} {'med':>4s} {'>=100':>5s} {'%ch':>5s}")
    for lo, hi, label, st, layers in STAGES:
        sel = [r for r in shots if lo <= r["pos"] < hi]
        groups = [("EM", np.array([r["cls"] for r in sel]), np.array([r.get("max_diff") or 0.0 for r in sel]))]
        for src in ("w", "a"):
            m = d["site"] == f"{src}-{st}"
            if layers:
                m &= (d["layer"] >= layers[0]) & (d["layer"] < layers[1]) & ~np.char.startswith(d["kind"], "x_")
            groups.append(({"w": "weights", "a": "activations"}[src], d["outcome"][m], np.nan_to_num(d["dmax"][m])))
        for src, cls, dm in groups:
            ch, nan, share, med, big = row(cls, dm)
            print(f"{label:<22s} {src:<12s} {len(cls):6d} {ch:7d} {nan:6.2f} {share:5.2f} {med:4.0f} {big:5d} {100 * big / max(ch, 1):5.1f}")


if __name__ == "__main__":
    main()
