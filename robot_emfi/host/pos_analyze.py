#!/usr/bin/env python3
"""Robot runs by pulse position in the inference (position = delay - trigger-to-end + 112.6 ms profiled
inference length), against reference runs. Classes of Section 6.2.
NaN, GPU-error, snapped and implausible shots take the run median end (see trusted_end()).
usage (from host/): pos_analyze.py out/v3fi/RUN [out/v3fi/REF ...] [--stages | --bin MS]
  --stages: one row per profiled stage; --bin MS: uniform bins of MS ms over the inference
"""
import argparse
import collections
import json
import math

import numpy as np

LIM = [(-120.4, 120.4), (-105, 105), (-97.9, 97.9), (-103.9, 103.9), (-180, 180), (0, 100)]
INFER_MS = 112.6
SNAP_MS = 0.5
MAX_END_MS = 120
DEFAULT_END_MS = 110.8      # trigger at the inference onset; for runs without one trusted end
BINS = [(0, 40), (40, 45), (45, 50), (50, 55), (55, 60), (60, 65), (65, 70), (70, 113)]
STAGES = [(0, 1.3, "image conversion"), (1.3, 31.1, "vision encoder, top"), (31.1, 61.0, "vision encoder, side"),
          (61.0, 70.0, "prefill"), (70.0, INFER_MS, "denoise (10 steps)"), (INFER_MS, 200, "after the inference")]


def classify(r):
    o = r["outcome"].split("+")[0]
    if o != "CORRUPTED":
        return o
    if not np.isfinite(np.array(r["clean"], dtype=float)).all():
        return "recompute hit"           # the pulse missed the inference and hit the fault-free recompute
    if r["max_diff"] < 10:
        return "small"
    c = np.asarray(r["chunk"], np.float32)
    inside = all(LIM[j][0] - 0.5 <= c[:, j].min() and c[:, j].max() <= LIM[j][1] + 0.5 for j in range(6))
    return "silent" if inside and np.abs(np.diff(c, axis=0)).max() <= 10 else "erratic"


def detected_end(r):
    e = r.get("end_ms")
    return e if isinstance(e, (int, float)) and math.isfinite(e) else None


def trusted_end(r):
    """The detected end, unless the pulse changed the probe activity: after a NaN or a GPU error the
    activity fades 0-50 ms after the pulse (likely the NaN data toggle less), and an end within SNAP_MS
    of the pulse is the pulse itself. An end after MAX_END_MS is no end: a late pulse's ChipSHOUTER
    recharge can fill the quiet gap after the inference (2026-11-03, ends ~150 ms at 380 V)."""
    e = detected_end(r)
    if (e is None or r["cls"] in ("NAN", "ERROR", "NO_RESULT") or abs(r["delay_ms"] - e) <= SNAP_MS
            or e > MAX_END_MS):
        return None
    return e


def load(prefix):
    """Shots without a trusted end take the run's median trusted end."""
    rows = [json.loads(line) for line in open(prefix + ".jsonl")]
    shots = [r for r in rows if "n" in r and "outcome" in r and r.get("triggered")]
    for r in shots:
        r["cls"] = classify(r)
    ends = [e for r in shots if (e := trusted_end(r)) is not None]
    median_end = float(np.median(ends)) if ends else DEFAULT_END_MS
    for r in shots:
        e = trusted_end(r)
        e = median_end if e is None else e
        r["pos"] = r["delay_ms"] - e + INFER_MS if e is not None else float("nan")
    return rows, shots


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p, d = k / n, 1 + z * z / n
    m, h = (p + z * z / (2 * n)) / d, z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, m - h), min(1.0, m + h)


def summary(name, rows, shots, bins):
    n, c = len(shots), collections.Counter(r["cls"] for r in shots)
    pos = np.array([r["pos"] for r in shots])
    ctrl = [r for r in rows if r.get("control")]
    dirty = sum(bool(x.get("nan") or x.get("max_diff") or x.get("canary_max_diff")) for x in ctrl)
    (dl, dh), (sl, sh) = wilson(c["NAN"], n), wilson(c["silent"], n)
    print(f"{name}: {n} triggered pulses at {np.nanpercentile(pos, 5):.0f}-{np.nanpercentile(pos, 95):.0f} ms | "
          f"detected {c['NAN']} ({c['NAN'] / n:.1%}, {dl:.0%}-{dh:.0%}) | silent {c['silent']} ({c['silent'] / n:.1%}, "
          f"{sl:.0%}-{sh:.0%}) | small {c['small']} | erratic {c['erratic']} | GPU err {c['ERROR'] + c['NO_RESULT']} | "
          f"recompute hits {c['recompute hit']} | "
          f"controls {len(ctrl)} ({dirty} dirty)")
    for a, b, label in bins:
        sel = [r for r in shots if a <= r["pos"] < b]
        if sel:
            cc = collections.Counter(r["cls"] for r in sel)
            print(f"    {a:5.1f}-{b:5.1f} ms {label:22s}: {len(sel):4d} pulses | detected {cc['NAN'] / len(sel):5.1%} | "
                  f"silent {cc['silent'] / len(sel):5.1%} | small {cc['small'] / len(sel):5.1%}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--stages", action="store_true")
    ap.add_argument("--bin", type=float)
    args = ap.parse_args()
    if args.stages:
        bins = STAGES
    elif args.bin:
        bins = [(a, a + args.bin, "") for a in np.arange(0, 120, args.bin)]
    else:
        bins = [(a, b, "") for a, b in BINS]
    for i, prefix in enumerate(args.runs):
        summary(("RUN " if i == 0 else "ref ") + prefix.rsplit("/", 1)[-1], *load(prefix), bins)
