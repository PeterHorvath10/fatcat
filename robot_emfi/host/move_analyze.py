#!/usr/bin/env python3
"""Moving-arm episodes (fi_v3_run.py --move): every pulse with its outcome and, for changed chunks,
how far the commands the arm executed moved from the fault-free replay.

The fault-free replay reruns the rollout's control loop (merge_chunk with 0.3 old / 0.7 new blending,
pop, 12 deg/tick slew limit) on the session's logged chunks, once with the pulsed chunk swapped for the
worker's fault-free recompute and once with the pulsed chunk itself.
Classes of Section 6.2 (silent = chunk >= 10 deg, inside the joint limits, no step > 10 deg).
Fetches the episode's session logs from the Jetson into out/v3fi/npz first.
usage (from host/): move_analyze.py out/v3fi/prefill_e1 [more episodes...]
"""
import collections
import glob
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
from site_config import JETSON

JOINTS = ("pan", "lift", "elbow", "wrist_flex", "wrist_roll", "gripper")
LIM = [(-120.4, 120.4), (-105, 105), (-97.9, 97.9), (-103.9, 103.9), (-180, 180), (0, 100)]
CHUNK, OLD_W, NEW_W, SLEW = 50, 0.30, 0.70, 12.0          # the rollout's chunk length, blend and slew limit


def replay(npz, swap=None):
    """The rollout's loop on logged chunks. swap = (source_step, chunk) replaces that chunk."""
    d = np.load(npz)
    res_t, res_step, res_chunk = d["res_t"], d["res_step"], d["res_chunk"]
    first = int(np.nonzero(res_step == 0)[0][-1])               # post-nudge first chunk
    loop_cmd_t = d["cmd_t"][d["cmd_t"] > res_t[first]]
    loop_cmd = d["cmd"][d["cmd_t"] > res_t[first]]
    obs_before = d["obs"][d["obs_t"] < loop_cmd_t[0]]
    last = np.asarray(obs_before[-1], np.float32)
    buf = {i: res_chunk[first][i].astype(np.float32) for i in range(CHUNK)}
    arrivals = collections.defaultdict(list)
    for j in range(first + 1, len(res_t)):
        arrivals[int(np.searchsorted(loop_cmd_t, res_t[j]))].append(j)
    out, latest, underruns = [], -1, 0
    for step in range(len(loop_cmd)):
        for j in arrivals.get(step, []):
            src = int(res_step[j])
            chunk = swap[1] if swap is not None and src == swap[0] else res_chunk[j]
            new = {}
            for i in range(CHUNK):
                k = src + i
                if k <= latest:
                    continue
                a = np.asarray(chunk[i], np.float32)
                new[k] = (OLD_W * buf[k] + NEW_W * a).astype(np.float32) if k in buf else a.copy()
            buf = new
        a = buf.pop(step, None)
        if a is None:
            underruns += 1
            a = last
        delta = np.asarray(a, np.float32) - last
        a = (last + np.clip(delta, -SLEW, SLEW)).astype(np.float32)
        out.append(a)
        last, latest = a, step
    return np.array(out), loop_cmd, underruns


def classify(r):
    o = r["outcome"].split("+")[0]
    if o != "CORRUPTED":
        return o
    if r["max_diff"] < 10:
        return "small"
    c = np.asarray(r["chunk"], np.float32)
    inside = all(LIM[j][0] - 0.5 <= c[:, j].min() and c[:, j].max() <= LIM[j][1] + 0.5 for j in range(6))
    return "silent" if inside and np.abs(np.diff(c, axis=0)).max() <= 10 else "erratic"

total = collections.Counter()
for prefix in sys.argv[1:]:
    name = prefix.rsplit("/", 1)[-1]
    subprocess.run(["scp", "-q", f"{JETSON}:~/fi_logs/v3/{name}_s*.npz", "out/v3fi/npz/"], check=False)
    tags = dict(re.findall(r"session (\d+): v3 rollout \d+ s.*?remote log ~/fi_logs/v3/(\S+\.npz)",
                           open(prefix + ".stdout").read()))
    rows = [json.loads(line) for line in open(prefix + ".jsonl")]
    shots = [r for r in rows if "n" in r and "outcome" in r]
    ctrl = [r for r in rows if r.get("control")]
    dirty = sum(bool(x.get("nan") or x.get("max_diff") or x.get("canary_max_diff")) for x in ctrl)
    print(f"== {name}: {len(shots)} pulses, {len(ctrl)} controls ({dirty} dirty)")
    for r in shots:
        cls = classify(r)
        total[cls] += 1
        npz = Path("out/v3fi/npz") / tags.get(str(r["session"]), "-")
        d = np.load(npz)
        t_s = d["req_t"][list(d["req_step"]).index(r["target_src"])] - d["cmd_t"][0] if r["target_src"] in d["req_step"] else float("nan")
        line = f"  pulse {r['n']:2d} at {t_s:5.1f} s  {cls:<11s}"
        if cls not in ("OK", "NAN", "ERROR", "NO_RESULT"):
            k, c = np.asarray(r["clean"], np.float32), np.asarray(r["chunk"], np.float32)
            cf, _, _ = replay(str(npz), swap=(r["target_src"], k))
            ac, _, _ = replay(str(npz), swap=(r["target_src"], c))
            dev = np.abs(ac - cf).max(0)
            line += (f" chunk {r['max_diff']:5.1f} deg, executed command {dev.max():5.1f} deg "
                     f"({JOINTS[int(dev.argmax())]})")
        print(line)
    for f in sorted(glob.glob(f"out/v3fi/npz/{name}_s*.npz")):
        d = np.load(f)
        if len(d["cmd"]):
            print(f"  {Path(f).name}: {d['cmd_t'][-1] - d['cmd_t'][0]:.1f} s of motion, {len(d['res_chunk'])} chunks")
print("all:", dict(total))
