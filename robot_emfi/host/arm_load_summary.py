#!/usr/bin/env python3
"""Load-time EMFI on the live rollout (arm_load_run.py): one row per campaign and a pooled row.

Per campaign: pulses fired, loads that crashed (no reply after a fired pulse), loads the pulse corrupted
(at least one 16-byte block of the expert's weights read back wrong), and the plan class of each
corrupted load on the eight reference observations (worst class: nan = detected, erratic, silent,
small). Launches that never fired (e.g. aborted at robot connect) are left out. For every corrupted
load it lists the weights hit, the number of garbled blocks and the largest plan change.

usage: arm_load_summary.py RUN.jsonl [RUN.jsonl ...]
"""
import ast
import collections
import json
import sys
from pathlib import Path

CLASSES = ("nan", "silent", "erratic", "small")


def rows(path):
    return [json.loads(line) for line in open(path) if line.strip()]


def summary(name, shots):
    fired = [r for r in shots if r.get("emfi") == "FIRED"]
    crashed = [r for r in fired if r.get("outcome") == "NO_REPLY"]
    corrupt = [r for r in fired if (r.get("n_bad_blocks") or 0) > 0]
    cls = collections.Counter(r.get("outcome") for r in corrupt)
    print(f"{name:24s} {len(fired):6d} {len(crashed):7d} {len(corrupt):9d}  "
          + "  ".join(f"{cls[c]:7d}" for c in CLASSES))
    return corrupt


def main():
    print(f"{'campaign':24s} {'pulses':>6s} {'crashed':>7s} {'corrupted':>9s}  "
          + "  ".join(f"{c:>7s}" for c in CLASSES))
    pooled, hits = [], []
    for path in sys.argv[1:]:
        shots = [r for r in rows(path) if "attempt" in r]
        hits += [(Path(path).stem, r) for r in summary(Path(path).stem, shots)]
        pooled += shots
    if len(sys.argv) > 2:
        summary("pooled", pooled)
    for name, r in hits:
        blocks = r.get("blocks")
        blocks = ast.literal_eval(blocks) if isinstance(blocks, str) else blocks or []
        weights = sorted({b["weight"] for b in blocks})
        print(f"  {name} attempt {r['attempt']}: delay {r.get('delay_ms')} ms, {r['n_bad_blocks']} blocks in "
              f"{', '.join(weights) or r.get('kinds')}, {r.get('outcome')}, max plan change {r.get('max_dev_deg')}")


if __name__ == "__main__":
    main()
