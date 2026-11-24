"""Per-prompt timing of one layer's MLP burst during prefill, from the scope traces of a run.

For every prompt slot: the start of burst L<layer> relative to the scope trigger over that prompt's regular traces
(median, 10th/90th percentile, standard deviation), the median burst end, and the fraction of traces whose whole
layer sequence starts late (first burst after timing.late_start_s; the GPU sometimes starts the prefill ~0.35 ms
later).

usage: PYTHONPATH=<jailbreak_emfi> python -m analysis.scope.layer_timing_table RUN_DIR OUT_CSV [--cache DIR]
The prompt text is deliberately not written (no harmful prompts in outputs); join on the prompt slot if needed.
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
from pathlib import Path

import numpy as np

from common import config
from analysis.scope import trace as st


def prompt_tokens(run_dir: Path) -> dict[int, int]:
    out = {}
    with open(run_dir / "results.jsonl") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                out.setdefault(int(r["prompt_slot"]), int(r["usage"]["prompt_tokens"]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("out_csv", type=Path)
    ap.add_argument("--cache", type=Path)
    a = ap.parse_args()

    cfg = config.load("analysis.toml")
    bp = st.BurstParams.from_config(cfg)
    rule = st.LayerRule.for_model(cfg, st.served_model(a.run_dir))
    layer = cfg.get("timing.layer", int)
    late_s = cfg.get("timing.late_start_s", float)
    min_bursts = cfg.get("timing.min_bursts", int)

    starts, ends, first = (collections.defaultdict(list) for _ in range(3))
    # pulse position is not needed here, so no PulseParams (summaries are cached without a pulse time)
    for s in st.run_summaries(a.run_dir, bp, None, cfg.get("scope.workers", int), a.cache):
        b = st.find_bursts(s.rms, s.t0, bp)
        if not st.is_regular(b, rule) or len(b) < max(min_bursts, layer + 1):
            continue
        p = int(s.meta["prompt_slot"])
        starts[p].append(b[layer][0] * 1e3)
        ends[p].append(b[layer][1] * 1e3)
        first[p].append(b[0][0])

    toks = prompt_tokens(a.run_dir)
    L = f"L{layer}"
    rows = []
    for p in sorted(starts):
        x = np.array(starts[p])
        start = float(np.median(x))
        r = {"prompt": p, "prompt_tokens": toks.get(p, ""), "scope_traces": len(x)}
        r[f"{L}_start_ms"] = round(start, 3)
        r |= {"scope_burst_end_ms": round(float(np.median(ends[p])), 3),
              f"{L}_start_p10_ms": round(float(np.percentile(x, 10)), 3),
              f"{L}_start_p90_ms": round(float(np.percentile(x, 90)), 3),
              f"{L}_start_std_ms": round(float(x.std()), 3),
              "late_start_frac": round(float(np.mean(np.array(first[p]) > late_s)), 3)}
        rows.append(r)

    a.out_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(a.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fields)
        w.writeheader()
        w.writerows(rows)
    all_starts = np.concatenate([np.array(v) for v in starts.values()]) if starts else np.array([])
    if len(all_starts):
        meds = [float(np.median(v)) for v in starts.values()]
        print(f"{L} burst start: median {np.median(all_starts):.2f} ms over {len(all_starts)} regular traces; "
              f"per-prompt medians {min(meds):.2f}-{max(meds):.2f} ms; "
              f"mean within-prompt sd {np.mean([np.std(v) for v in starts.values()]):.2f} ms")
    print(f"wrote {a.out_csv} ({len(rows)} prompts)")


if __name__ == "__main__":
    main()
