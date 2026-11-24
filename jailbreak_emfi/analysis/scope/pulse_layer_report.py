"""Per-run report: in which layer burst does the EM pulse land?

For every scope trace of a run: the layer bursts, whether they form a regular layer sequence, and the burst that
contains the pulse. Prints the totals (e.g. "pulse in burst L8 for 1801 of 1828 regular traces") and writes one CSV
row per trace.

usage: PYTHONPATH=<jailbreak_emfi> python -m analysis.scope.pulse_layer_report RUN_DIR OUT_CSV [--cache DIR]
  RUN_DIR: an EMFI run folder (scope/tNNNNN.npz, results.jsonl, controller.log); read only.
  --cache: keep the per-trace summaries there (must not be inside RUN_DIR), so a second pass is fast.
"""

from __future__ import annotations

import argparse
import collections
import csv
from pathlib import Path

from common import config
from analysis.scope import trace as st

FIELDS = ["trace", "trial_index", "prompt_slot", "prompt_repeat", "glitch_voltage", "primary_duration",
          "n_bursts", "regular", "hit_burst", "pulse_ms", "pulse_peak_mv", "where", "detail"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("out_csv", type=Path)
    ap.add_argument("--cache", type=Path)
    a = ap.parse_args()
    if a.cache and a.run_dir.resolve() in a.cache.resolve().parents:
        ap.error("--cache must not be inside the run folder (run folders are read only)")

    cfg = config.load("analysis.toml")
    bp, pp = st.BurstParams.from_config(cfg), st.PulseParams.from_config(cfg)
    rule = st.LayerRule.for_model(cfg, st.served_model(a.run_dir))

    n = 0
    cat = collections.Counter()
    hits = collections.Counter()     # regular traces: burst index of the pulse (None = not in a burst)
    a.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(a.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, FIELDS)
        w.writeheader()
        for s in st.run_summaries(a.run_dir, bp, pp, cfg.get("scope.workers", int), a.cache):
            b, regular, hit = st.layer_info(s, bp, rule)
            where, detail = st.describe_hit(b, regular, hit, s.tp, rule)
            n += 1
            cat[where] += 1
            if regular:
                hits[hit] += 1
            m = s.meta
            w.writerow(dict(trace=s.name, trial_index=m.get("trial_index"), prompt_slot=m.get("prompt_slot"),
                            prompt_repeat=m.get("prompt_repeat"), glitch_voltage=m.get("glitch_voltage"),
                            primary_duration=m.get("primary_duration"), n_bursts=len(b), regular=int(regular),
                            hit_burst="" if hit is None else hit, pulse_ms=f"{s.tp * 1e3:.4f}",
                            pulse_peak_mv=f"{s.peak * 1e3:.1f}", where=where, detail=detail))

    n_reg = sum(hits.values())
    print(f"{a.run_dir.name}: {n} traces, model {rule.name}")
    print("  " + ", ".join(f"{k} {v}" for k, v in cat.most_common()))
    print(f"  regular layer sequences: {n_reg}")
    in_burst = {k: v for k, v in hits.items() if k is not None}
    if in_burst:
        top, top_n = max(in_burst.items(), key=lambda kv: kv[1])
        print(f"  pulse in burst L{top} for {top_n} of {n_reg} regular traces "
              f"({100 * top_n / max(n_reg, 1):.1f} %; {100 * top_n / n:.1f} % of all traces)")
    print("  per burst (regular traces): " + ", ".join(
        f"{'none' if k is None else 'L%d' % k}: {v}" for k, v in sorted(hits.items(), key=lambda kv: (kv[0] is None, kv[0] or 0))))
    print(f"wrote {a.out_csv}")


if __name__ == "__main__":
    main()
