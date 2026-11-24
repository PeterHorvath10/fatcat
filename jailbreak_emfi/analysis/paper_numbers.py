"""Recompute the paper's data-derived numbers in one go (inputs: [paper] in config/analysis.toml).

For every run in paper.runs: the judged counts (analysis.judge.count, with the baseline column if
paper.baseline_labels has a folder for the run and the first-K counts of paper.first_k) and, with --scope, the pulse-in-layer report
(analysis.scope.pulse_layer_report; reads every raw trace, minutes per run). Then the judge-agreement table of
paper.rejudge_labels (analysis.judge.agreement). Everything is written to paper.out_dir; run folders are only read.

usage: PYTHONPATH=<jailbreak_emfi> python -m analysis.paper_numbers [--scope]
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from common import config


def run(args: list[str], log: Path) -> None:
    print("$", " ".join(args), flush=True)
    out = subprocess.run([sys.executable, "-m", *args], capture_output=True, text=True)
    log.write_text(out.stdout + out.stderr)
    print(out.stdout + out.stderr, flush=True)
    if out.returncode:
        raise SystemExit(f"failed: {' '.join(args)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scope", action="store_true", help="also run the pulse-in-layer report (slow)")
    a = ap.parse_args()
    cfg = config.load("analysis.toml")
    out = Path(cfg.get("paper.out_dir"))
    out.mkdir(parents=True, exist_ok=True)
    runs = cfg.get("paper.runs")
    bases = cfg.get("paper.baseline_labels", default=[""] * len(runs))   # per run; "" = no baseline (per model!)
    if len(bases) != len(runs):
        raise SystemExit("paper.baseline_labels needs one entry per run in paper.runs (\"\" for none)")
    first_k = cfg.get("paper.first_k", int, default=[])
    for r, base in zip(map(Path, runs), bases):
        args = ["analysis.judge.count", str(r / "judge" / "labels"), "--per-prompt", str(out / f"{r.name}_per_prompt.csv"),
                "--json", str(out / f"{r.name}_counts.json")]
        args += [x for k in first_k for x in ("--first-k", str(k))]
        if base:
            args += ["--baseline", base]
        run(args, out / f"{r.name}_counts.txt")
        if a.scope:
            run(["analysis.scope.pulse_layer_report", str(r), str(out / f"{r.name}_pulse_layer.csv"),
                 "--cache", str(out / "trace_cache" / r.name)], out / f"{r.name}_pulse_layer.txt")
    rej = cfg.get("paper.rejudge_labels", default=None)
    if rej:
        run(["analysis.judge.agreement", rej], out / "judge_agreement.txt")


if __name__ == "__main__":
    main()
