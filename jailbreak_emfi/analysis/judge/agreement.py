"""Judge agreement table: per labelled answer set, REFUSED/UNSAFE/OTHER counts of each of two judges, the share of
answers both judges label the same, and Cohen's kappa.

usage: PYTHONPATH=<jailbreak_emfi> python -m analysis.judge.agreement LABELS_DIR [TAG ...]
  LABELS_DIR holds one folder per judge with <TAG>.labels.jsonl; default: every tag that both judges labelled.
Used for the judge columns of the wipe table (tab:emfi-judged-outcomes, fig:emfi-judged-outcomes) and the
"the two judges agree on 70-100 % of answers" statement.
"""

from __future__ import annotations

import collections
import sys
from pathlib import Path

from analysis.judge.labels import LABELS, UNCLEAR, load_labels, short_judge_name


def cohen_kappa(a: list[str], b: list[str]) -> float:
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = collections.Counter(a), collections.Counter(b)
    pe = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / n ** 2
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def row(labels_dir: Path, tag: str) -> tuple:
    labels = load_labels(labels_dir, tag)
    if len(labels) != 2:
        raise ValueError(f"{tag}: need exactly two judges, found {sorted(labels)}")
    (ja, la), (jb, lb) = sorted(labels.items())
    common = sorted(set(la) & set(lb))
    a, b = [la[i] for i in common], [lb[i] for i in common]
    counts = ["/".join(str(sum(x == k for x in lab.values())) for k in LABELS) for lab in (la, lb)]
    unclear = sum(x == UNCLEAR for x in a + b)
    return (ja, jb), counts, sum(x == y for x, y in zip(a, b)) / len(common), cohen_kappa(a, b), unclear


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    d = Path(sys.argv[1])
    tags = sys.argv[2:]
    if not tags:
        per_judge = [{f.name[:-len(".labels.jsonl")] for f in j.glob("*.labels.jsonl")} for j in d.iterdir() if j.is_dir()]
        tags = sorted(set.intersection(*per_judge)) if per_judge else []
    header = None
    for t in tags:
        (ja, jb), (ca, cb), agree, kappa, unclear = row(d, t)
        if header is None:
            header = f"{'set':28s} {short_judge_name(ja) + ' R/U/O':>16s} {short_judge_name(jb) + ' R/U/O':>14s}  agree  kappa"
            print(header)
        print(f"{t:28s} {ca:>16s} {cb:>14s}  {agree:5.2f}  {kappa:5.2f}" + (f"  ({unclear} UNCLEAR)" if unclear else ""))


if __name__ == "__main__":
    main()
