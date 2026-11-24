"""Prompt files: plain text, one prompt per line (indexed by the controller's prompt slot), and the
JailbreakBench behaviours CSV (columns Index, Goal, Target, Behavior, Category, Source).

The 100 harmful JailbreakBench goals in the text file are the Goal column of harmful-behaviors.csv, in
the same order.
"""

from __future__ import annotations

import csv
from pathlib import Path


def load_lines(path: Path, limit: int | None = None) -> list[str]:
    prompts = [line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    if not prompts:
        raise ValueError(f"no prompts found in {path}")
    return prompts[:limit] if limit else prompts


def load_jbb_csv(path: Path) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"no rows in {path}")
    return rows
