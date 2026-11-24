"""Judge label files and the answer index used everywhere in the judging.

Layout (written by analysis.judge.judge): <labels_dir>/<judge model with ':' -> '_'>/<tag>.labels.jsonl, one row per
answer: {"index": ..., "label": REFUSED|UNSAFE|OTHER|UNCLEAR, "raw": <last 200 chars of the judge's reply>}.
Answer index of an EMFI run: prompt_slot * 100 + prompt_repeat (so slot = index // 100).
"""

from __future__ import annotations

import collections
import json
from pathlib import Path

REFUSED, UNSAFE, OTHER, UNCLEAR = "REFUSED", "UNSAFE", "OTHER", "UNCLEAR"
LABELS = (REFUSED, UNSAFE, OTHER)
INDEX_STRIDE = 100     # index = slot * INDEX_STRIDE + repeat (fewer than 100 attempts per prompt)


def answer_index(slot: int, repeat: int) -> int:
    return slot * INDEX_STRIDE + repeat


def slot_of(index: int) -> int:
    return index // INDEX_STRIDE


def judge_dir_name(model: str) -> str:
    return model.replace(":", "_")


def read_jsonl(path: str | Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def load_labels(labels_dir: str | Path, tag: str) -> dict[str, dict[int, str]]:
    """{judge folder name: {answer index: label}} for every judge that labelled <tag>."""
    out = {}
    for f in sorted(Path(labels_dir).glob(f"*/{tag}.labels.jsonl")):
        # some older label files store the index as a string
        out[f.parent.name] = {int(r["index"]): r["label"] for r in read_jsonl(f)}
    if not out:
        raise FileNotFoundError(f"no */{tag}.labels.jsonl under {labels_dir}")
    return out


def by_answer(labels: dict[str, dict[int, str]]) -> dict[int, dict[str, str]]:
    """{answer index: {judge: label}}."""
    out = collections.defaultdict(dict)
    for judge, rows in labels.items():
        for i, lab in rows.items():
            out[i][judge] = lab
    return dict(out)


def short_judge_name(folder: str) -> str:
    """'deepseek-v4.1-flash_cloud' -> 'deepseek', 'glm-5.3_cloud' -> 'glm'."""
    return folder.split("_")[0].split("-v")[0].split("-")[0]
