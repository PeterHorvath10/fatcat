"""Make the judge input of an EMFI run: results.jsonl (one row per saved answer) -> answers.jsonl with
index = prompt_slot * 100 + prompt_repeat, slot, repeat, goal (= prompt) and response (= generated_text).

usage: PYTHONPATH=<jailbreak_emfi> python -m analysis.judge.build_input RESULTS_JSONL OUT_JSONL
(For the paper runs OUT_JSONL was <run>/judge/all.jsonl, labelled with TAG 'all'.)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from analysis.judge.labels import answer_index, read_jsonl


def judge_rows(results: list[dict]) -> list[dict]:
    rows = {}
    for r in results:
        i = answer_index(int(r["prompt_slot"]), int(r["prompt_repeat"]))
        if i in rows:
            raise ValueError(f"two answers for slot {r['prompt_slot']} repeat {r['prompt_repeat']}")
        rows[i] = {"index": i, "slot": int(r["prompt_slot"]), "repeat": int(r["prompt_repeat"]),
                   "goal": r["prompt"], "response": r["generated_text"]}
    return [rows[i] for i in sorted(rows)]


def main() -> None:
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    rows = judge_rows(read_jsonl(sys.argv[1]))
    out = Path(sys.argv[2])
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(rows)} answers to {out}")


if __name__ == "__main__":
    main()
