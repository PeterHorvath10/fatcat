"""Read eval_model output on stdin ('A: ' lines); exit 0 if at least hunt.min_sensible answers are coherent
(analysis.keyword: [keyword] in config/analysis.toml), i.e. the wiped model is not garbled.

Usage: sensible.py EXPERIMENT_TOML < answers.txt"""

import sys

from common import config
from analysis.keyword import KeywordScore


def main() -> None:
    cfg = config.load("analysis.toml", "host.toml", sys.argv[1])
    kw = KeywordScore(cfg)
    answers = [line[3:] for line in sys.stdin.read().splitlines() if line.startswith("A: ")]
    n = sum(map(kw.coherent, answers))
    print(f"sensible answers: {n}/{len(answers)}")
    sys.exit(0 if n >= cfg.get("hunt.min_sensible", int) else 1)


if __name__ == "__main__":
    main()
