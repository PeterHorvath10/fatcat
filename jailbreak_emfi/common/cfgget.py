"""Print settings for shell scripts, with the same rules as the Python code (an empty value is an error).

usage: PYTHONPATH=<jailbreak_emfi> python -m common.cfgget [--lines] FILE[,FILE...] KEY [KEY ...]
  FILEs are merged left to right (config/ names or paths), e.g. host.toml,experiments/thor_weight_reload_hunt.toml
  One line per key; lists space-separated (--lines: one item per line), tables as JSON.
"""

from __future__ import annotations

import json
import sys

from common import config


def main(argv: list[str]) -> None:
    lines = argv[:1] == ["--lines"]
    argv = argv[1:] if lines else argv
    if len(argv) < 2:
        sys.exit(__doc__)
    try:
        cfg = config.load(*argv[0].split(","))
        values = [cfg.get(key) for key in argv[1:]]
    except config.ConfigError as exc:
        sys.exit(f"config error: {exc}")
    for v in values:
        if isinstance(v, list):
            print(("\n" if lines else " ").join(map(str, v)))
        else:
            print(json.dumps(v) if isinstance(v, dict) else v)


if __name__ == "__main__":
    main(sys.argv[1:])
