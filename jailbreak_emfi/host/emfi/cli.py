"""Command line shared by the experiment entry scripts (run_*.py).

Settings come from config/host.toml plus the experiment TOML; the options below only choose what varies
from run to run (how many trials, where to log, which positions)."""

from __future__ import annotations

import argparse
import uuid
from pathlib import Path

from common import config

from . import runner


def _uuid(value: str) -> str:
    try:
        return uuid.UUID(value).hex
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a UUID") from exc


def _indices(text: str) -> frozenset[int]:
    """A comma list, or a file with one index per line."""
    path = Path(text)
    items = path.read_text().split() if path.is_file() else text.split(",")
    return frozenset(int(x) for x in items if x.strip())


def _stops(text: str) -> tuple[int, int]:
    x, y = (int(v) for v in text.lower().split("x"))
    return x, y


def parser(description: str, default_experiment: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--experiment", default=default_experiment,
                   help="experiment TOML (relative to config/ or a path; default: %(default)s)")
    p.add_argument("--host-config", default="host.toml", help="host TOML (default: %(default)s)")
    p.add_argument("-t", "--trials", type=int, help="trials per position (default: run.trials)")
    p.add_argument("-n", "--trials-per-prompt", type=int, help="default: prompts.trials_per_prompt")
    p.add_argument("--start-prompt", type=int, help="first prompt slot (default: prompts.start_prompt)")
    p.add_argument("--trial-log", type=Path, help="CSV of the trials (default: log.trial_log_dir/<time>.csv)")
    p.add_argument("--collection-id", type=_uuid, help="UUID of this run (default: a new one)")
    p.add_argument("--stay", action="store_true", help="never move the stage (stage.mode = stay)")
    p.add_argument("--position-index", type=int, help="only this stage position")
    p.add_argument("--indices", type=_indices, help="only these positions: comma list or file")
    p.add_argument("--start-index", type=int, default=0, help="skip positions before this one (resume a scan)")
    p.add_argument("--stops", type=_stops, help="XxY grid, replaces the Inspector dialog's grid")
    return p


def main(description: str, default_experiment: str, argv=None) -> int:
    args = parser(description, default_experiment).parse_args(argv)
    runner.main(config.load(args.host_config, args.experiment), args)
    return 0
