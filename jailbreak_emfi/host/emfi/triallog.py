"""Per-trial CSV log (one row per trial, appended and flushed at once so a crash loses nothing).

Delays and widths are in VC Glitcher counts."""

from __future__ import annotations

import contextlib
import csv
import time
from pathlib import Path

from .glitcher import GlitchParams

FIELDS = [
    "timestamp", "collection_id", "trial_index", "attempt_index", "active_prompt", "reset_hit", "error",
    "glitch_voltage", "primary_delay", "primary_duration", "primary_repeat",
    "position_x", "position_y", "position_z",
]


class TrialLog:
    def __init__(self, writer, fh):
        self._writer, self._fh = writer, fh

    def write(self, trial_index: int, attempt_index: int, collection_id: str, p: GlitchParams,
              position, reset_hit: bool = False, error: str = "", active_prompt: int | None = None) -> None:
        self._writer.writerow({
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "collection_id": collection_id,
            "trial_index": trial_index,
            "attempt_index": attempt_index,
            "active_prompt": "" if active_prompt is None else active_prompt,   # prompt slot
            "reset_hit": reset_hit,
            "error": error,
            "glitch_voltage": p.amplitude_v,
            "primary_delay": p.primary_delay,
            "primary_duration": p.primary_width,
            "primary_repeat": p.primary_repeat,
            "position_x": position[0],
            "position_y": position[1],
            "position_z": position[2],
        })
        self._fh.flush()


@contextlib.contextmanager
def open_log(path: Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        if header:
            writer.writeheader()
            fh.flush()
        yield TrialLog(writer, fh)
