"""Riscure XYZ stage through Inspector's Python API (imported lazily).

Two modes:
  dialog  Inspector's parameter dialog asks for the stage and the grid (defaults from [stage]; the dialog
          remembers the last values); the scan visits every grid position.
  stay    the stage is never touched (no Inspector needed); all trials run at the probe's current position,
          logged as NaN coordinates.
"""

from __future__ import annotations

import contextlib
import sys
import time
from dataclasses import dataclass
from typing import Iterator

NAN_POSITION = [float("nan")] * 3

# Inspector dialog labels (also the keys of its parameter cache)
XYZ_DEVICE = "XYZ device"
X_STOPS = "Number of stops in W-E direction"
Y_STOPS = "Number of stops in N-S direction"
SCAN_COUNT = "Number of scans"
HOP_HEIGHT = "Hop Height"


@dataclass(frozen=True)
class StageSettings:
    mode: str                     # "dialog" or "stay"
    x_stops: int | None = None    # dialog defaults
    y_stops: int | None = None
    scan_count: int | None = None
    hop_height: int | None = None
    stops_override: tuple[int, int] | None = None   # --stops XxY: replaces the dialog's grid


def open_table(ins, name: str):
    """Open the connected XYZ table called `name` (as Inspector lists it)."""
    from riscure.inspector import JavaError
    desc = next((d for d in ins.get_connected_xyz_tables() if d["name"] == name), None)
    if desc is None:
        sys.exit(f"Opening connection to XYZ table ... ERROR\nNo XYZ table named {name!r}.")
    print(f"Opening connection to {name} ... ", end="")
    try:
        table = ins.open_xyz_table(desc)
    except JavaError:
        sys.exit("ERROR\nPlease make sure the selected xyz table was calibrated, that the calibration is "
                 "valid, and that the table is not in use by something else.")
    print("OK")
    return table


@contextlib.contextmanager
def positions(s: StageSettings) -> Iterator:
    """Yield an iterable of (index, position, move) where move() puts the probe there (a no-op in stay mode)."""
    if s.mode == "stay":
        yield [(0, NAN_POSITION, lambda: print("Position: current (table not moved)"))]
        return
    if s.mode != "dialog":
        raise ValueError(f"unknown stage mode {s.mode!r} (dialog or stay)")
    import riscure.inspector
    from riscure.inspector import ConstantIntParameter, SelectionParameter
    from riscure.inspector.util import restore_parameter_cache
    with riscure.inspector.connect() as ins:
        print("Requesting user input in Inspector dialog ... ", end="")
        names = [d["name"] for d in ins.get_connected_xyz_tables()]
        print(names)
        defaults = {
            XYZ_DEVICE: SelectionParameter(ins, XYZ_DEVICE, names or ["NO OPTIONS SET"]),
            X_STOPS: ConstantIntParameter(ins, X_STOPS, s.x_stops),
            Y_STOPS: ConstantIntParameter(ins, Y_STOPS, s.y_stops),
            SCAN_COUNT: ConstantIntParameter(ins, SCAN_COUNT, s.scan_count),
            HOP_HEIGHT: ConstantIntParameter(ins, HOP_HEIGHT, s.hop_height),
        }
        params = ins.parameter_configuration_dialog(defaults, cache=restore_parameter_cache())
        print("OK")
        table = open_table(ins, params[XYZ_DEVICE])
        print("Creating position source... ", end="")
        source = ins.create_basic_xyz_table_source(table)
        source.settings.x_stops, source.settings.y_stops = s.stops_override or (params[X_STOPS], params[Y_STOPS])
        source.settings.scan_count = params[SCAN_COUNT]
        source.settings.hop_height = params[HOP_HEIGHT]
        print(f"Grid: {source.settings.x_stops} x {source.settings.y_stops} stops")

        def mover(pos):
            def move():
                t0 = time.time()
                table.set_position(pos)
                print("Position: ", pos)
                print(f"Time taken for Setting XYZ position: {int(1000 * (time.time() - t0))}ms")
            return move

        yield ((i, pos, mover(pos)) for i, pos in enumerate(source))


def goto(table_name: str, pos: list[float]) -> None:
    import riscure.inspector
    with riscure.inspector.connect() as ins:
        table = open_table(ins, table_name)
        try:
            print(f"moving to {pos} ...", flush=True)
            table.set_position(pos)
            print("done")
        finally:
            table.close()


def settings_from_config(cfg, mode: str, stops_override=None) -> StageSettings:
    if mode == "stay":
        return StageSettings("stay")
    return StageSettings(mode, cfg.get("stage.x_stops", int), cfg.get("stage.y_stops", int),
                         cfg.get("stage.scan_count", int), cfg.get("stage.hop_height", int), stops_override)
