"""Thermal pacing for Thor: before a trial, if Thor is at or above pause_c, wait until it is at or below
resume_c. Thor's internal fan failed during the experiments; with only an external fan, unpaced GPU load
reaches >105 C within a minute and the board shuts off (which looks like a hang)."""

from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass

# hottest thermal zone in millidegrees C (Linux sysfs on the Jetson)
HOTTEST_ZONE_CMD = "cat /sys/class/thermal/thermal_zone*/temp 2>/dev/null | sort -n | tail -1"


@dataclass(frozen=True)
class ThermalSettings:
    ssh_host: str
    pause_c: float
    resume_c: float
    poll_s: float
    ssh_timeout_s: float


def board_temp_c(ssh_host: str, timeout_s: float) -> float | None:
    """Hottest thermal zone over ssh, or None if unreadable (e.g. the board is rebooting)."""
    try:
        out = subprocess.run(["ssh", "-o", f"ConnectTimeout={int(timeout_s)}", ssh_host, HOTTEST_ZONE_CMD],
                             capture_output=True, text=True, timeout=3 * timeout_s).stdout.strip()
        return int(out) / 1000 if out else None
    except (subprocess.SubprocessError, ValueError):
        return None


def guard(t: ThermalSettings | None) -> None:
    if t is None:
        return
    temp = board_temp_c(t.ssh_host, t.ssh_timeout_s)
    if temp is None or temp < t.pause_c:
        return
    print(f"Board at {temp:.1f} C, pausing until <= {t.resume_c:.0f} C", flush=True)
    while temp is None or temp > t.resume_c:
        time.sleep(t.poll_s)
        temp = board_temp_c(t.ssh_host, t.ssh_timeout_s)
    print(f"Board at {temp:.1f} C, resuming", flush=True)


def settings_from_config(cfg) -> ThermalSettings | None:
    if not cfg.get("thermal.enabled", bool):
        return None
    return ThermalSettings(
        ssh_host=cfg.get(f"boards.{cfg.get('experiment.board')}.ssh_host"),
        pause_c=cfg.get("thermal.pause_c", float),
        resume_c=cfg.get("thermal.resume_c", float),
        poll_s=cfg.get("thermal.poll_s", float),
        ssh_timeout_s=cfg.get("thermal.ssh_timeout_s", float),
    )
