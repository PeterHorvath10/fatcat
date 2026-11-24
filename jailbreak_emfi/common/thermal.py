"""Temperature of a Jetson: the hottest sysfs thermal zone (used by the Jetson workers)."""

from __future__ import annotations

from pathlib import Path

THERMAL_ROOT = Path("/sys/class/thermal")
MILLIDEGREES_PER_DEGREE = 1000.0


def max_temp_c(root: Path = THERMAL_ROOT) -> float:
    """Hottest thermal zone in degrees C (0.0 if none can be read)."""
    temps = []
    for zone in root.glob("thermal_zone*/temp"):
        try:
            temps.append(int(zone.read_text()) / MILLIDEGREES_PER_DEGREE)
        except (OSError, ValueError):
            pass
    return max(temps) if temps else 0.0
