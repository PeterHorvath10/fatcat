"""systemctl helpers for the worker scripts.

Two ways to get root: a sudoers NOPASSWD rule for exactly `systemctl restart <model units>` (the query
worker's corruption watchdog, rendered by jetson/install.py), or the sudo password read from stdin by the
interactive tools (never stored).
"""

from __future__ import annotations

import subprocess
from typing import Sequence

SYSTEMCTL = "/usr/bin/systemctl"   # absolute: the sudoers rule names this exact binary
LIST_TIMEOUT_S = 15


def unit_name(unit: str) -> str:
    return unit if unit.endswith(".service") else f"{unit}.service"


def active_unit(candidates: Sequence[str]) -> str | None:
    """The first unit of candidates that is active, or None."""
    names = [unit_name(u) for u in candidates]
    try:
        out = subprocess.run([SYSTEMCTL, "list-units", "--no-legend", "--state=active", *names],
                             capture_output=True, text=True, timeout=LIST_TIMEOUT_S).stdout
    except (OSError, subprocess.SubprocessError) as error:
        print(f"systemd: could not list units: {error}", flush=True)
        return None
    for line in out.splitlines():
        fields = line.replace("●", " ").split()   # failed units are prefixed with a bullet
        if fields and fields[0] in names:
            return fields[0]
    return None


def systemctl_command(action: str, units: Sequence[str], *, with_password: bool) -> list[str]:
    sudo = ["sudo", "-S", "-p", ""] if with_password else ["sudo", "-n"]
    return [*sudo, SYSTEMCTL, action, *(unit_name(u) for u in units)]


def systemctl(action: str, units: Sequence[str], *, sudo_password: str | None = None,
              timeout_s: float | None = None) -> bool:
    """Run sudo systemctl ACTION UNITS...; True on success (errors are printed, not raised)."""
    command = systemctl_command(action, units, with_password=sudo_password is not None)
    try:
        done = subprocess.run(command, input=None if sudo_password is None else sudo_password + "\n",
                              capture_output=True, text=True, timeout=timeout_s)
    except (OSError, subprocess.SubprocessError) as error:
        print(f"systemd: {action} {' '.join(units)} failed: {error}", flush=True)
        return False
    if done.returncode != 0:
        print(f"systemd: '{' '.join(command)}' exited {done.returncode}: {done.stderr.strip()}", flush=True)
        return False
    return True


def is_active(units: Sequence[str]) -> str:
    """Output of systemctl is-active for units (one state per line)."""
    return subprocess.run([SYSTEMCTL, "is-active", *(unit_name(u) for u in units)],
                          capture_output=True, text=True).stdout.strip()
