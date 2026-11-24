"""Thermal guard: pause GPU work while the board is hot.

Thor's internal fan is dead (only an external fan cools it), and back-to-back generation drives the
junction temperature past 100 C within seconds. Every script that keeps the GPU busy calls
ThermalGuard.wait() before each request: above pause_c it blocks until the hottest thermal zone is
back below resume_c.
"""

from __future__ import annotations

import time
from pathlib import Path

from common import config
from common.thermal import THERMAL_ROOT, max_temp_c

class ThermalGuard:
    def __init__(self, pause_c: float, resume_c: float, poll_s: float, root: Path = THERMAL_ROOT):
        if resume_c > pause_c:
            raise ValueError(f"thermal resume ({resume_c} C) must not be above pause ({pause_c} C)")
        self.pause_c, self.resume_c, self.poll_s, self.root = pause_c, resume_c, poll_s, root

    @classmethod
    def from_config(cls, cfg: config.Config) -> "ThermalGuard":
        return cls(cfg.get("thermal.pause_c", float), cfg.get("thermal.resume_c", float),
                   cfg.get("thermal.poll_s", float))

    def temp(self) -> float:
        return max_temp_c(self.root)

    def wait(self) -> None:
        if self.temp() < self.pause_c:
            return
        print(f"thermal: {self.temp():.1f} C, waiting", flush=True)
        while self.temp() > self.resume_c:
            time.sleep(self.poll_s)
