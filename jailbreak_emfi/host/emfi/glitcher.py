"""Keysight Riscure VC Glitcher (embedded VCC mode) driving the EMFI transient probe.

One trial = one glitch pattern: a single pulse `delay` counts after the trigger.

The vendor SDK (vcglitchersdk) is imported lazily so that the pure parts of this module (units, the
GlitchParams record) can be used and tested without it.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

# Instrument fact: one VC Glitcher delay/width count is 2 ns (measured: 650M counts, repeat 1 -> 1.32 s).
NS_PER_COUNT = 2
# The VC Glitcher rejects a delay of 0 counts; the shortest delay is this many counts.
MIN_DELAY_COUNTS = 2


def ns_to_counts(ns: float) -> int:
    return int(ns) // NS_PER_COUNT


def counts_to_ms(counts: int) -> float:
    return counts * NS_PER_COUNT / 1e6


@dataclass(frozen=True)
class GlitchParams:
    """One trial's pattern. Delays and widths in VC Glitcher counts (NS_PER_COUNT ns each)."""
    amplitude_v: float
    primary_delay: int
    primary_width: int
    primary_repeat: int


def _sdk():
    from vcglitchersdk.enum import EVCG_TRIGGER_EDGE, EVCG_TRIGGER_SRC, GLITCH_MODE
    from vcglitchersdk.vcglitcher import VCGlitcher
    return EVCG_TRIGGER_EDGE, EVCG_TRIGGER_SRC, GLITCH_MODE, VCGlitcher


class Glitcher:
    """Thin wrapper over VCGlitcher. Use as a context manager (opens/closes the device)."""

    def __init__(self, device_index: int, vcc_v: float, clk_high_v: float, initial_glitch_v: float):
        edge, src, mode, vcg_cls = _sdk()
        self._edge, self._src, self._mode = edge, src, mode
        self.vcc_v, self.clk_high_v, self.initial_glitch_v = vcc_v, clk_high_v, initial_glitch_v
        self.vcg = vcg_cls()
        print(f"SDK version: {self.vcg.sdk_get_version()}")
        print(f"SDK snapshot version: {self.vcg.sdk_is_snapshot_version()}")
        count = self.vcg.device_list()
        if device_index >= count:
            raise RuntimeError(f"device index {device_index} is unavailable; found {count} device(s)")
        self.vcg.device_get_info(device_index)

    def __enter__(self):
        self.vcg.__enter__()
        print("VCG opened...")
        n_pattern = self.vcg.evcg_get_guaranteed_pattern_number()
        self.vcg.set_vcc_glitch_parameter(self.vcc_v, self.clk_high_v, self.initial_glitch_v)
        self.vcg.set_mode(self._mode.EMBEDDED_VCC)
        self.vcg.evcg_clear_pattern()
        print(f"N_pattern: {n_pattern}")
        return self

    def __exit__(self, *exc):
        return self.vcg.__exit__(*exc)

    def load(self, p: GlitchParams, software_trigger: bool) -> None:
        """Program the pattern and the trigger source (not armed yet)."""
        self.vcg.set_vcc_glitch_parameter(self.vcc_v, self.clk_high_v, p.amplitude_v)
        print(f"Glitch voltage: {p.amplitude_v}")
        print(f"Primary delay: {p.primary_delay}")
        print(f"Primary duration: {p.primary_width}")
        print(f"Primary repeat: {p.primary_repeat}")
        self.vcg.evcg_add_glitch(p.primary_delay, p.primary_width, p.primary_repeat)
        # evcg_soft_start() only starts the pattern with the SW_TRIGGER source; with TRIGGER_IN it is
        # ignored and the delay counts from the next scope edge instead (measured: soft start + TRIGGER_IN
        # never clears busy without an edge; SW_TRIGGER gives 2.0 ns per delay count).
        src = self._src.SW_TRIGGER if software_trigger else self._src.TRIGGER_IN
        self.vcg.evcg_trigger_config(src, self._edge.RISING)
        self.vcg.evcg_set_pattern()

    def arm(self) -> None:
        self.vcg.evcg_set_arm(True)

    def disarm(self) -> None:
        self.vcg.evcg_set_arm(False)
        self.vcg.evcg_clear_pattern()

    def wait_external(self, timeout_s: float) -> bool:
        """Wait for the scope edge + pattern. After timeout_s without an edge, start in software (so the
        trial still ends) and return False."""
        deadline = time.monotonic() + timeout_s
        while self.vcg.evcg_busy():
            if time.monotonic() >= deadline:
                print(f"No external trigger received after {timeout_s:g} s; triggering in software")
                self.vcg.evcg_soft_start()
                return False
        return True

    def soft_start(self, busy_timeout_s: float) -> float | None:
        """Start the pattern now and wait until it is done (bounded: evcg_busy() is not guaranteed to clear
        after a software start, and an unbounded wait hangs the whole run). Returns the time taken, or None
        on timeout (the caller's disarm then stops it)."""
        self.vcg.evcg_soft_start()
        started = time.monotonic()
        while self.vcg.evcg_busy():
            if time.monotonic() - started >= busy_timeout_s:
                print(f"Glitcher still busy {busy_timeout_s:g} s after the soft trigger; disarming", flush=True)
                return None
        return time.monotonic() - started
