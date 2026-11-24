"""Glitch delay after the trigger edge, per trial. Two strategies ([glitch].delay_mode):

  window      fixed window [start, end) in steps; trial k takes element (k mod trials_per_prompt) mod len
              when prompts are used (so every prompt sees the same delays), k mod len otherwise. A single
              fixed delay is a window one count wide (end = start + 2 ns).
  random      uniform random element of the window (Thor memory-check scans).

All delays are returned in VC Glitcher counts.
"""

from __future__ import annotations

import random

from .glitcher import NS_PER_COUNT, ns_to_counts


def window_counts(start_ns: float, end_ns: float, step_ns: float) -> list[int]:
    counts = list(range(ns_to_counts(start_ns), ns_to_counts(end_ns), ns_to_counts(step_ns)))
    if not counts:
        raise ValueError(f"empty delay window {start_ns}..{end_ns} ns")
    return counts


class DelayPicker:
    def __init__(self, mode: str, *, window: list[int], trials_per_prompt: int | None = None,
                 rng: random.Random):
        if mode not in ("window", "random"):
            raise ValueError(f"unknown delay mode {mode!r}")
        self.mode, self.window, self.tpp, self.rng = mode, window, trials_per_prompt, rng

    def describe(self) -> str:
        w = self.window
        return (f"{self.mode} delay {w[0] * NS_PER_COUNT / 1e6:g}..{w[-1] * NS_PER_COUNT / 1e6:g} ms "
                f"({len(w)} steps of {(w[1] - w[0]) * NS_PER_COUNT if len(w) > 1 else 0} ns)")

    def __call__(self, k: int, slot: int | None) -> int:
        if self.mode == "random":
            return self.rng.choice(self.window)
        idx = k % self.tpp if self.tpp else k
        return self.window[idx % len(self.window)]
