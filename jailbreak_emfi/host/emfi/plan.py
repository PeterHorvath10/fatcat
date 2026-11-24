"""Glitch parameters and prompt slot of each trial, from the experiment's [glitch] and [prompts] settings.

Amplitude and pulse width are drawn at random per trial (uniformly from the configured lists); the delay
comes from delays.DelayPicker. One pulse per trial.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from .delays import DelayPicker, window_counts
from .glitcher import GlitchParams, ns_to_counts


@dataclass(frozen=True)
class Prompting:
    """Prompt slots: the worker advances to the next prompt every trials_per_prompt trials (and wraps the
    slot around its own prompt list)."""
    trials_per_prompt: int
    start_prompt: int

    def slot(self, k: int) -> tuple[int, int]:
        """(prompt slot, 1-based repeat) for the k-th trial."""
        return self.start_prompt + k // self.trials_per_prompt, k % self.trials_per_prompt + 1


class Planner:
    def __init__(self, amplitudes_v, widths, primary_repeat: int, delay: DelayPicker,
                 rng: random.Random):
        self.amplitudes_v, self.widths, self.primary_repeat = list(amplitudes_v), list(widths), primary_repeat
        self.delay, self.rng = delay, rng

    def params(self, k: int, slot: int | None) -> GlitchParams:
        width = self.rng.choice(self.widths)
        amplitude = self.rng.choice(self.amplitudes_v)
        return GlitchParams(amplitude_v=amplitude, primary_delay=self.delay(k, slot), primary_width=width,
                            primary_repeat=self.primary_repeat)


def widths_counts(min_ns: float, max_ns: float, step_ns: float) -> list[int]:
    """Pulse widths min..max inclusive, in counts."""
    return list(range(ns_to_counts(min_ns), ns_to_counts(max_ns) + 1, ns_to_counts(step_ns)))


def from_config(cfg, trials_per_prompt: int | None, start_prompt: int, use_prompts: bool,
                rng: random.Random) -> tuple[Planner, Prompting | None]:
    mode = cfg.get("glitch.delay_mode")
    prompting = Prompting(trials_per_prompt, start_prompt) if use_prompts else None
    window = window_counts(cfg.get("glitch.window_start_ns", float), cfg.get("glitch.window_end_ns", float),
                           cfg.get("glitch.window_step_ns", float))
    picker = DelayPicker(mode, window=window, trials_per_prompt=trials_per_prompt if use_prompts else None,
                         rng=rng)
    planner = Planner(
        amplitudes_v=cfg.get("glitch.amplitudes_v", float),
        widths=widths_counts(cfg.get("glitch.width_ns_min", float), cfg.get("glitch.width_ns_max", float),
                             cfg.get("glitch.width_ns_step", float)),
        primary_repeat=cfg.get("glitch.primary_repeat", int),
        delay=picker,
        rng=rng,
    )
    return planner, prompting
