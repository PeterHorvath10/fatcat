"""The trial loop shared by every experiment.

Per position (stage) and trial:
  wait for the worker's COLLECT -> load the glitch pattern -> arm scope + glitcher -> send the
  acknowledgement (the worker starts its GPU work; the scope edge triggers the glitcher) -> wait for the
  pattern -> [read the weight report] -> log the trial.
A dead connection (the board hung or crashed) is logged as reset_hit, the board is power-cycled and the run
goes on. What differs between the experiments is configuration (config/experiments/*.toml): the delay
strategy, pulse lists, prompt slots and acknowledgement form, trigger source, stage mode, thermal pacing.
"""

from __future__ import annotations

import contextlib
import random
import socket
import subprocess
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from . import plan as plan_mod
from . import reset as reset_mod
from . import scope as scope_mod
from . import stage as stage_mod
from . import target as target_mod
from . import thermal as thermal_mod
from .glitcher import Glitcher, GlitchParams
from .triallog import open_log
from common import protocol


@dataclass(frozen=True)
class TriggerSettings:
    software: bool               # True: start the pattern in software software_delay_s after the ack
    arm_to_ack_s: float          # pause between arming and sending the ack
    timeout_s: float | None      # scope trigger: start in software after this long without an edge
    software_delay_s: float | None
    busy_timeout_s: float | None


@dataclass(frozen=True)
class RunSettings:
    collection_id: str
    trials: int                      # per position
    pause_s: float                   # before every trial
    cool_every: int                  # every this many trials ...
    cool_s: float                    # ... pause this much longer
    trial_counter: str               # which trial count drives delays and prompt slots: "run" (continues
                                     # across positions) or "position" (restarts at every position)
    next_position_on_reset: bool     # a scan moves to the next position after a reset
    position_index: int | None       # only this position
    indices: frozenset[int] | None   # only these positions
    start_index: int                 # skip positions before this one
    trial_log: Path
    fetch_remote: str | None         # scp source fetched at the end (the worker's own log)
    fetch_local_dir: str | None
    capture_dir: Path | None
    capture_channel: str | None
    capture_wait_s: float | None


@dataclass
class Experiment:
    """Everything the loop needs; hardware access is injected so the loop can run against fakes."""
    run: RunSettings
    target: target_mod.TargetSettings
    trigger: TriggerSettings
    planner: plan_mod.Planner
    prompting: plan_mod.Prompting | None
    stage: stage_mod.StageSettings
    thermal: thermal_mod.ThermalSettings | None
    reset: reset_mod.ResetSettings | None


def run_trial(sock, glitcher, scope, p: GlitchParams, exp: Experiment, trial_index: int, attempt_index: int,
              slot: int | None, repeat: int | None) -> None:
    sock.settimeout(exp.target.socket_timeout_s)
    protocol.wait_for_collect(sock)
    print(f"Run: {trial_index}; attempt_index: {attempt_index}")
    ack = target_mod.acknowledgement(exp.target, attempt_index, exp.run.collection_id, slot, repeat)
    trig = exp.trigger
    try:
        glitcher.load(p, software_trigger=trig.software)
        if scope is not None:
            scope_mod.arm_single(scope)
        glitcher.arm()
        time.sleep(trig.arm_to_ack_s)
        sock.sendall(ack)
        ack_sent = time.monotonic()
        if trig.software:
            # the worker starts its work as soon as the ack arrives; the glitch delay counts from the start
            start_at = ack_sent + trig.software_delay_s
            while time.monotonic() < start_at:
                pass
            took = glitcher.soft_start(trig.busy_timeout_s)
            print(f"Soft trigger {trig.software_delay_s:.3f} s after the ack", flush=True)
            if took is not None:
                print(f"Glitch pattern done {took:.3f} s after the soft trigger", flush=True)
        elif glitcher.wait_external(trig.timeout_s):
            # pattern done = scope edge + glitch delay; minus the delay this is when the edge came after the ack
            print(f"Glitch pattern done {time.monotonic() - ack_sent:.3f} s after the ack", flush=True)
    finally:
        glitcher.disarm()


def _positions_wanted(i: int, r: RunSettings) -> bool:
    if r.position_index is not None and i != r.position_index:
        return False
    if r.indices is not None and i not in r.indices:
        return False
    return i >= r.start_index


def run(exp: Experiment, glitcher, connect: Callable[[], socket.socket], scope, positions,
        reset_board: Callable[[], None]) -> None:
    """The loop. `positions` yields (index, position, move); `connect` returns a connected socket."""
    r = exp.run
    sock = connect()
    k_run = 0            # trials of the whole run
    try:
        with open_log(r.trial_log) as log, glitcher:
            for i, position, move in positions:
                if not _positions_wanted(i, r):
                    continue
                move()
                t = 0
                while t < r.trials:
                    k = k_run if r.trial_counter == "run" else t
                    slot = repeat = None
                    if exp.prompting:
                        slot, repeat = exp.prompting.slot(k)
                        if k % exp.prompting.trials_per_prompt == 0:
                            print(f"Prompt slot {slot}", flush=True)
                    p = exp.planner.params(k, slot)
                    if t % r.cool_every == 0:
                        print("Letting GPU cool...")
                        time.sleep(r.cool_s)
                    thermal_mod.guard(exp.thermal)
                    error = None
                    try:
                        time.sleep(r.pause_s)
                        run_trial(sock, glitcher, scope, p, exp, t, i, slot, repeat)
                        if exp.target.weight_report:
                            target_mod.print_weight_report(sock)
                        log.write(t, i, r.collection_id, p, position, active_prompt=slot)
                    except (ConnectionError, OSError, socket.timeout) as exc:
                        error = str(exc)
                        print(f"Socket failed during trial {t}: {exc}")
                        log.write(t, i, r.collection_id, p, position, reset_hit=True, error=error,
                                  active_prompt=slot)
                        with contextlib.suppress(Exception):
                            sock.close()
                        reset_board()
                        sock = connect()
                        if r.next_position_on_reset and r.position_index is None:
                            t = r.trials   # a scan moves on to the next position; a fixed position keeps going
                    except Exception as exc:
                        log.write(t, i, r.collection_id, p, position, error=str(exc), active_prompt=slot)
                        raise
                    finally:
                        if scope is not None and r.capture_dir is not None:
                            meta = {"trial_index": k_run, "prompt_slot": slot, "prompt_repeat": repeat,
                                    "error": error, "glitch_voltage": float(p.amplitude_v),
                                    "primary_delay": int(p.primary_delay),
                                    "primary_duration": int(p.primary_width), "primary_repeat": p.primary_repeat}
                            scope_mod.save_capture(scope, r.capture_dir, f"t{k_run:05d}", r.capture_channel,
                                                   meta, r.capture_wait_s)
                        t += 1
                        k_run += 1
    finally:
        with contextlib.suppress(Exception):
            sock.close()


# ---- building an Experiment from the configuration --------------------------------------------------------

def _opt(cfg, key, cast=None):
    return cfg.get(key, cast, default=None)


def build(cfg, args) -> Experiment:
    """Read and validate every setting before any instrument is touched (a missing value stops here)."""
    board = f"boards.{cfg.get('experiment.board')}"
    ack_form = cfg.get("target.ack_form")
    if ack_form not in target_mod.ACK_FORMS:
        raise ValueError(f"target.ack_form must be one of {target_mod.ACK_FORMS}")
    weight_report = cfg.get("target.weight_report", bool)
    if weight_report and ack_form != "slot_weights":
        raise ValueError("target.weight_report needs target.ack_form = slot_weights")
    use_prompts = ack_form != "short"
    tpp = args.trials_per_prompt or (cfg.get("prompts.trials_per_prompt", int) if use_prompts else None)
    start_prompt = args.start_prompt if args.start_prompt is not None else (
        cfg.get("prompts.start_prompt", int) if use_prompts else 0)
    seed = _opt(cfg, "run.seed", int)
    planner, prompting = plan_mod.from_config(cfg, tpp, start_prompt, use_prompts, random.Random(seed))

    software = cfg.get("trigger.source") == "software"
    trigger = TriggerSettings(
        software=software,
        arm_to_ack_s=cfg.get("trigger.arm_to_ack_s", float),
        timeout_s=None if software else cfg.get("trigger.timeout_s", float),
        software_delay_s=cfg.get("trigger.software_delay_s", float) if software else None,
        busy_timeout_s=cfg.get("trigger.soft_busy_timeout_s", float) if software else None,
    )
    tgt = target_mod.TargetSettings(
        host=cfg.get(f"{board}.target_host"), port=cfg.get(f"{board}.target_port", int),
        socket_timeout_s=cfg.get("network.socket_timeout_s", float),
        connect_retry_s=cfg.get("network.connect_retry_s", float),
        connect_settle_s=cfg.get("network.connect_settle_s", float),
        ack_form=ack_form, weight_report=weight_report)

    stage_mode = "stay" if args.stay else cfg.get("stage.mode")
    stage = stage_mod.settings_from_config(cfg, stage_mode, args.stops)
    capture = cfg.get("scope.enabled", bool) and cfg.get("scope.capture", bool)
    trial_log = args.trial_log or Path(cfg.get("log.trial_log_dir")) / f"trial_parameters{time.strftime('%Y-%m-%dT%H%M%S')}.csv"
    fetch = cfg.get("results.fetch", bool)
    run_settings = RunSettings(
        collection_id=args.collection_id or uuid.uuid4().hex,
        trials=args.trials or cfg.get("run.trials", int),
        pause_s=cfg.get("run.pause_s", float),
        cool_every=cfg.get("run.cool_every", int),
        cool_s=cfg.get("run.cool_s", float),
        trial_counter=cfg.get("run.trial_counter"),
        next_position_on_reset=cfg.get("run.next_position_on_reset", bool) and stage_mode != "stay",
        position_index=args.position_index,
        indices=args.indices,
        start_index=args.start_index,
        trial_log=Path(trial_log),
        fetch_remote=cfg.get("results.fetch_remote") if fetch else None,
        fetch_local_dir=cfg.get("results.fetch_local_dir") if fetch else None,
        capture_dir=Path(cfg.get("scope.capture_dir")) if capture else None,
        capture_channel=cfg.get("scope.channel") if capture else None,
        capture_wait_s=cfg.get("scope.capture_wait_s", float) if capture else None,
    )
    return Experiment(run_settings, tgt, trigger, planner, prompting, stage,
                      thermal_mod.settings_from_config(cfg), reset_mod.settings_from_config(cfg))


def describe(exp: Experiment) -> None:
    r = exp.run
    print(f"Collection ID: {r.collection_id}", flush=True)
    print(f"Delays: {exp.planner.delay.describe()}; voltages {exp.planner.amplitudes_v}; widths (counts) "
          f"{exp.planner.widths}", flush=True)
    if exp.prompting:
        print(f"Advancing prompt slot every {exp.prompting.trials_per_prompt} trials from slot "
              f"{exp.prompting.start_prompt}", flush=True)


def main(cfg, args) -> None:
    """Run an experiment on the real instruments."""
    exp = build(cfg, args)
    describe(exp)
    glitcher = Glitcher(cfg.get("glitcher.device_index", int), cfg.get("glitcher.vcc_v", float),
                        cfg.get("glitcher.clk_high_v", float), cfg.get("glitcher.initial_glitch_v", float))
    start = time.time()
    with scope_mod.open_from_config(cfg) as scope, stage_mod.positions(exp.stage) as positions:
        run(exp, glitcher, lambda: target_mod.connect(exp.target), scope, positions,
            lambda: reset_mod.reset_board(exp.reset))
    print(f"It took {time.time() - start}s")
    if exp.run.fetch_remote:
        subprocess.call(["scp", exp.run.fetch_remote, exp.run.fetch_local_dir])
