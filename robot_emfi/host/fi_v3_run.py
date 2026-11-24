#!/usr/bin/env python3
"""EM pulses into the inferences of the live rollout (Section 6.4). The arm holds still unless --move.

The Jetson runs smolvla_212_v3_rollout_async.py through fi_v3.py, which logs every motor command and
drops it unless --move, and fi_worker.py, which recomputes every inference fault-free and runs the
canary after each reply (one "FIV3" line per inference). The rollout requests a chunk every 19 ticks
(~634.5 ms). Every second inference is a target; the ones in between are controls:

  "[ASYNC REQUEST]" line of control inference k arrives (~ its GPU onset)
    -> Husky armed --lead-ms before the predicted onset of inference k+1
    -> probe signal -> ADC trigger -> glitch after the delay -> EM pulse
    -> long decimated capture: where did the real inference start and end (it ends ~112.8 ms after
       its onset, then the worker's quiet gap follows), and where did the pulse land?
  FIV3 line of inference k+1: chunk vs the fault-free recompute, canary vs its golden chunk.

Injector (--emfi):
  chipshouter (default)  armed once, it fires on every Husky glitch through its SMB trigger, so the
                         Husky's glitch length is the pulse width (--width-ns, 80-960 ns in 20 ns
                         steps); --voltage sets the capacitor bank. It cannot report whether it fired:
                         emfi = FIRED means it was armed before and after a Husky trigger. It stays
                         armed through ordinary rollout restarts (NaN aborts) and is disarmed for crash
                         recovery (reboots, Arduino resets, USB rebinds) and before the Husky's MCX
                         changes mode.
  faultycat              armed, charged and fired per shot (fire() waits for its TRIGGER_IN);
                         --width-us sets its pulse.

Placing the shots: the trigger is not the inference onset. With the cameras decoding at 30 fps the
probe is never quiet, and the Husky fires on the first burst after arming, ~15-25 ms before the
onset. So every shot is placed afterwards from its own trace: the end of the real inference is a
sharp drop, and the prefill is [-51.7, -42.6] ms before that end (profiled, +-0.3 ms).
pulse_rel_end_ms = delay - end_ms; pos_analyze.py turns it into the position in the inference.

--char: same timing, nothing fires (--glitch: the glitch module live anyway, to calibrate the trigger
with the exact sweep configuration); --levels alternates trigger levels.
--move: the arm executes the plans (fi_v3.py with FI_V3_MOVE=1), and every rollout end, including NaN
aborts, first ramps it back to rest before torque goes off. --sessions 1 --session-s 30 runs one
episode, e.g. with the scene reset by hand in between.
--nan-continue: a NaN chunk no longer ends the rollout; fi_worker.py (FI_V3_NAN=drop) keeps the
previous plan instead. This is the paper's convention.
--no-limits: the rollout's slew limit (12 deg/tick) and first-action check are off, so a faulted
chunk reaches the servos at full size (only their own position limits remain). The ramp back to
rest at every rollout end stays.
--pulses N --gap-ms G (ChipSHOUTER): every shot fires N pulses G ms apart (Husky num_glitches), the
first at the delay; delay_ms stays the first pulse, and each row logs glitches_sent.
--every: the target's own request is taken as the control for the next shot, so every inference is
pulsed when the shot before is done in time (an inference it is late for is skipped, "late by").
Only inferences that followed an unpulsed one still give a control row.
--double GAP_MS:A:B:STEP:SHOTS (FaultyCat): adds two-pulse shots, shuffled in with the single ones.
The FaultyCat fires trigger -> delay -> pulse -> delay -> pulse (repeat=2), so its delay is
GAP_MS - width and the Husky triggers it that much earlier. delay_ms is always the first pulse
after the Husky trigger; the second is GAP_MS later.

usage: fi_v3_run.py OUT --char --levels=-0.18,-0.20,-0.22 --shots 25
       fi_v3_run.py OUT --move --nan-continue --sessions 1 --session-s 30 --delays 59:68:0.25 --shots 1 \
           --level=-0.20 --voltage 300 --width-ns 80 --seed 1
       fi_v3_run.py OUT --emfi faultycat --delays 60:77:1 --shots 10 --level=-0.12
writes OUT.jsonl (one row per target and per control), OUT_env.npz, OUT.console
"""

import argparse
import json
import queue
import random
import re
import statistics
import subprocess
import threading
import time
from pathlib import Path

import chipwhisperer as cw
import numpy as np
import serial
try:                      # FaultyCat support (--emfi faultycat) is optional; the paper's runs use the ChipSHOUTER
    from faultycat import EmfiEngine, EmfiTrigger
    from faultycmd.protocols.emfi import EmfiClient, EmfiState
except ImportError:
    EmfiEngine = EmfiTrigger = EmfiClient = EmfiState = None
from husky import CLK, parse_delays, setup, setup_glitch
from shouter import PORT as SHOUTER_PORT
from shouter import WIDTH_NS, Shouter

PERIOD_S = 0.6345          # 19 ticks of v3 (profiled)
PREFILL_REL_END = (-51.66, -42.64)   # v3 prefill window relative to the last kernel end (profiled)
from site_config import ARDUINO_PORT as ARDUINO   # on the Jetson's reset line, or None
from site_config import CONDA_ACTIVATE, JETSON, JETSON_DIR, deploy_jetson
RESET_AFTER_S = 200        # the Jetson's own crash reboots were back after 125-154 s
THR = 0.12                 # envelope level that counts as GPU activity
REMOTE = (f"{CONDA_ACTIVATE} && cd {JETSON_DIR} && "
          f"printf '\\n' | {{env}}python -u {JETSON_DIR}/fi_v3.py ~/fi_logs/v3/{{tag}} smolvla_212_v3_rollout_async.py "
          "--duration {dur} --seed {seed}")


def alive():
    try:
        return subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5", JETSON, "true"],
                              capture_output=True, timeout=15).returncode == 0
    except subprocess.TimeoutExpired:      # a hanging kernel can accept the connection and never answer
        return False


def hard_reset():
    """Reset the Jetson through the Arduino Uno wired to it: "10" -> "ON", the Jetson drops off
    the network ~10 s later and is back ~65 s after the command (tested 2026-10-29)."""
    if not ARDUINO:
        return "no reset line configured (site_config.ARDUINO_PORT)"
    try:
        with serial.Serial(ARDUINO, 9600, timeout=0.2) as s:
            time.sleep(3)                  # opening the port reboots the Uno; its bootloader takes ~2 s
            s.reset_input_buffer()
            s.write(b"10\n")
            s.flush()
            t0, reply = time.monotonic(), b""
            while time.monotonic() - t0 < 3 and b"\n" not in reply:
                reply += s.read(64)
        return reply.decode(errors="replace").strip() or "no reply"
    except (OSError, serial.SerialException) as exc:
        return f"failed: {exc}"


def deploy():
    """Copy the Jetson-side scripts and data to JETSON_DIR (site_config.py)."""
    deploy_jetson()


def recover(n_crash, wait_s=900, settle_s=60, reset=False):
    """After a Jetson crash: wait for it to reboot, let the boot services (max-perf, cameras)
    finish, and restore the wrapper scripts that the reboot wiped from /tmp. If it stays down
    for RESET_AFTER_S (hung, no reboot of its own), or reset=True, reset it through the
    Arduino (at most twice). False if it does not come back within wait_s."""
    print(f"--- Jetson unreachable (crash {n_crash}); waiting up to {wait_s // 60} min for it to reboot"
          if not reset else f"--- Jetson unusable (crash {n_crash}); resetting it", flush=True)
    t0 = last_reset = time.monotonic()
    n_reset = 0
    if reset:
        last_reset = time.monotonic() - RESET_AFTER_S
    while time.monotonic() - t0 < wait_s:
        if n_reset < 2 and time.monotonic() - last_reset >= RESET_AFTER_S and (reset or not alive()):
            reset, n_reset, last_reset = False, n_reset + 1, time.monotonic()
            print(f"--- hard reset {n_reset}/2 through the Arduino: {hard_reset()}", flush=True)
            t_down = time.monotonic()
            while alive() and time.monotonic() - t_down < 45:      # it keeps answering for ~10 s
                time.sleep(2)
            continue
        if not alive():
            time.sleep(5)
            continue
        print(f"--- Jetson back after {time.monotonic() - t0:.0f} s; settling {settle_s} s, restoring {JETSON_DIR}",
              flush=True)
        time.sleep(settle_s)
        try:
            deploy()
            st = subprocess.run(["ssh", "-o", "BatchMode=yes", JETSON, "uptime; systemctl is-active jetson-max-perf.service"],
                                capture_output=True, text=True, timeout=30).stdout
        except (subprocess.SubprocessError, OSError) as exc:
            print(f"    restore failed ({type(exc).__name__}); waiting again", flush=True)
            continue
        print("    " + " | ".join(st.split("\n")).strip(" |"), flush=True)
        return True
    print("Jetson did not come back. Stopping.", flush=True)
    return False


def usb_reset(n_crash, settle_s=15):
    """After the Jetson's xHCI controller died (all USB gone: arm, cameras): rebind tegra-xusb
    so the devices re-enumerate. True if both arm ports and the cameras are back."""
    print(f"--- USB devices missing (crash {n_crash}); rebinding the xHCI controller", flush=True)
    # needs a sudoers rule that lets this user run the rebind without a password (sudo -n never prompts)
    cmd = ('sudo -n sh -c "echo 3610000.usb > /sys/bus/platform/drivers/tegra-xusb/unbind; '
           'sleep 2; echo 3610000.usb > /sys/bus/platform/drivers/tegra-xusb/bind"; '
           f'sleep {settle_s}; ls /dev/serial/by-id/ | grep -c 1a86; ls /dev/video* | wc -l')
    try:
        out = subprocess.run(["ssh", "-o", "BatchMode=yes", JETSON, cmd], capture_output=True, text=True,
                             timeout=60).stdout.split()
    except (subprocess.SubprocessError, OSError) as exc:
        print(f"    rebind failed ({type(exc).__name__})", flush=True)
        return False
    print(f"    arm ports {out[0] if out else '?'}, video nodes {out[1] if len(out) > 1 else '?'}", flush=True)
    return len(out) == 2 and int(out[0]) >= 2 and int(out[1]) >= 2


def kill_leftovers():
    """Kill rollout/worker processes of an earlier session that are still running on the Jetson
    (a pulse can hang the GPU: the worker then waits on a fence forever and the rollout keeps
    the cameras open). Returns the killed pids."""
    pat = f"[/]{JETSON_DIR[1:]}/fi_(v3|worker)"
    cmd = f"pgrep -f '{pat}'; pkill -9 -f '{pat}'; true"
    try:
        out = subprocess.run(["ssh", "-o", "BatchMode=yes", JETSON, cmd], capture_output=True, text=True,
                             timeout=30).stdout
    except (subprocess.SubprocessError, OSError):
        return []
    return out.split()


class Session:
    """One v3 rollout over SSH; a reader thread timestamps every line on arrival."""

    def __init__(self, tag, dur, seed, console, move=False, nan_continue=False, no_limits=False):
        env = ("FI_V3_MOVE=1 " if move else "") + ("FI_V3_NAN=drop " if nan_continue else "")
        cmd = REMOTE.format(tag=tag, dur=dur, seed=seed, env=env) + (" --max-tick-deg 0 --force" if no_limits else "")
        self.p = subprocess.Popen(["ssh", "-o", "BatchMode=yes", "-o", "ServerAliveInterval=2",
                                   "-o", "ServerAliveCountMax=5", JETSON, cmd],
                                  stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                  text=True, bufsize=1)
        self.tag = tag
        self.requests = queue.Queue()
        self.fiv3 = []
        self.cond = threading.Condition()
        self.ended = threading.Event()
        self.aborted = None
        self.at_rest = False
        self.console = console
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        for line in self.p.stdout:
            t = time.monotonic()
            self.console.write(f"{t:.4f} {line}")
            s = line.strip()
            if s.startswith("[ASYNC REQUEST]"):
                self.requests.put((t, int(re.search(r"source=(\d+)", s).group(1))))
            elif s.startswith("FIV3 "):
                d = json.loads(s[5:])
                d["t_local"] = t
                with self.cond:
                    self.fiv3.append(d)
                    self.cond.notify_all()
            elif "ROLLOUT ABORTED" in s or "AOT worker" in s and "error" in s.lower():
                self.aborted = self.aborted or s
            elif s.startswith("[fi_v3] arm back at rest"):
                self.at_rest = True
        self.ended.set()
        with self.cond:
            self.cond.notify_all()

    def next_request(self, timeout):
        deadline = time.monotonic() + timeout
        while True:
            try:
                return self.requests.get(timeout=0.05)
            except queue.Empty:
                if self.ended.is_set() and self.requests.empty() or time.monotonic() > deadline:
                    return None

    def wait_fiv3(self, source, timeout):
        deadline = time.monotonic() + timeout
        with self.cond:
            while True:
                hit = [d for d in self.fiv3 if d.get("source_step") == source]
                if hit:
                    return hit[-1]
                left = deadline - time.monotonic()
                if left <= 0 or self.ended.is_set():
                    return None
                self.cond.wait(min(left, 0.1))

    def close(self, timeout):
        try:
            self.p.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            self.p.terminate()

    def stop(self, timeout):
        """End the rollout early as Ctrl-C on the Jetson does: its finally disconnects the arm, which
        fi_v3.py turns into the ramp back to rest. Only cutting the ssh (close() on its timeout) can
        stop it mid-motion without the ramp (2026-11-04, move380_pfwin_every_e7)."""
        cmd = f"pkill -INT -f '^python -u {JETSON_DIR}/fi_v3.py [^ ]*/fi_logs/v3/{self.tag} '; true"
        try:
            subprocess.run(["ssh", "-o", "BatchMode=yes", JETSON, cmd], timeout=30)
        except (subprocess.SubprocessError, OSError):
            pass
        self.close(timeout)


def features(tr, decimate):
    dt_ms = decimate / CLK * 1e3
    b = max(1, int(round(0.1 / dt_ms)))                       # 0.1 ms envelope bins
    n = len(tr) // b
    x = tr[: n * b] - np.median(tr)
    env = np.abs(x).reshape(n, b).max(axis=1)
    bin_ms = b * dt_ms
    act = env > THR
    first = float(np.argmax(act) * bin_ms) if act.any() else None
    # Real inference end: the envelope, median-filtered over 0.7 ms (an EMFI pulse after the
    # end leaves a short blip that must not count as activity), drops below 0.1 for 5 ms.
    k = 7
    sm = np.median(np.lib.stride_tricks.sliding_window_view(np.pad(env, k // 2, mode="edge"), k), axis=1)
    end, quiet = None, int(round(5 / bin_ms))
    for j in range(int(90 / bin_ms), n - quiet):
        if sm[j - 1] >= 0.1 and sm[j:j + quiet].max() < 0.1:
            end = float(j * bin_ms)
            break
    peak = int(np.argmax(np.abs(x)))
    return {"first_active_ms": first, "end_ms": end, "peak_ms": peak * dt_ms, "peak_abs": float(np.abs(x[peak]))}, env


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--char", action="store_true", help="trigger alignment only: no EMFI")
    ap.add_argument("--glitch", action="store_true", help="--char: keep the glitch module live (FaultyCat stays idle)")
    ap.add_argument("--levels", default="-0.12", help="--char: trigger levels to alternate")
    ap.add_argument("--delays", default="80:96:1", help="ms after the Husky trigger: a:b:step or a,b,c")
    ap.add_argument("--shots", type=int, default=10, help="per delay (sweep) or per level (--char)")
    ap.add_argument("--double", action="append", default=[], metavar="GAP_MS:A:B:STEP:SHOTS",
                    help="two pulses GAP_MS apart, first pulse A:B:STEP ms after the Husky trigger (repeatable)")
    ap.add_argument("--pulses", type=int, default=1,
                    help="ChipSHOUTER: pulses per shot, --gap-ms apart, the first at the delay (Husky num_glitches)")
    ap.add_argument("--gap-ms", type=float, default=1.0, help="--pulses: start-to-start spacing of the pulses")
    ap.add_argument("--level", type=float, default=-0.12)
    ap.add_argument("--emfi", choices=("chipshouter", "faultycat"), default="chipshouter")
    ap.add_argument("--voltage", type=int, help="ChipSHOUTER capacitor bank, 150-500 V (required with it)")
    ap.add_argument("--width-ns", type=int, default=160, help="ChipSHOUTER pulse = Husky glitch length")
    ap.add_argument("--cs-highz", action="store_true", help="ChipSHOUTER trigger input ~1.8 kohm instead of 50 ohm")
    ap.add_argument("--cs-max-temp", type=int, default=65, help="stop if a ChipSHOUTER sensor reaches this (C)")
    ap.add_argument("--width-us", type=int, default=50, help="FaultyCat pulse width")
    ap.add_argument("--trig-pulse-us", type=float, default=10.0, help="FaultyCat: Husky trigger pulse length")
    ap.add_argument("--gain-db", type=float, default=10)
    ap.add_argument("--lead-ms", type=float, default=40.0, help="arm the Husky this long before the predicted onset")
    ap.add_argument("--decimate", type=int, default=75)
    ap.add_argument("--window-ms", type=float, default=170.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-crashes", type=int, default=5, help="stop after this many Jetson crashes")
    ap.add_argument("--restart-pause-s", type=float, default=10.0,
                    help="wait before restarting the rollout after an abort (lets the servo supply settle)")
    ap.add_argument("--move", action="store_true",
                    help="the arm moves for real (FI_V3_MOVE=1); back to rest before torque off at every session end")
    ap.add_argument("--sessions", type=int, default=0, help="stop after this many rollouts that started (0: no limit)")
    ap.add_argument("--session-s", type=int, default=0, help="cap each rollout at this many seconds (0: plan length)")
    ap.add_argument("--nan-continue", action="store_true",
                    help="a NaN chunk does not end the rollout: the worker drops it and the previous plan continues")
    ap.add_argument("--every", action="store_true",
                    help="pulse every inference the timing allows, not every second one (fewer controls)")
    ap.add_argument("--no-limits", action="store_true",
                    help="rollout without its slew limit (--max-tick-deg 0: policy actions go to the servos "
                         "unclipped) and without the first-action check (--force); return to rest stays")
    args = ap.parse_args()
    shout = args.emfi == "chipshouter"
    if shout:
        assert args.char or args.voltage is not None, "--voltage is required with the ChipSHOUTER"
        step = round(1e9 / CLK)
        assert step <= args.width_ns <= WIDTH_NS[1] and args.width_ns % step == 0, \
            f"--width-ns must be {step}-{WIDTH_NS[1]} in steps of {step} ns (Husky glitch clock)"
        if args.width_ns < WIDTH_NS[0]:
            print(f"note: {args.width_ns} ns is below the ChipSHOUTER's {WIDTH_NS[0]} ns spec minimum", flush=True)
        assert args.emfi != "faultycat" or EmfiClient, "--emfi faultycat needs the faultycat and faultycmd packages"
    assert not args.double, "--double is FaultyCat only (its own delay fires the second pulse)"
    trig_cycles = (round(args.width_ns * 1e-9 * CLK) if shout else max(1, round(args.trig_pulse_us * 1e-6 * CLK)))
    assert 1 <= args.pulses <= 32 and (args.pulses == 1 or shout and not args.char), \
        "--pulses is 1-32 and ChipSHOUTER only (the FaultyCat has --double)"
    assert args.pulses == 1 or args.gap_ms * 1e-3 * CLK > trig_cycles + 2, "--gap-ms shorter than a pulse"

    if args.char:
        plan = [{"delay_ms": 85.0 if args.glitch else None, "level": float(lv)}
                for _ in range(args.shots) for lv in args.levels.split(",")]
    else:
        plan = [{"delay_ms": d, "level": args.level, "repeat": 1, "gap_ms": None}
                for d in parse_delays(args.delays) for _ in range(args.shots)]
        for spec in args.double:
            gap, a, b, step, n = spec.split(":")
            plan += [{"delay_ms": d, "level": args.level, "repeat": 2, "gap_ms": float(gap)}
                     for d in parse_delays(f"{a}:{b}:{step}") for _ in range(int(n))]
            assert float(a) >= float(gap) - args.width_us / 1e3, f"--double {spec}: first pulse before the FaultyCat delay"
        random.Random(args.seed).shuffle(plan)

    deploy()                                                  # also after a Jetson reboot outside a run
    log = open(args.out + ".jsonl", "a", buffering=1)
    console = open(args.out + ".console", "a", buffering=1)
    envs = []
    client = emfi = shouter = None
    if shout and (not args.char or Path(SHOUTER_PORT).exists()):
        # connected and disarmed before the Husky is touched (also for --char: the MCX drives
        # it either way, and setup_glitch puts the long scope trigger on the MCX for a while)
        shouter = Shouter(None if args.char else args.voltage, None if args.char else args.width_ns,
                          term_50=not args.cs_highz)
    scope = cw.scope()
    if not args.char and not shout:
        client = EmfiClient.discover().__enter__()
        emfi = EmfiEngine(client)
    sess = None

    def hv_off():
        # Only for crash recovery: re-charging from 0 V after every NaN abort heated it to 59 C in
        # 10 min. Its arm timeout (2 min without a trigger) still disarms it if a restart drags on.
        if shouter is not None and not shouter.disarm():
            raise RuntimeError(f"ChipSHOUTER did not disarm ({shouter.last}); stopping")

    def hot():
        """Log the ChipSHOUTER temperatures; True (stop) if one reached --cs-max-temp. It faults
        itself at 80 C, but one of its sensors reads 0 and they have frozen once."""
        if shouter is None or args.char:
            return False
        t, v = shouter.temps(), shouter.measured_v()
        log.write(json.dumps({"shouter_temps": t, "shouter_v_measured": v, "after_shot": i - 1, "t": time.time()}) + "\n")
        print(f"--- ChipSHOUTER mosfet/transformer/diode {t} C, bank {v} V", flush=True)
        if t is not None and max(t) >= args.cs_max_temp:
            print(f"ChipSHOUTER at {max(t)} C (limit {args.cs_max_temp}). Stopping.", flush=True)
            return True
        return False

    i = 0
    try:
        setup(scope, args.gain_db)                            # glitch off, MCX inert
        if not args.char or args.glitch:
            setup_glitch(scope, args.level, trig_cycles)
            # Arm the glitch BEFORE the ADC trigger: the default ("after_scope") arms it ~12 ms
            # later, and an early trigger in that gap would never reach the FaultyCat.
            scope.glitch.arm_timing = "before_scope"
            if args.pulses > 1:                               # the ChipSHOUTER is still disarmed here
                scope.glitch.num_glitches = args.pulses
                scope.glitch.repeat = [trig_cycles] * args.pulses
        if emfi is not None:
            emfi.trigger = EmfiTrigger.EXT_RISING
            emfi.delay_us = 0
            emfi.width_us = args.width_us
        scope.trigger.module = "ADC"
        scope.adc.presamples = 0
        scope.adc.decimate = args.decimate
        scope.adc.samples = int(args.window_ms * 1e-3 * CLK / args.decimate)
        scope.adc.timeout = 0.5
        print(f"Husky: level {args.level if not args.char else args.levels}, decimate {scope.adc.decimate}, "
              f"{scope.adc.samples} samples = {scope.adc.samples * args.decimate / CLK * 1e3:.0f} ms, "
              f"glitch {'off' if args.char and not args.glitch else 'on'}; plan {len(plan)} targets", flush=True)
        if shouter is not None and not args.char:
            print(f"ChipSHOUTER: {args.voltage} V, pulse {args.width_ns} ns = Husky glitch {trig_cycles} cycles, "
                  f"trigger input {'high-Z' if args.cs_highz else '50 ohm'}"
                  + (f", {args.pulses} pulses {args.gap_ms:g} ms apart per shot" if args.pulses > 1 else ""), flush=True)

        i, n_session, bus_retries, n_crash, resets_at_shot, n_started = 0, 0, 0, 0, 0, 0
        while i < len(plan):
            n_session += 1
            if n_session > 1:
                time.sleep(args.restart_pause_s)
            if hot():
                break
            dur = int((2 * (len(plan) - i) + 8) * PERIOD_S + 5)
            if args.session_s:
                dur = min(dur, args.session_s)
            tag = f"{args.out.rsplit('/', 1)[-1]}_s{n_session}_{int(time.time())}"
            print(f"--- session {n_session}: v3 rollout {dur} s{', ARM MOVES' if args.move else ''}, "
                  f"remote log ~/fi_logs/v3/{tag}.npz", flush=True)
            if args.move and shouter is not None and not args.char:
                shouter.ready()          # charge during the rollout's start: a 30 s episode has no time to spare
            sess = Session(tag, dur, args.seed + n_session, console, move=args.move, nan_continue=args.nan_continue,
                           no_limits=args.no_limits)
            first = sess.next_request(240)
            if first is None and not alive():     # crashed during the restart pause or the start
                n_crash += 1
                log.write(json.dumps({"crash": True, "session": n_session, "after_shot": i - 1, "t": time.time()}) + "\n")
                hv_off()
                if n_crash > args.max_crashes or not recover(n_crash):
                    break
                continue
            if first is None:
                # servo bus hiccup while connecting (missing motor, input voltage error, no status
                # packet): nothing was enabled yet, so reconnecting after the pause is safe
                if bus_retries < 3 and re.search(r"motor check failed|RxPacketError|TxRxResult", sess.aborted or ""):
                    bus_retries += 1
                    print(f"--- arm did not connect ({(sess.aborted or '').splitlines()[0][:120]}); "
                          f"retry {bus_retries}/3", flush=True)
                    continue
                if re.search(r"Missing (follower|leader) port|Input/output error|No such device|Cannot open",
                             sess.aborted or ""):
                    n_crash += 1          # logged as a crash row so analyses skip it like one
                    log.write(json.dumps({"crash": True, "usb": True, "session": n_session, "after_shot": i - 1,
                                          "t": time.time()}) + "\n")
                    hv_off()
                    if n_crash > args.max_crashes or not (usb_reset(n_crash) or recover(n_crash, reset=True)):
                        break
                    continue
                raise RuntimeError(f"v3 rollout did not start: {sess.aborted}")
            bus_retries = 0
            n_started += 1
            periods, last, outcome = [], first, None
            carry, pulsed = None, set()               # --every: the target's request is the next control
            while i < len(plan):
                ev = carry if carry is not None else sess.next_request(5.0)
                carry = None
                if ev is None:
                    break
                if 0.55 < ev[0] - last[0] < 0.72:
                    periods.append(ev[0] - last[0])
                last = ev
                t_ctrl, src_ctrl = ev
                target = src_ctrl + 19
                shot = plan[i]
                period = statistics.median(periods[-8:]) if periods else PERIOD_S
                t_pred = t_ctrl + period
                row = {"n": i, "session": n_session, "delay_ms": shot["delay_ms"], "level": shot["level"],
                       "control_src": src_ctrl, "target_src": target, "t": time.time()}
                if shot["delay_ms"] is not None:
                    first = round(shot["delay_ms"] * 1e-3 * CLK)
                    # Husky: glitch i starts ext_offset[i] + 2 cycles after the start of glitch i - 1
                    scope.glitch.ext_offset = (first if args.pulses == 1 else
                                               [first] + [round(args.gap_ms * 1e-3 * CLK) - 2] * (args.pulses - 1))
                if args.char:
                    scope.trigger.level = shot["level"]
                elif shouter is not None:
                    row.update(voltage=args.voltage, width_ns=args.width_ns)
                    if args.pulses > 1:
                        row.update(pulses=args.pulses, pulse_gap_ms=args.gap_ms)
                    if not shouter.ready():
                        print(f"[{i}] ChipSHOUTER not armed ({shouter.last}), retry next pair", flush=True)
                        # it once rebooted ~7 s into every re-charge (2026-10-31, 210 V) and never armed
                        if shouter.resets - resets_at_shot >= 3:
                            raise RuntimeError(f"ChipSHOUTER reset {shouter.resets - resets_at_shot} times "
                                               "without a shot in between; stopping")
                        continue
                else:
                    fc_delay_us = round(shot["gap_ms"] * 1e3) - args.width_us if shot["repeat"] > 1 else 0
                    scope.glitch.ext_offset = round((shot["delay_ms"] - fc_delay_us / 1e3) * 1e-3 * CLK)
                    emfi.repeat, emfi.delay_us = shot["repeat"], fc_delay_us
                    row.update(repeat=shot["repeat"], gap_ms=shot["gap_ms"], fc_delay_us=fc_delay_us)
                    emfi.arm()
                    if not emfi.wait_for_charged(0.5):
                        emfi.disarm()
                        print(f"[{i}] HV not charged in time, retry next pair", flush=True)
                        continue
                    emfi.fire(trigger_timeout_ms=1500)
                wait = t_pred - args.lead_ms / 1e3 - time.monotonic()
                if wait < 0.003:
                    if emfi is not None:
                        emfi.disarm()
                    print(f"[{i}] late by {-wait * 1e3:.0f} ms, retry next pair", flush=True)
                    continue
                time.sleep(wait)
                scope.arm()
                t_arm = time.monotonic()
                timed_out = scope.capture()
                trace = None if timed_out else scope.get_last_trace()
                row["triggered"] = not timed_out
                if args.pulses > 1:
                    row["glitches_sent"] = scope.glitch.actual_num_glitches
                if shouter is not None and not args.char:
                    cs_state, cs_faults = shouter.after_shot()
                    row["shouter"] = cs_state
                    if cs_faults:
                        row["shouter_faults"] = cs_faults
                    row["emfi"] = ("FIRED" if cs_state == "armed" and row["triggered"] else
                                   "NO_TRIG" if cs_state == "armed" else cs_state.upper())
                elif emfi is not None:
                    st = emfi.status
                    deadline = time.monotonic() + 1.0
                    while st.state == EmfiState.WAITING and time.monotonic() < deadline:
                        st = emfi.status
                    row["faultycat"] = row["emfi"] = st.state.name
                    row["width_us"], row["fc_width_us"] = args.width_us, st.pulse_width_us_actual
                    row["fc_delay_us_actual"] = st.delay_us_actual
                    if st.state != EmfiState.FIRED:
                        emfi.disarm()
                tev = sess.next_request(1.0)
                if tev is not None:
                    row["target_req_after_arm_ms"] = (tev[0] - t_arm) * 1e3
                    if tev[1] != target:
                        row["target_src_seen"] = tev[1]
                if trace is not None:
                    feat, env = features(trace, args.decimate)
                    row.update(feat)
                    if shot["delay_ms"] is not None and feat["end_ms"] is not None:
                        row["pulse_rel_end_ms"] = shot["delay_ms"] - feat["end_ms"]
                        row["in_prefill"] = PREFILL_REL_END[0] <= row["pulse_rel_end_ms"] <= PREFILL_REL_END[1]
                        if shot.get("repeat", 1) > 1:
                            row["pulse2_rel_end_ms"] = row["pulse_rel_end_ms"] + shot["gap_ms"]
                    envs.append(env.astype(np.float16))
                else:
                    envs.append(np.zeros(1, np.float16))
                res = sess.wait_fiv3(target, 3.0)
                pulsed.add(target)
                # a control that was itself a pulsed target (--every) is logged with its own row
                ctrl = None if src_ctrl in pulsed else sess.wait_fiv3(src_ctrl, 0.1)
                keys = ("kind", "infer_ms", "nan", "max_diff", "n_diff", "used_max_diff", "first_used_diff",
                        "canary_max_diff", "verify_error", "error", "chunk", "clean")
                if res is None:
                    row["outcome"] = "NO_RESULT"
                else:
                    row.update({k: res[k] for k in keys if k in res})
                    if res.get("nan"):                        # v3 worker raises -> rollout aborts
                        row["outcome"] = "NAN"
                    elif res.get("error") or res.get("verify_error") or res.get("kind") == "error":
                        row["outcome"] = "ERROR"
                    elif res.get("max_diff", 0) > 0:
                        row["outcome"] = "CORRUPTED"
                    else:
                        row["outcome"] = "OK"
                    if res.get("canary_max_diff", 0) > 0:
                        row["outcome"] += "+CANARY"
                log.write(json.dumps(row) + "\n")
                if ctrl is not None:
                    log.write(json.dumps({"control": True, "session": n_session, "src": src_ctrl,
                                          **{k: ctrl[k] for k in keys if k in ctrl}}) + "\n")
                mode = f" x2+{shot['gap_ms']:g}" if shot.get("repeat", 1) > 1 else ""
                print(f"[{i:4d}] {'char ' + format(shot['level'], '+.2f') if args.char else format(shot['delay_ms'], '6.1f') + ' ms' + mode}"
                      f" | trig {row['triggered']!s:5s} | emfi {row.get('emfi', '-'):8s}"
                      f" | arm->req {row.get('target_req_after_arm_ms', float('nan')):5.1f} ms"
                      f" | end {row.get('end_ms') or float('nan'):6.1f} rel_end {row.get('pulse_rel_end_ms', float('nan')):6.1f}"
                      f"{' PREFILL' if row.get('in_prefill') else '        '} peak {row.get('peak_abs', 0):.2f}@{row.get('peak_ms', 0):.1f}"
                      f" | {row['outcome']} max_diff {row.get('max_diff')}", flush=True)
                if args.every and tev is not None and tev[1] == target:
                    carry = tev                       # arm for the very next inference
                i += 1
                outcome = row["outcome"]
                if shouter is not None:
                    resets_at_shot = shouter.resets
                if i % 25 == 0 and hot():
                    break
                if "CANARY" in row["outcome"]:
                    raise RuntimeError("canary mismatch: persistent corruption on the Jetson; stopping")
                if row["outcome"] in ("ERROR", "NO_RESULT") or row["outcome"] == "NAN" and not args.nan_continue:
                    break                                     # v3 aborts on these; start a new session
            hung = outcome == "NO_RESULT"
            # --move: leave time for the ~6 s return to rest, or the leftover kill below cuts it off
            sess.close(timeout=(15 if hung else max(10, dur)) + (10 if args.move else 0))
            print(f"--- session {n_session} ended: {sess.aborted or 'finished'}", flush=True)
            if alive():
                left = kill_leftovers()
                if left:
                    print(f"--- rollout still running after the session ended (GPU hang?): killed {' '.join(left)}",
                          flush=True)
                    log.write(json.dumps({"crash": True, "hang": True, "session": n_session, "after_shot": i - 1,
                                          "t": time.time()}) + "\n")
            if not alive():
                n_crash += 1
                log.write(json.dumps({"crash": True, "session": n_session, "after_shot": i - 1, "t": time.time()}) + "\n")
                hv_off()
                if n_crash > args.max_crashes:
                    print(f"Jetson crashed {n_crash} times (limit {args.max_crashes}). Stopping.", flush=True)
                    break
                if not recover(n_crash):
                    break
            if args.sessions and n_started >= args.sessions:
                break
    finally:
        try:
            if shouter is not None and not shouter.close():
                print(f"!!! ChipSHOUTER did not confirm it is disarmed ({shouter.last}): disarm it by hand", flush=True)
            if emfi is not None:
                emfi.disarm()
        finally:
            if client is not None:
                client.__exit__(None, None, None)
            if shouter is None:
                scope.io.glitch_trig_mcx = "trigger"
            scope.glitch.enabled = False      # MCX left on the glitch path: inert (husky.setup)
            scope.dis()
            if sess is not None and sess.p.poll() is None:
                sess.stop(timeout=60)
                print(f"--- rollout stopped early: "
                      f"{'arm back at rest' if sess.at_rest else 'NO arm-back-at-rest line, check the arm'}", flush=True)
            np.savez_compressed(args.out + "_env.npz", *envs)
            log.close()
            console.close()


if __name__ == "__main__":
    main()
