#!/usr/bin/env python3
"""EMFI on GPU memory loads: gpumem.cu on the Jetson, the ChipSHOUTER fired by the Husky.

gpumem fills a buffer far larger than the GPU's L2 with a known pattern; one shot is one kernel
that streams it from DRAM for --kernel-ms (~96 GB/s) and compares every loaded word with its
expected value (computed from the address, never loaded). Per shot:

  ChipSHOUTER armed (it stays armed between shots) -> "go ID IDLE_MS" -> Husky armed (~70 ms)
  -> TIO4 driven high --trig-ms after the go line (software trigger; the kernel starts 13-14.5 ms
  later, measured 2026-10-31) -> glitch at ext_offset = onset + delay -> EM pulse (the Husky
  glitch length is the pulse width).
  Reply: every corrupted load (word, expected, loaded, re-read 20 us later, GPU time since the
  kernel start), the words actually checked, and a post-shot pass (L2 flushed) for corruption
  left in DRAM, which gpumem rewrites.

Not a level trigger on the probe: idle spikes on it reach the same 0.22 as the kernel, so the
Husky fired on the first one after arming, ~150 ms before the kernel (tested 2026-10-31). The
kernel does show in the probe's RMS (0.018 idle, 0.040 busy: the 10-25 MHz band x7), so each
capture measures the kernel's onset, and the host timing holds it within ~1.5 ms. --delays are
ms into the kernel: ext_offset adds the running median of the measured onsets, and every shot
logs where its pulse actually landed (pulse_in_kernel_ms = ext_offset - its own onset). For a
corrupted load, gpumem's GPU timestamp gives the same thing independently (first_bad_ms).

--mode write: the shot stores the buffer once (2 GB, ~23 ms) with a new pattern, and the pulse
hits the writes; the buffer is then read back from DRAM ~30 ms later. MEM is then a corrupted (or
lost) write, and first_bad_ms is when the first bad word was stored.

--mode sm: no memory under test; every thread runs a register-only chain and stores its state at
100 checkpoints (gpumem.cu). Outcome SM: thread states differ from the fault-free run; each record
gives the thread's SM and the first wrong checkpoint, and how many checkpoints are wrong (1: only
that store was hit; all later ones: the register state itself was corrupted).

Outcomes: OK; LOAD (loads read wrong, DRAM fine afterwards); MEM (DRAM content wrong after the
shot); +COUNT (the kernel checked fewer or more words than it should: control flow); ERROR (CUDA
error, gpumem restarted); NO_REPLY (hang or crash, recovered as in fi_v3_run.py).

Every --control-every-th shot is a control: the Husky is not armed, so nothing fires.
--char: no EMFI (ChipSHOUTER held disarmed, glitch module off), every shot captured: where do
the kernel's onset and end fall after the trigger? Its median onset is the --onset-ms to use.

The sense probe stays on its capacitor; the injection tip is the spot under test.

usage: gpumem_run.py OUT --char --shots 20
       gpumem_run.py OUT --voltage 260 --width-ns 80 --delays 5:95:10 --shots 20
       gpumem_run.py OUT --voltage 260 --mb 1 ...      (buffer inside the 2 MB L2: loads hit L2)
       gpumem_run.py OUT --voltage 260 --width-ns 80 --mode write --delays 2:20:2 --shots 20
writes OUT.jsonl (gpumem's ready line, then one row per shot), OUT_rms.npz (probe RMS per shot, 0.5 ms bins)
"""

import argparse
import json
import os
import random
import select
import statistics
import subprocess
import time
from pathlib import Path

import chipwhisperer as cw
import numpy as np
from fi_v3_run import alive, recover
from husky import CLK, parse_delays, setup, setup_glitch
from shouter import PORT as SHOUTER_PORT
from shouter import WIDTH_NS, Shouter

from site_config import JETSON, JETSON_DIR
REMOTE_DIR = f"{JETSON_DIR}/gpumem"                  # survives reboots, /tmp does not
NVCC = "/usr/local/cuda/bin/nvcc -O3 -arch=sm_87"   # Orin
RMS_BIN_MS = 0.5
KEYS = ("kernel_ms", "gpu_ms", "passes", "checked", "expected", "n_bad", "recs", "verify_bad", "verify_recs",
        "restored", "t_wait_ms", "seed", "seed_prev")


def deploy():
    """Copy gpumem.cu to the Jetson and build it there if it changed (scp -p keeps the mtime)."""
    src = Path(__file__).resolve().parents[1] / "jetson" / "gpumem.cu"
    ssh = ["ssh", "-o", "BatchMode=yes", JETSON]
    subprocess.run(ssh + [f"mkdir -p {REMOTE_DIR} fi_logs"], check=True, timeout=30)
    subprocess.run(["scp", "-pq", str(src), f"{JETSON}:{REMOTE_DIR}/"], check=True, timeout=60)
    subprocess.run(ssh + [f"cd {REMOTE_DIR} && {{ [ gpumem -nt gpumem.cu ] || {NVCC} -o gpumem gpumem.cu; }}"],
                   check=True, timeout=600)


def kill_gpumem():
    """A gpumem that never replied (GPU hang) still holds the GPU."""
    try:
        subprocess.run(["ssh", "-o", "BatchMode=yes", JETSON, "pkill -9 -x gpumem; true"], capture_output=True,
                       timeout=30)
    except subprocess.SubprocessError:
        pass


class GpuMem:
    """gpumem over SSH: "go ID IDLE_MS" -> one JSON line."""

    def __init__(self, opts):
        cmd = f"cd {REMOTE_DIR} && exec ./gpumem {opts} 2>>~/fi_logs/gpumem.stderr"
        self.p = subprocess.Popen(["ssh", "-o", "BatchMode=yes", "-o", "ServerAliveInterval=2",
                                   "-o", "ServerAliveCountMax=3", JETSON, cmd],
                                  stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
        self.ready = self.result(120)

    def go(self, shot, idle_ms):
        self.p.stdin.write(f"go {shot} {idle_ms}\n")
        self.p.stdin.flush()

    def result(self, timeout):
        r, _, _ = select.select([self.p.stdout], [], [], timeout)
        if not r:
            return None
        line = self.p.stdout.readline()
        return json.loads(line) if line else None

    def close(self):
        try:
            self.p.stdin.write("quit\n")
            self.p.stdin.flush()
            self.p.wait(timeout=10)
        except Exception:
            self.p.kill()


def features(tr, decimate, thr):
    """Kernel onset and end in the capture (ms after the trigger) from the probe RMS: 0.5 ms bins,
    median over 2.5 ms (idle bursts are shorter), above thr for 10 ms. Onset 0: the kernel was
    already running at the trigger, so the onset (and the pulse position) is unknown."""
    dt_ms = decimate / CLK * 1e3
    b = int(round(RMS_BIN_MS / dt_ms))
    n = len(tr) // b
    rms = (tr[: n * b] - np.median(tr)).reshape(n, b).std(axis=1)
    act = np.median(np.lib.stride_tricks.sliding_window_view(np.pad(rms, 2, mode="edge"), 5), axis=1) > thr
    run = int(round(10 / RMS_BIN_MS))
    on = next((j for j in range(n - run) if act[j:j + run].all()), None)
    end = None if on is None else next((j for j in range(on + run, n - run) if not act[j:j + run].any()), None)
    return {"onset_ms": None if on is None else on * b * dt_ms, "end_ms": None if end is None else end * b * dt_ms,
            "idle_rms": float(np.median(rms[:on])) if on else None,
            "busy_rms": float(np.median(rms[on:end])) if on is not None else None}, rms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--char", action="store_true", help="no EMFI: trigger and kernel alignment only")
    ap.add_argument("--voltage", type=int, help="ChipSHOUTER capacitor bank, 150-500 V (required unless --char)")
    ap.add_argument("--width-ns", type=int, default=80, help="ChipSHOUTER pulse = Husky glitch length")
    ap.add_argument("--cs-highz", action="store_true", help="ChipSHOUTER trigger input ~1.8 kohm instead of 50 ohm")
    ap.add_argument("--cs-max-temp", type=int, default=65, help="stop if a ChipSHOUTER sensor reaches this (C)")
    ap.add_argument("--delays", default="5:95:10", help="pulse, ms into the kernel: a:b:step or a,b,c")
    ap.add_argument("--shots", type=int, default=10, help="per delay (--char: in total)")
    ap.add_argument("--control-every", type=int, default=5, help="every Nth shot is a control without pulse; 0: none")
    ap.add_argument("--period-s", type=float, default=1.3,
                    help="at least this long between pulses (ChipSHOUTER heat)")
    ap.add_argument("--trig-ms", type=float, default=270, help="TIO4 trigger this long after the go line")
    ap.add_argument("--onset-ms", type=float, default=14,
                    help="kernel onset after the trigger, until measured (median of --char)")
    ap.add_argument("--window-ms", type=float, default=140, help="capture after the trigger")
    ap.add_argument("--rms-thr", type=float, default=0.031, help="probe RMS that counts as the kernel running")
    ap.add_argument("--gain-db", type=float, default=10)
    ap.add_argument("--adc-level", type=float,
                    help="trigger on the near-field probe instead of TIO4: Husky ADC level (-0.5..0.5); the pulse "
                         "lands delay ms after the probe crosses it (the kernel onset)")
    ap.add_argument("--live", help="write each shot's probe capture + row here (live_view.py shows it)")
    ap.add_argument("--arm-ms", type=float, default=200,
                    help="with --adc-level: arm the Husky this long after go (after the go burst, which trips the "
                         "level, and before the kernel ~284 ms after go)")
    ap.add_argument("--decimate", type=int, default=75)
    ap.add_argument("--mode", default="read", choices=("read", "write", "sm"),
                    help="pulse the loads, the stores, or register-only SM work")
    ap.add_argument("--mb", type=float, help="gpumem buffer (default 256, write 2048); below the 2 MB L2 the loads hit L2")
    ap.add_argument("--kernel-ms", type=float, default=100, help="read mode (write: one pass over --mb)")
    ap.add_argument("--pattern", default="hash", choices=("hash", "zeros", "ones", "alt"))
    ap.add_argument("--load", default="cg", choices=("cg", "ca", "cv", "nc"), help="PTX cache operator of the loads")
    ap.add_argument("--idle-ms", type=float, default=300, help="GPU idle before each kernel")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-crashes", type=int, default=5, help="stop after this many Jetson crashes")
    args = ap.parse_args()
    if args.mb is None:
        args.mb = 2048 if args.mode == "write" else 1 if args.mode == "sm" else 256
    assert args.char or args.voltage is not None, "--voltage is required unless --char"
    step = round(1e9 / CLK)
    assert step <= args.width_ns <= WIDTH_NS[1] and args.width_ns % step == 0, \
        f"--width-ns must be {step}-{WIDTH_NS[1]} in steps of {step} ns (Husky glitch clock)"
    trig_cycles = round(args.width_ns * 1e-9 * CLK)

    if args.char:
        plan = [None] * args.shots
    else:
        delays = [d for d in parse_delays(args.delays) for _ in range(args.shots)]
        assert 0 <= min(delays), "delays are ms into the kernel"
        random.Random(args.seed).shuffle(delays)
        plan = []
        for k, d in enumerate(delays):
            if args.control_every and k % args.control_every == 0:
                plan.append(None)
            plan.append(d)

    opts = f"--mode {args.mode} --mb {args.mb:g} --ms {args.kernel_ms:g} --pattern {args.pattern} --load {args.load}"
    log = open(args.out + ".jsonl", "a", buffering=1)
    rmss = []
    onsets = []                                         # measured, for the running median
    shouter = None
    if not args.char or Path(SHOUTER_PORT).exists():
        # connected and disarmed before the Husky is touched: setup_glitch puts the long scope
        # trigger on the MCX for a while
        shouter = Shouter(None if args.char else args.voltage, None if args.char else args.width_ns,
                          term_50=not args.cs_highz)
    scope = cw.scope()
    g = None
    i, n_crash, resets_at_shot, t_pulse = 0, 0, 0, 0.0

    def start():
        gm = GpuMem(opts)
        r = gm.ready or {}
        if not r.get("ready"):
            gm.close()
            return None
        log.write(json.dumps({**r, "opts": opts, "t": time.time()}) + "\n")
        print(f"gpumem {r['mode']}: {r['mb']:.0f} MB {r['pattern']}, ld.{r['load']}, {r['passes']} passes = "
              f"{r['kernel_ms']:.1f} ms ({r['mb'] / r['pass_ms']:.1f} GB/s), self-test {r['selftest_shots']} shots "
              f"{r['selftest_bad']} bad", flush=True)
        if r["selftest_bad"]:
            raise RuntimeError("gpumem self-test reads wrong without any pulse; stopping")
        return gm

    def hv_off():
        if shouter is not None and not shouter.disarm():
            raise RuntimeError(f"ChipSHOUTER did not disarm ({shouter.last}); stopping")

    def restart(why):
        """gpumem exited (CUDA error) or hangs (no reply): start it again, after killing it, or after
        the Jetson's crash recovery (reset through the Arduino if gpumem cannot start). None: stop."""
        nonlocal n_crash
        g.close()
        reset = False
        if alive():
            kill_gpumem()
            gm = start()
            if gm is not None:
                return gm
            reset = True                      # the GPU does not come back without a reset
        n_crash += 1
        log.write(json.dumps({"crash": True, "why": why, "reset": reset, "after_shot": i - 1, "t": time.time()}) + "\n")
        hv_off()
        if n_crash > args.max_crashes:
            print(f"Jetson crashed {n_crash} times (limit {args.max_crashes}). Stopping.", flush=True)
            return None
        if not recover(n_crash, reset=reset):
            return None
        return start()

    def hot():
        """Log the ChipSHOUTER temperatures; True (stop) if one reached --cs-max-temp."""
        if shouter is None or args.char:
            return False
        t, v = shouter.temps(), shouter.measured_v()
        log.write(json.dumps({"shouter_temps": t, "shouter_v_measured": v, "after_shot": i - 1, "t": time.time()}) + "\n")
        print(f"--- ChipSHOUTER mosfet/transformer/diode {t} C, bank {v} V", flush=True)
        if t is not None and max(t) >= args.cs_max_temp:
            print(f"ChipSHOUTER at {max(t)} C (limit {args.cs_max_temp}). Stopping.", flush=True)
            return True
        return False

    try:
        deploy()
        setup(scope, args.gain_db)                            # glitch off, MCX inert
        if not args.char:
            setup_glitch(scope, -0.12, trig_cycles)            # (its ADC level is replaced below)
            scope.glitch.arm_timing = "before_scope"
        if args.adc_level is not None:                         # hardware trigger: the probe edge itself
            scope.trigger.module = "ADC"
            scope.trigger.level = args.adc_level
        else:
            scope.trigger.module = "basic"                     # software trigger: TIO4 driven high
            scope.trigger.triggers = "tio4"
            scope.adc.basic_mode = "high"
        scope.io.tio4 = "gpio_low"
        scope.adc.presamples = 0
        scope.adc.decimate = args.decimate
        scope.adc.samples = int(args.window_ms * 1e-3 * CLK / args.decimate)
        scope.adc.timeout = 1.0
        n_pulse = sum(d is not None for d in plan)
        print(f"Husky: " + (f"ADC level trigger {args.adc_level:g} on the probe, armed {args.arm_ms:g} ms after go"
                            if args.adc_level is not None else f"TIO4 trigger {args.trig_ms:g} ms after go")
              + f" ({args.gain_db:g} dB), {scope.adc.samples} samples = "
              f"{scope.adc.samples * args.decimate / CLK * 1e3:.0f} ms; plan {len(plan)} shots"
              + ("" if args.char else f", {n_pulse} pulses, {len(plan) - n_pulse} controls"), flush=True)
        if not args.char:
            print(f"ChipSHOUTER: {args.voltage} V, pulse {args.width_ns} ns = Husky glitch {trig_cycles} cycles, "
                  f"trigger input {'high-Z' if args.cs_highz else '50 ohm'}", flush=True)
        g = start()
        if g is None:
            raise RuntimeError("gpumem did not start")
        last = max((d for d in plan if d is not None), default=0)
        if last >= g.ready["kernel_ms"] - 1:
            raise RuntimeError(f"pulse at {last:g} ms, but the kernel lasts {g.ready['kernel_ms']:.1f} ms")

        while i < len(plan):
            delay = plan[i]
            pulse = delay is not None
            row = {"n": i, "delay_ms": delay, "control": not args.char and not pulse, "t": time.time()}
            if pulse:
                time.sleep(max(0.0, t_pulse + args.period_s - time.monotonic()))
                row.update(voltage=args.voltage, width_ns=args.width_ns)
                if not shouter.ready():
                    print(f"[{i}] ChipSHOUTER not armed ({shouter.last}), retrying", flush=True)
                    if shouter.resets - resets_at_shot >= 3:
                        raise RuntimeError(f"ChipSHOUTER reset {shouter.resets - resets_at_shot} times "
                                           "without a shot in between; stopping")
                    # Queried back to back while charging, it rebooted ~7 s into every charge, 3 times
                    # (2026-10-31, 260 V); polled every 0.5 s it armed in 1.6 s.
                    time.sleep(1.0)
                    continue
                onset_est = 0.0 if args.adc_level is not None else \
                    statistics.median(onsets[-10:]) if onsets else args.onset_ms
                row.update(onset_est_ms=onset_est, offset_ms=onset_est + delay)
                scope.glitch.ext_offset = round((onset_est + delay) * 1e-3 * CLK)
            g.go(f"s{i}", args.idle_ms)
            t_go = time.monotonic()
            trace = None
            if (pulse or args.char) and args.adc_level is not None:
                time.sleep(max(0.0, t_go + args.arm_ms / 1e3 - time.monotonic()))
                scope.arm()                                    # past the go burst; fires on the kernel edge
                row["armed_after_go_ms"] = (time.monotonic() - t_go) * 1e3
                timed_out = scope.capture()
                row["triggered"] = not timed_out
            elif pulse or args.char:
                scope.arm()
                time.sleep(max(0.0, t_go + args.trig_ms / 1e3 - time.monotonic()))
                row["trig_late_ms"] = (time.monotonic() - t_go) * 1e3 - args.trig_ms
                scope.io.tio4 = "gpio_high"
                timed_out = scope.capture()
                scope.io.tio4 = "gpio_low"
                row["triggered"] = not timed_out
            if pulse or args.char:
                trace = None if timed_out else scope.get_last_trace()
                if pulse:
                    t_pulse = time.monotonic()
                    cs_state, cs_faults = shouter.after_shot()
                    row["shouter"] = cs_state
                    if cs_faults:
                        row["shouter_faults"] = cs_faults
                    row["emfi"] = ("FIRED" if cs_state == "armed" and row["triggered"] else
                                   "NO_TRIG" if cs_state == "armed" else cs_state.upper())
            if trace is not None:
                feat, rms = features(trace, args.decimate, args.rms_thr)
                row.update(feat)
                if feat["onset_ms"]:                           # 0: the kernel was already running
                    onsets.append(feat["onset_ms"])
                    if pulse:
                        row["pulse_in_kernel_ms"] = row["offset_ms"] - feat["onset_ms"]
                rmss.append(rms.astype(np.float16))
            else:
                rmss.append(np.zeros(1, np.float16))
            res = g.result(args.idle_ms / 1e3 + 3 * args.kernel_ms / 1e3 + 5)
            if res is None:
                row["outcome"] = "NO_REPLY"
            elif res.get("error"):
                row["outcome"], row["error"] = "ERROR", res["error"]
            else:
                row.update({k: res[k] for k in KEYS})
                row["outcome"] = ("MEM" if res["verify_bad"] else ("SM" if args.mode == "sm" else "LOAD") if res["n_bad"]
                                  else "OK")
                if res["checked"] != res["expected"]:
                    row["outcome"] += "+COUNT"
                bad = res["verify_recs"] if args.mode == "write" else res["recs"]   # when stored / loaded
                if bad:
                    row["first_bad_ms"] = min(r[0] for r in bad) / 1e3
            log.write(json.dumps(row) + "\n")
            if args.live and trace is not None:                # the scan holds the Husky: show its captures
                tmp = args.live + ".tmp.npz"
                np.savez(tmp, trace=np.asarray(trace, np.float32)[::4], dt_ms=4 * args.decimate / CLK * 1e3,
                         row=json.dumps({k: v for k, v in row.items() if k not in ("recs", "verify_recs")}))
                os.replace(tmp, args.live)
            what = "char" if args.char else "control" if not pulse else f"{delay:6.1f} ms"
            ms = lambda k: float("nan") if row.get(k) is None else row[k]
            print(f"[{i:4d}] {what:9s} | trig {row.get('triggered', '-')!s:5s} | emfi {row.get('emfi', '-'):8s}"
                  f" | onset {ms('onset_ms'):5.1f} end {ms('end_ms'):6.1f} pulse@{ms('pulse_in_kernel_ms'):5.1f}"
                  f" | kernel {ms('kernel_ms'):6.1f} ms | {row['outcome']:9s}"
                  f" bad {row.get('n_bad', '-')} verify {row.get('verify_bad', '-')}"
                  + (f" first@{row['first_bad_ms']:.2f} ms" if "first_bad_ms" in row else "")
                  + (f" | {row['error']}" if "error" in row else ""), flush=True)
            i += 1
            if shouter is not None:
                resets_at_shot = shouter.resets
            if row.get("restored") is False:
                raise RuntimeError("buffer still wrong after rewriting it: persistent damage; stopping")
            if row["outcome"] in ("ERROR", "NO_REPLY"):
                g = restart(row["outcome"])
                if g is None:
                    break
            if i % 25 == 0 and hot():
                break
    finally:
        try:
            if shouter is not None and not shouter.close():
                print(f"!!! ChipSHOUTER did not confirm it is disarmed ({shouter.last}): disarm it by hand", flush=True)
        finally:
            scope.io.tio4 = "high_z"
            if shouter is None:
                scope.io.glitch_trig_mcx = "trigger"
            scope.glitch.enabled = False      # MCX left on the glitch path: inert (husky.setup)
            scope.dis()
            if g is not None:
                g.close()
            np.savez_compressed(args.out + "_rms.npz", *rmss)
            log.close()
            if onsets:
                q = np.percentile(onsets, [0, 25, 50, 75, 100])
                print(f"kernel onset after the trigger: median {q[2]:.1f} ms, IQR {q[1]:.1f}-{q[3]:.1f}, "
                      f"range {q[0]:.1f}-{q[4]:.1f} ({len(onsets)} shots)", flush=True)


if __name__ == "__main__":
    main()
