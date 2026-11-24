#!/usr/bin/env python3
"""Host side of the load campaigns (Section 6.6): pulse the decode weight load of the LIVE rollout, keep
the load only if it makes the policy erratic, and let the arm execute that faulted policy.

Each attempt launches smolvla_212_v3_rollout_async.py on the Jetson through arm_load_fi.py (arm gated on
the load verdict) with its TensorRT worker wrapped by arm_load_worker.py (pulses the decode deserialize,
classifies the loaded model). The flow per attempt:

  worker prints "ARMLOAD {decode_ready}"  -> this host places ONE ChipSHOUTER pulse in the decode copy
  worker prints "ARMLOAD {verdict ...}"   -> outcome of the faulted load over the 8 reference plans:
      erratic -> disarm HV and let the attempt run: the arm executes the faulted policy (raw, no slew /
                 no first-action check), then go_rest on exit. Campaign stops here.
      nan / small / other -> kill the attempt and relaunch (a fresh load + pulse).

Timing (verified with no HV, 2026-11-09): the host drives TIO4 the moment it reads decode_ready and the
glitch fires ext_offset after it. The decode copy shows on the probe as an RMS rise ~287 ms after
decode_ready (+-1 ms over 5 rollout loads; the worker idles --idle-ms first), so ext_offset = onset - lag +
delay, where lag is decode_ready -> TIO4 on this host and onset a running median measured per shot from
the 450 ms capture. Every row logs where its pulse landed in the copy (pulse_in_copy_ms). An ADC level
trigger does NOT work here: idle spikes (cameras etc.) reach 0.34 at 25 dB and fire it inside the idle.

The worker also reads the decode buffers back after load and names every 16-byte block the pulse
changed (weight, layer, NaN), against a clean golden load saved once with --capture-decode-golden.

The arm only ever moves on an erratic verdict. nan loads abort the rollout (no motion); held/small loads
are relaunched without motion. One SIGINT stops everything: the rollout returns the arm to rest and the
ChipSHOUTER is disarmed.

usage: arm_load_run.py OUT --char --attempts 3                 # no HV, no motion: validate the pipeline
       arm_load_run.py OUT --voltage 500 --width-ns 80 --delays 40:56:2 --attempts 40 --duration 120 --nan-drop
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
from fi_v3_run import alive
from husky import CLK, parse_delays, setup, setup_glitch
from shouter import PORT as SHOUTER_PORT
from shouter import WIDTH_NS, Shouter

from site_config import AOT_PYTHON, CONDA_ACTIVATE, JETSON, deploy_jetson
from site_config import JETSON_DIR as REMOTE_DIR
DEPS = ("arm_load_worker.py", "arm_load_fi.py", "wmem.py", "swfi.py", "onnx_inits.py")
ENV = ("PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 "
       f"FI_LOAD_WRAPPER={REMOTE_DIR}/arm_load_worker.py FI_LOAD_VERDICT={REMOTE_DIR}/arm_verdict.json "
       f"FI_LOAD_GOLDEN={REMOTE_DIR}/arm_golden.npz FI_LOAD_DECODE_GOLDEN={REMOTE_DIR}/arm_decode_golden.npz")


def ssh(cmd, **kw):
    return subprocess.run(["ssh", "-o", "BatchMode=yes", JETSON, cmd], **kw)


def deploy():
    """Copy the Jetson-side scripts (../jetson) and data to the Jetson (site_config.JETSON_DIR)."""
    deploy_jetson()


def kill_rollout(graceful=True):
    """Stop a running attempt. graceful: SIGINT first so the rollout returns the arm to rest, then SIGKILL
    any straggler (the worker holds GPU memory)."""
    pats = ("arm_load_fi.py", "smolvla_212_v3_rollout_async.py", "arm_load_worker.py")
    if graceful:
        ssh("; ".join(f"pkill -INT -f {p}" for p in pats) + "; true", capture_output=True, timeout=30)
        time.sleep(6.0)                                     # let go_rest finish (~5 s ramp)
    ssh("; ".join(f"pkill -9 -f {p}" for p in pats) + "; true", capture_output=True, timeout=30)


def readline_to(p, timeout):
    """Next stdout line within timeout, or None (EOF/timeout)."""
    if select.select([p.stdout], [], [], max(0.0, timeout))[0]:
        line = p.stdout.readline()
        return line.rstrip("\n") if line else None
    return None


def capture_golden(args):
    """Clean load, save the eight golden reference plans. No pulse."""
    print("[golden] clean reference capture (no HV)...", flush=True)
    cmd = (f"cd {REMOTE_DIR} && rm -f arm_verdict.json && exec env {ENV} {AOT_PYTHON} -u arm_load_worker.py "
           f"--capture {REMOTE_DIR}/arm_golden.npz 2>>~/fi_logs/arm_golden.stderr")
    p = subprocess.Popen(["ssh", "-o", "BatchMode=yes", "-o", "ServerAliveInterval=5", JETSON, cmd],
                         stdout=subprocess.PIPE, text=True, bufsize=1)
    t_end = time.monotonic() + args.golden_timeout
    while time.monotonic() < t_end:
        line = readline_to(p, t_end - time.monotonic())
        if line is None:
            continue
        if line.startswith("ARMLOAD "):
            d = json.loads(line[len("ARMLOAD "):])
            if d.get("kind") == "golden":
                p.wait(timeout=10)
                print(f"[golden] saved {d['n']} plans, finite={d['finite']}", flush=True)
                return d.get("finite", False)
    p.kill()
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--char", action="store_true", help="no EMFI, no motion: validate the handshake/verdict path")
    ap.add_argument("--voltage", type=int, help="ChipSHOUTER bank, 150-500 V (required unless --char)")
    ap.add_argument("--width-ns", type=int, default=80)
    ap.add_argument("--cs-max-temp", type=int, default=65)
    ap.add_argument("--delays", default="0:80:2", help="pulse, ms after the copy onset: a:b:step or a,b,c")
    ap.add_argument("--attempts", type=int, default=60, help="max load+pulse attempts before giving up")
    ap.add_argument("--duration", type=float, default=25.0, help="arm rollout length once erratic (s)")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--period-s", type=float, default=1.5, help="min seconds between pulses (ChipSHOUTER heat)")
    ap.add_argument("--idle-ms", type=float, default=300, help="worker GPU idle after decode_ready before deserialize")
    ap.add_argument("--onset-ms", type=float, default=287, help="copy onset after decode_ready, until measured")
    ap.add_argument("--onset-thr", type=float, default=0.06,
                    help="1 ms probe RMS that counts as the copy (idle ~0.035, copy 0.08-0.12 at 25 dB)")
    ap.add_argument("--window-ms", type=float, default=450, help="probe capture from decode_ready (idle + copy)")
    ap.add_argument("--gain-db", type=float, default=25)
    ap.add_argument("--decimate", type=int, default=200)
    ap.add_argument("--capture-decode-golden", action="store_true",
                    help="with --char --attempts 1: save the clean decode bytes + weight map the labels diff against")
    ap.add_argument("--nan-drop", action="store_true",
                    help="a NaN chunk during the rollout keeps the previous finite plan instead of aborting it")
    ap.add_argument("--test-plant", help="no-HV test, with --char: region:offset:hex16 or a saved *_fault.npz "
                                         "written into the decode weights right after their copy (checks the "
                                         "bypass, labels, classifier; replays a kept erratic fault)")
    ap.add_argument("--live", help="write each attempt's probe capture + verdict here (live_view.py shows it)")
    ap.add_argument("--skip-golden", action="store_true", help="reuse an existing arm_golden.npz")
    ap.add_argument("--golden-timeout", type=float, default=240)
    args = ap.parse_args()
    assert args.char or args.voltage is not None, "--voltage is required unless --char"
    step = round(1e9 / CLK)
    assert step <= args.width_ns <= WIDTH_NS[1] and args.width_ns % step == 0
    trig_cycles = round(args.width_ns * 1e-9 * CLK)
    delays = parse_delays(args.delays)

    log = open(args.out + ".jsonl", "a", buffering=1)
    rmss, onsets = [], []                                   # onsets: ms after decode_ready, measured per shot
    shouter = None
    if not args.char or Path(SHOUTER_PORT).exists():
        shouter = Shouter(None if args.char else args.voltage, None if args.char else args.width_ns)
    scope = cw.scope()
    t_pulse = 0.0
    moved = False
    last = {}                                               # this attempt's capture, for --live
    plant_spec = args.test_plant

    def hv_off():
        if shouter is not None and not shouter.disarm():
            print(f"!!! ChipSHOUTER did not disarm ({shouter.last}); disarm by hand", flush=True)

    def arm_and_wait(budget_s=15):
        """Arm the ChipSHOUTER and wait until it reports 'armed' (the HV charges for a few seconds).
        Pre-arming before the load means a decode_ready pulse never triggers it mid-arming, which is
        what latches fault_trigger_glitch."""
        if shouter is None or args.char:
            return True
        t_end = time.monotonic() + budget_s
        while time.monotonic() < t_end:
            if shouter.ready():
                return True
            time.sleep(0.5)
        print(f"[arm] ChipSHOUTER not armed after {budget_s:g}s ({shouter.last})", flush=True)
        return False

    def copy_onset(trace, start_ms):
        """Copy onset, ms after the trigger: the first 4 consecutive 1 ms probe-RMS bins above --onset-thr
        from start_ms on (15 ms before the expected onset: the idle has short camera bursts, e.g. ~8 ms
        at 264 ms in one load, which an open search mistakes for the copy)."""
        dt = args.decimate / CLK * 1e3
        b = max(1, int(round(1.0 / dt)))
        tr = np.asarray(trace, float)
        tr = tr - np.median(tr[: int(150 / dt)])
        n = len(tr) // b
        env = tr[: n * b].reshape(n, b).std(1)
        on = next((j for j in range(int(start_ms), n - 4) if (env[j:j + 4] > args.onset_thr).all()), None)
        return None if on is None else on * b * dt

    def fire(delay, row, t_ready):
        """Place one pulse delay ms after the decode copy onset: TIO4 now (at decode_ready), glitch at
        ext_offset = onset - lag + delay, with onset the running median measured per shot (ms after
        decode_ready) and lag the time this host took from reading decode_ready to the trigger. The capture
        then measures this shot's own onset, so every row says where its pulse actually landed."""
        nonlocal t_pulse
        pulse = not args.char
        if pulse:
            time.sleep(max(0.0, t_pulse + args.period_s - time.monotonic()))
            if not shouter.ready():            # never let the Husky trigger an un-armed ChipSHOUTER
                row["emfi"] = f"SKIP_NOT_ARMED:{shouter.last}"
                rmss.append(np.zeros(1, np.float16))
                return
        onset_est = statistics.median(onsets[-8:]) if onsets else args.onset_ms
        lag = (time.monotonic() - t_ready) * 1e3
        off = onset_est - lag + delay                                # pulse, ms after the trigger
        row.update(onset_est_ms=round(onset_est, 1), lag_ms=round(lag, 1))
        if pulse:
            row.update(voltage=args.voltage, delay_ms=delay, offset_ms=round(off, 2))
            scope.glitch.ext_offset = round(off * 1e-3 * CLK)
        scope.arm()
        scope.io.tio4 = "gpio_high"                                  # trigger now: capture idle + copy
        timed_out = scope.capture()
        scope.io.tio4 = "gpio_low"
        row["triggered"] = not timed_out
        trace = None if timed_out else scope.get_last_trace()
        last.update(trace=trace, lag=lag, onset_est=onset_est, off=off)
        if pulse:
            t_pulse = time.monotonic()
            cs_state, cs_faults = shouter.after_shot()
            row["shouter"] = cs_state
            if cs_faults:
                row["shouter_faults"] = cs_faults
            row["emfi"] = ("FIRED" if cs_state == "armed" and row["triggered"] else
                           "NO_TRIG" if cs_state == "armed" else cs_state.upper())
        if trace is not None:
            on = copy_onset(trace, max(150.0, onset_est - lag - 15))  # ms after the trigger
            if on is not None:
                row["onset_ms"] = round(on + lag, 1)                 # ms after decode_ready
                onsets.append(on + lag)
                if pulse:
                    row["pulse_in_copy_ms"] = round(off - on, 2)     # where this pulse actually landed
            rmss.append(np.asarray(trace).astype(np.float16))
        else:
            rmss.append(np.zeros(1, np.float16))

    def fetch(remote, local, what):
        r = subprocess.run(["scp", "-q", f"{JETSON}:{remote}", local], capture_output=True, text=True, timeout=120)
        print(f"[keep] {what} -> {local}" if r.returncode == 0 else
              f"!!! could not fetch {what} ({r.stderr.strip()[:120]}); it is on the Jetson at {remote}", flush=True)

    def show(k, row, verdict):
        """--live: the probe capture of this attempt with the copy onset and the pulse marked (ms after the
        trigger), and its verdict, for live_view.py."""
        if not args.live or last.get("trace") is None:
            return
        on = (row["onset_ms"] if row.get("onset_ms") is not None else last["onset_est"]) - last["lag"]
        live_row = {"n": k, "voltage": args.voltage, "delay_ms": row.get("delay_ms"), "onset_est_ms": on,
                    "offset_ms": None if args.char else last["off"], "outcome": verdict.get("outcome"),
                    "n_bad": verdict.get("n_bad_blocks"),
                    "extra": f"{','.join(verdict.get('kinds') or [])} max_dev {verdict.get('max_dev_deg') or 0:.1f} deg"}
        tmp = args.live + ".tmp.npz"
        np.savez(tmp, trace=np.asarray(last["trace"], np.float32)[::4], dt_ms=4 * args.decimate / CLK * 1e3,
                 row=json.dumps(live_row))
        os.replace(tmp, args.live)
        last.clear()

    def attempt(k):
        """One launch: fire on decode_ready, return the load verdict dict (or a crash marker)."""
        delay = random.choice(delays)
        row = {"attempt": k, "delay_ms": delay, "t": time.time()}
        out = f"{REMOTE_DIR}/arm_{Path(args.out).name}_{k}"         # per run, so logs are not overwritten
        row["remote"] = out
        # the main/robot process runs in the conda lerobot env (like fi_v3_run.py); the TensorRT worker
        # it spawns stays on the AOT python (roll's start_worker), rewritten to arm_load_worker.py.
        cmd = (f"{CONDA_ACTIVATE} && cd {REMOTE_DIR} && "
               f"rm -f {REMOTE_DIR}/arm_verdict.json && printf '\\n' | env {ENV} FI_LOAD_IDLE_MS={args.idle_ms:g} "
               + ("FI_LOAD_CAPTURE_DECODE=1 " if args.capture_decode_golden else "")
               + (f"FI_LOAD_TEST_PLANT={plant_spec} " if plant_spec else "")
               + f"FI_LOAD_FAULT={out}_fault.npz "
               + ("FI_LOAD_NAN=drop " if args.nan_drop else "") +
               f"python -u {REMOTE_DIR}/arm_load_fi.py {out} smolvla_212_v3_rollout_async.py "
               f"--duration {args.duration:g} --max-tick-deg 0 --force --seed {args.seed} "
               f"2>>~/fi_logs/arm_load.stderr")
        p = subprocess.Popen(["ssh", "-o", "BatchMode=yes", "-o", "ServerAliveInterval=5",
                              "-o", "ServerAliveCountMax=4", JETSON, cmd],
                             stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, text=True, bufsize=1)
        t_end = time.monotonic() + args.idle_ms / 1e3 + 240          # model load can take a while
        fired = False
        while time.monotonic() < t_end:
            line = readline_to(p, min(5.0, t_end - time.monotonic()))
            if line is None:
                if p.poll() is not None:
                    return p, {"outcome": "NO_REPLY"}, row
                continue
            if not line.startswith("ARMLOAD "):
                if line and not line.startswith("[TRT]"):
                    print("   | " + line[:160], flush=True)
                continue
            d = json.loads(line[len("ARMLOAD "):])
            if d.get("kind") == "decode_ready" and not fired:
                t_ready = time.monotonic()
                fired = True
                fire(delay, row, t_ready)
                print(f"[{k:3d}] delay {delay:g} ms | {row.get('emfi','-')} | lag {row.get('lag_ms')} ms, "
                      f"onset {row.get('onset_ms')} ms after decode_ready, pulse landed "
                      f"{row.get('pulse_in_copy_ms')} ms into the copy", flush=True)
            elif d.get("kind") == "verdict":
                row.update({key: d.get(key) for key in (
                    "max_dev_deg", "per_obs", "decode_deser_ms", "decode_regions_mb", "n_bad_blocks", "kinds",
                    "o_proj_hit", "blocks", "golden_saved", "wmap_entries", "covered", "label_error",
                    "test_planted", "fault_saved")
                    if key in d})
                row["verdict"] = d.get("outcome")
                return p, d, row
        return p, {"outcome": "TIMEOUT"}, row

    try:
        deploy()
        if args.test_plant and Path(args.test_plant).is_file():     # a kept fault file: replay all its blocks
            plant_spec = f"{REMOTE_DIR}/plant_{Path(args.test_plant).name}"
            subprocess.run(["scp", "-q", args.test_plant, f"{JETSON}:{plant_spec}"], check=True, timeout=60)
        setup(scope, args.gain_db)
        if not args.char:
            setup_glitch(scope, -0.12, trig_cycles)             # glitch on the scope trigger (its ADC level is
            scope.glitch.arm_timing = "before_scope"            # unused: the trigger is switched to TIO4 below)
        scope.trigger.module = "basic"                          # software trigger at decode_ready
        scope.trigger.triggers = "tio4"
        scope.adc.basic_mode = "high"
        scope.io.tio4 = "gpio_low"
        scope.adc.presamples = 0
        scope.adc.decimate = args.decimate
        scope.adc.samples = int(args.window_ms * 1e-3 * CLK / args.decimate)
        scope.adc.timeout = args.window_ms / 1e3 + 0.5
        print(f"Husky: TIO4 at decode_ready, {args.gain_db:g} dB, {scope.adc.samples} samples = "
              f"{scope.adc.samples*args.decimate/CLK*1e3:.0f} ms; copy onset {args.onset_ms:g} ms until measured"
              + ("" if args.char else f"; ChipSHOUTER {args.voltage} V, {args.width_ns} ns"), flush=True)

        if not args.skip_golden and not capture_golden(args):
            raise RuntimeError("golden capture failed (no finite reference plans)")

        arm_fails = 0
        for k in range(args.attempts):
            if not alive():
                print("Jetson unreachable; stopping.", flush=True)
                break
            if not arm_and_wait():                       # armed before the load, so decode_ready can fire it
                arm_fails += 1
                if arm_fails >= 3:
                    print("ChipSHOUTER will not arm (3x); stopping.", flush=True)
                    break
                continue
            arm_fails = 0
            p, verdict, row = attempt(k)
            row["outcome"] = verdict.get("outcome")
            log.write(json.dumps(row) + "\n")
            show(k, row, verdict)
            print(f"[{k:3d}] verdict: {verdict.get('outcome')} (max_dev {verdict.get('max_dev_deg')} deg) | "
                  f"bad blocks {verdict.get('n_bad_blocks')} {','.join(verdict.get('kinds') or [])} | "
                  f"decode {verdict.get('decode_regions_mb')} MB, deserialize {verdict.get('decode_deser_ms')} ms"
                  + (f" | GOLDEN SAVED ({verdict.get('wmap_entries')} weights, {verdict.get('covered')})"
                     if verdict.get("golden_saved") else "")
                  + (f" | label error: {verdict['label_error']}" if verdict.get("label_error") else "")
                  + (f" | TEST PLANT {verdict['test_planted']}" if verdict.get("test_planted") else ""), flush=True)
            for b in (verdict.get("blocks") or [])[:4]:
                if b.get("weight") is not None:
                    print(f"        block {b.get('region')}+{b.get('offset')} {b.get('weight')} {b.get('where')} "
                          f"nan {b.get('nan')} max|v| {b.get('maxabs')}", flush=True)

            if verdict.get("outcome") == "erratic":
                hv_off()                                    # arm is about to move: no more HV
                print("\n" + "=" * 70, flush=True)
                print("*** ERRATIC LOAD KEPT IN MEMORY -- ARM IS NOW EXECUTING THE FAULTED POLICY ***",
                      flush=True)
                print(f"*** raw policy actions (no slew / no first-action check), {args.duration:g} s. "
                      f"Ctrl-C to stop early (returns the arm to rest). ***", flush=True)
                print("=" * 70, flush=True)
                moved = True
                fs = verdict.get("fault_saved") or {}
                if fs.get("path"):                           # keep the faulted weights (all blocks, both bytes)
                    fetch(fs["path"], args.out + "_fault.npz", f"faulted weights ({fs.get('blocks')} blocks)")
                elif fs.get("error"):
                    print(f"!!! fault not saved: {fs['error']}", flush=True)
                while p.poll() is None:                      # let the arm run; stream its log lines
                    line = readline_to(p, 2.0)
                    if line and not line.startswith("[TRT]"):
                        print("   | " + line[:160], flush=True)
                print("[arm] rollout finished; arm returned to rest.", flush=True)
                fetch(row["remote"] + ".npz", args.out + "_motion.npz", "motion log")
                break

            kill_rollout(graceful=False)        # non-erratic: the arm never moved, so SIGKILL (no go_rest)
            if (k + 1) % 20 == 0 and shouter is not None:
                t = shouter.temps()
                print(f"--- ChipSHOUTER temps {t} C", flush=True)
                if t and max(t) >= args.cs_max_temp:
                    print(f"ChipSHOUTER at {max(t)} C; stopping.", flush=True)
                    break
        else:
            print(f"no erratic load in {args.attempts} attempts.", flush=True)
    except KeyboardInterrupt:
        print("\n[interrupt] stopping: returning arm to rest, disarming.", flush=True)
    finally:
        hv_off()
        try:
            kill_rollout(graceful=True)
        finally:
            scope.io.tio4 = "high_z"
            if shouter is None:
                scope.io.glitch_trig_mcx = "trigger"
            scope.glitch.enabled = False
            scope.dis()
            if shouter is not None and not shouter.close():
                print(f"!!! ChipSHOUTER did not confirm disarm ({shouter.last}); check by hand", flush=True)
            if rmss:
                np.savez_compressed(args.out + "_rms.npz", *rmss)
            log.close()
            print(f"done{' (arm moved)' if moved else ''}.", flush=True)


if __name__ == "__main__":
    main()
