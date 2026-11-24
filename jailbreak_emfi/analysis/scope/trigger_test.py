"""Live scope-trigger test: which trigger setting puts the scope trigger at the same point of the first MLP burst
(layer 0) every time?

No glitcher, no pulse. Per request: the scope is armed (SINGLE), one chat request goes to the Jetson's vLLM server
(few output tokens, the JailbreakBench harmful prompts in turn), the near-field channel is read, and the layer
bursts are found with analysis.scope.trace. The original trigger settings are saved first and restored at the end.

usage: PYTHONPATH=<jailbreak_emfi> python -m analysis.scope.trigger_test N_PER_SETTING SETTING [SETTING ...]
  SETTING: interval | nedge:LEVEL_V:N:IDLE_S   (e.g. nedge:0.075:60:2E-3)
  interval = the scope's own (saved) trigger; nedge = trigger on the N-th rising edge after IDLE_S without edges.
Writes into trigger_test.out_dir: captures/<setting>_<i>.npz, results_<time>.json, trigger_settings_before.json.

This is the only analysis script that talks to instruments; run it on purpose only.
"""

from __future__ import annotations

import contextlib
import json
import sys
import time
from pathlib import Path

import numpy as np

from common import config, http
from host.emfi import scope as scopeio
from analysis.scope import trace as st

def nedge_commands(channel: str, level: str, n: str, idle: str) -> list[str]:
    return [":TRIGger:TYPE NEDGe", f":TRIGger:NEDGe:SOURce {channel}", ":TRIGger:NEDGe:SLOPe RISing",
            f":TRIGger:NEDGe:LEVel {level}", f":TRIGger:NEDGe:IDLE {idle}", f":TRIGger:NEDGe:EDGE {n}"]


def setting_commands(name: str, channel: str) -> list[str] | None:
    """None for 'interval' (restore the saved trigger), else the Nth-edge commands."""
    if name == "interval":
        return None
    _, level, n, idle = name.split(":")
    return nedge_commands(channel, level, n, idle)


def chat(url: str, model: str, prompt: str, max_tokens: int, temperature: float, timeout_s: float) -> None:
    http.post(url, {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens,
                    "temperature": temperature}, timeout_s)


def main() -> None:
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    n_per = int(sys.argv[1])
    settings = sys.argv[2:]
    cfg = config.load("analysis.toml")

    out = Path(cfg.get("trigger_test.out_dir"))
    (out / "captures").mkdir(parents=True, exist_ok=True)
    channel = cfg.get("trigger_test.scope_channel")
    bp = st.BurstParams.from_config(cfg, min_burst_end_s=cfg.get("trigger_test.min_burst_end_s", float))
    prompts = [l.strip() for l in open(cfg.get("trigger_test.prompts_file")) if l.strip()]
    prompts = prompts[:cfg.get("trigger_test.n_prompts", int)]
    url, model = cfg.get("trigger_test.vllm_url"), cfg.get("trigger_test.model")
    max_tokens, temp = cfg.get("trigger_test.max_tokens", int), cfg.get("trigger_test.temperature", float)
    req_timeout = cfg.get("trigger_test.request_timeout_s", float)
    polls, poll_s = cfg.get("trigger_test.trigger_polls", int), cfg.get("trigger_test.poll_s", float)
    settle_s, pause_s = cfg.get("trigger_test.settle_s", float), cfg.get("trigger_test.pause_s", float)
    t_start = time.time()

    stack = contextlib.ExitStack()
    scope = stack.enter_context(scopeio.open_resource(scopeio.ScopeSettings(
        resource=cfg.get("trigger_test.scope_resource"), usb_vendor_id=None, usb_product_id=None, usb_serial=None,
        timeout_ms=cfg.get("trigger_test.scope_timeout_ms", int), visa_backend=cfg.get("trigger_test.visa_backend"))))
    saved = scopeio.trigger_state(scope)
    json.dump(saved, open(out / "trigger_settings_before.json", "w"), indent=1)
    print("saved trigger settings:", saved, flush=True)
    results, k = {}, 0
    try:
        for name in settings:
            cmds = setting_commands(name, channel)
            if cmds is None:
                scopeio.restore_trigger_state(scope, saved)
            else:
                for c in cmds:
                    scope.write(c)
            time.sleep(settle_s)
            rows = []
            for i in range(n_per):
                slot = k % len(prompts)
                k += 1
                scopeio.arm_single(scope)
                scope.query("*OPC?")              # armed; send the request right away (like the EMFI controller)
                chat(url, model, prompts[slot], max_tokens, temp, req_timeout)
                status = scopeio.wait_stopped(scope, polls * poll_s)
                if not status.lower().startswith("stop"):
                    print(name, i, "no trigger", flush=True)
                    scope.write(":STOP")
                    rows.append(None)
                    continue
                w = scopeio.read_waveform(scope, channel)
                codes, pre = w.codes, w.preamble
                np.savez_compressed(out / "captures" / f"{name.replace(':', '_')}_{i:03d}.npz", c2=codes,
                                    preamble=np.frombuffer(pre, np.uint8),
                                    meta=json.dumps({"setting": name, "prompt_slot": slot}))
                v, t0, dt = w.volts, w.t0_s, w.dt_s
                b = st.find_bursts(st.rms_bins(v, dt, bp.rms_bin_s), t0, bp)
                s0 = [x for x, _ in b]
                first_after = next((x for x in s0 if x > -cfg.get("trigger_test.after_trigger_slack_s", float)), None)
                rows.append({"bursts_ms": [round(x * 1e3, 4) for x in s0][:12],
                             "first_burst_ms": round(s0[0] * 1e3, 4) if s0 else None,
                             "burst_containing_trigger": st.burst_at(b, 0.0),
                             "first_burst_after_trigger_ms": round(first_after * 1e3, 4) if first_after is not None else None})
                r = rows[-1]
                print(f"{name} {i}: first burst {r['first_burst_ms']} ms, trigger inside burst "
                      f"{r['burst_containing_trigger']}, bursts {r['bursts_ms'][:4]}", flush=True)
                time.sleep(pause_s)
            results[name] = rows
            json.dump(results, open(out / f"results_{int(t_start)}.json", "w"), indent=1)
    finally:
        scopeio.restore_trigger_state(scope, saved)
        mode = saved[":TRIGger:MODE?"]
        scope.write(":TRIGger:MODE AUTO" if mode.upper().startswith("AUTO") else f":TRIGger:MODE {mode}")
        print("restored trigger settings", flush=True)
        stack.close()


if __name__ == "__main__":
    main()
