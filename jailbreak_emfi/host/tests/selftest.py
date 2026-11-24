"""Host self-tests without hardware: PYTHONPATH=<jailbreak_emfi> python3 host/tests/selftest.py

- config: every shipped TOML parses; building an experiment from the empty TOMLs stops with a ConfigError
- protocol: the three acknowledgement forms, read back with the worker-side functions
- delays/plan: window and random delays, widths, prompt slots
- scope: WAVEDESC/block parsing on a synthetic preamble
- runner: the full trial loop against a fake worker thread over a socketpair, with a fake glitcher,
  a weight report, a dropped connection (reset path) and the trial CSV
"""

from __future__ import annotations

import csv
import random
import socket
import struct
import sys
import tempfile
import threading
from pathlib import Path

import numpy as np

from common import config, protocol
from host.emfi import cli, delays, plan, runner, scope, target, triallog
from host.emfi.glitcher import NS_PER_COUNT, GlitchParams

PAPER_CODE = Path(__file__).resolve().parents[2]
EXPERIMENTS = sorted((PAPER_CODE / "config" / "experiments").glob("*.toml"))
UUID = "0123456789abcdef0123456789abcdef"


def check(cond, what):
    if not cond:
        raise AssertionError(what)
    print(f"  ok  {what}")


def args(**kw):
    a = cli.parser("t", "x").parse_args([])
    for k, v in kw.items():
        setattr(a, k, v)
    return a


def test_config():
    print("config")
    for exp in EXPERIMENTS:
        cfg = config.load("host.toml", exp)
        try:
            runner.build(cfg, args())
        except config.ConfigError as exc:
            check("is empty" in str(exc), f"{exp.name}: empty TOML stops with ConfigError ({str(exc)[:50]}...)")
        else:
            raise AssertionError(f"{exp.name}: built without settings")


# Dummy values for every empty key, by key name; lists/tables where the code wants them.
DUMMY = {
    "amplitudes_v": [0.0, 1.0], "reset_command": ["true"], "layers": [1, 2], "checkpoint_order": [0, 1, 2],
    "gate_up_shift_ms": {"gate_up_proj": 0}, "ack_form": "slot", "delay_mode": "window", "source": "scope",
    "reset_method": "command", "mode": "stay", "trial_counter": "run", "channel": "C2",
    "window_start_ns": 1000, "window_end_ns": 3000, "window_step_ns": 100, "width_ns_min": 8,
    "width_ns_max": 16, "width_ns_step": 4,
}


def _fill(node, prefix=""):
    out = {}
    for k, v in node.items():
        if isinstance(v, dict):
            out[k] = _fill(v, prefix + k + ".")
        elif v == "":
            out[k] = DUMMY.get(k, "1")
        else:
            out[k] = v
    return out


def _toml(d, prefix=""):
    lines, tables = [], []
    for k, v in d.items():
        if isinstance(v, dict) and k != "gate_up_shift_ms":
            tables.append((prefix + k, v))
        elif isinstance(v, dict):
            lines.append(f"{k} = {{" + ", ".join(f"{a} = {b}" for a, b in v.items()) + "}")
        else:
            lines.append(f"{k} = " + (repr(v).replace("'", '"') if not isinstance(v, bool) else str(v).lower()))
    text = "\n".join(lines) + "\n"
    for name, t in tables:
        text += f"\n[{name}]\n" + _toml(t, name + ".")
    return text


def test_config_filled():
    print("config (every key filled with dummy values)")
    try:
        import tomllib
    except ModuleNotFoundError:
        import tomli as tomllib
    tmp = Path(tempfile.mkdtemp())
    host = tmp / "host.toml"
    hdata = _fill(tomllib.load(open(PAPER_CODE / "config" / "host.toml", "rb")))
    hdata["scope"]["resource"] = "TCPIP::192.0.2.1::INSTR"
    host.write_text(_toml(hdata))
    for exp in EXPERIMENTS:
        e = tmp / exp.name
        edata = _fill(tomllib.load(open(exp, "rb")))
        e.write_text(_toml(edata))
        cfg = config.load(host, e)
        x = runner.build(cfg, args())
        check(x.run.trials == 1, f"{exp.name}: builds with every key set")
        if exp.name.startswith("thor_fixed"):
            check(scope.settings_commands(cfg)[0].startswith(":TIMebase:SCALe"), "scope settings commands")


def test_protocol():
    print("protocol")
    for form, reader, expect in [
        ("short", protocol.request_trial, (None, None, None)),
        ("slot", protocol.request_trial_any, (7, 3, None)),
        ("slot_weights", protocol.request_prompt_trial, (7, 3, protocol.CHECK_WEIGHTS_REPORT)),
    ]:
        t = target.TargetSettings("h", 1, 1, 1, 0, form, weight_report=True)
        a, b = socket.socketpair()
        out = {}
        th = threading.Thread(target=lambda: out.setdefault("h", reader(b)))
        th.start()
        protocol.wait_for_collect(a)
        a.sendall(target.acknowledgement(t, 42, UUID, 7, 3))
        th.join(5)
        h = out["h"]
        check((h.attempt_index, h.collection_id, h.prompt_slot, h.prompt_repeat, h.check_weights)
              == (42, UUID, *expect), f"ack form {form} round trip")
        a.close(), b.close()


def test_delays():
    print("delays / plan")
    w = delays.window_counts(7_450_000, 7_450_002, 1_000_000)
    check(w == [3_725_000], "fixed delay 7.45 ms = one window element 3725000 counts")
    w = delays.window_counts(7_290_000, 7_390_000, 200)
    p = delays.DelayPicker("window", window=w, trials_per_prompt=50, rng=random.Random(0))
    check([p(k, None) for k in (0, 49, 50, 51)] == [w[0], w[49], w[0], w[1]], "window index = k mod tpp")
    p = delays.DelayPicker("random", window=w, rng=random.Random(0))
    check(all(p(k, None) in w for k in range(100)), "random delay within the window")
    check(plan.widths_counts(16, 40, 4) == list(range(8, 22, 2)), "widths 16-40 ns = counts 8..20")
    pr = plan.Prompting(20, 3)
    check([pr.slot(k) for k in (0, 19, 20)] == [(3, 1), (3, 20), (4, 1)], "prompt slot/repeat (1-based)")
    check(NS_PER_COUNT == 2, "2 ns per count")


def test_scope():
    print("scope")
    pre = bytearray(b"#9000000346" + b"WAVEDESC" + bytes(400))
    base = pre.find(b"WAVEDESC")
    for name, val in [("v_per_div", 0.05), ("offset_v", 0.0117), ("codes_per_div", 6800.0), ("dt_s", 2e-9)]:
        off, fmt = scope.WAVEDESC_FIELDS[name]
        struct.pack_into("<" + fmt, pre, base + off, val)
    struct.pack_into("<d", pre, base + 180, 0.00789)
    desc = scope.parse_wavedesc(bytes(pre))
    codes = np.array([0, 6800, -6800], "<i2")
    block = b"#16" + codes.tobytes()
    v, t0, dt = scope.to_volts(scope.parse_block(block), desc)
    check(np.allclose(v, [-0.0117, 0.05 - 0.0117, -0.05 - 0.0117], atol=1e-6), "codes -> volts")
    check(abs(t0 - (0.00789 - 3 * 2e-9 / 2)) < 1e-12 and abs(dt - 2e-9) < 1e-15, "t0 = delay - span/2")


class FakeGlitcher:
    def __init__(self):
        self.loaded = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def load(self, p, software_trigger):
        self.loaded.append((p, software_trigger))

    def arm(self):
        pass

    def disarm(self):
        pass

    def wait_external(self, timeout_s):
        return True

    def soft_start(self, busy_timeout_s):
        return 0.0


def fake_worker(sock, trials, drop_at, reports):
    """Thor reload-worker stand-in: serve `trials` trials, send a weight report, drop the link once."""
    seen = []
    for n in range(trials):
        h = protocol.request_prompt_trial(sock)
        seen.append(h)
        if n == drop_at:
            sock.close()
            return seen
        protocol.send_weight_report(sock, h.attempt_index, reports[n % len(reports)], 1, b'{"tensors": []}')
    return seen


def test_runner():
    print("runner (fake worker over a socketpair)")
    tmp = Path(tempfile.mkdtemp())
    rng = random.Random(1)
    window = delays.window_counts(7_450_000, 7_450_002, 1_000_000)
    exp = runner.Experiment(
        run=runner.RunSettings(UUID, trials=5, pause_s=0, cool_every=5, cool_s=0, trial_counter="run",
                               next_position_on_reset=False, position_index=None, indices=None, start_index=0,
                               trial_log=tmp / "trials.csv", fetch_remote=None, fetch_local_dir=None,
                               capture_dir=None, capture_channel=None, capture_wait_s=None),
        target=target.TargetSettings("fake", 0, 5, 0, 0, "slot_weights", weight_report=True),
        trigger=runner.TriggerSettings(False, 0, 1, None, None),
        planner=plan.Planner([0.0, 2.0], plan.widths_counts(16, 40, 4), 1,
                             delays.DelayPicker("window", window=window, trials_per_prompt=2, rng=rng), rng=rng),
        prompting=plan.Prompting(2, 0), stage=None, thermal=None, reset=None)
    pairs, results, resets = [], [], []

    def connect():
        a, b = socket.socketpair()
        # first worker drops the link at its 3rd trial (index 2); the second serves the rest
        n, drop = (3, 2) if not pairs else (7, -1)
        th = threading.Thread(target=lambda: results.append(fake_worker(b, n, drop, ["MATCH", "CHANGED"])))
        th.start()
        pairs.append((a, b, th))
        return a

    g = FakeGlitcher()
    positions = [(0, [1.0, 2.0, 3.0], lambda: None), (1, [4.0, 5.0, 6.0], lambda: None)]
    runner.run(exp, g, connect, None, positions, lambda: resets.append(1))
    for _, b, th in pairs:
        th.join(2)
        b.close()
    rows = list(csv.DictReader(open(tmp / "trials.csv")))
    check(list(rows[0].keys()) == triallog.FIELDS, "CSV columns as expected")
    check(len(rows) == 10 and len(resets) == 1, "2 positions x 5 trials, one reset")
    check([r["reset_hit"] for r in rows[:4]] == ["False", "False", "True", "False"], "dropped trial logged as reset")
    check(all(r["primary_delay"] == "3725000" and "secondary_delay" not in r for r in rows), "delay logged, single pulse")
    hs = results[0] + results[1]
    check([(h.prompt_slot, h.prompt_repeat) for h in hs[:4]] == [(0, 1), (0, 2), (1, 1), (1, 2)],
          "worker sees prompt slot/repeat advancing")
    check(all(h.check_weights == protocol.CHECK_WEIGHTS_REPORT for h in hs), "weight report requested")
    check(isinstance(g.loaded[0][0], GlitchParams) and g.loaded[0][1] is False, "pattern loaded, scope trigger")


def main():
    for t in (test_config, test_config_filled, test_protocol, test_delays, test_scope, test_runner):
        t()
    print("all host self-tests passed")


if __name__ == "__main__":
    sys.exit(main())
