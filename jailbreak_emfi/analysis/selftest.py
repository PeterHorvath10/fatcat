"""Offline self-tests of the analysis code (no instruments, no judge calls, no run folders needed).

usage: PYTHONPATH=<jailbreak_emfi> python -m analysis.selftest
"""

from __future__ import annotations

import json
import struct
import tempfile
from pathlib import Path

import numpy as np

from common import config
from analysis.dram.dram_map import self_test as dram_self_test
from analysis.judge import agreement, count
from analysis.judge.build_input import judge_rows
from analysis.judge.judge import parse
from analysis.judge.labels import load_labels, by_answer
from analysis.scope import trace as st
from host.emfi import scope


def test_config_empty() -> None:
    cfg = config.load("analysis.toml")
    for key in ("scope.rms_bin_s", "judge.models", "judge.ollama_url", "layer_rules.gemma.served_model"):
        try:
            cfg.get(key)
        except config.ConfigError:
            continue
        raise AssertionError(f"shipped config has a value for {key}")


def test_judge_parse() -> None:
    cases = {"UNSAFE": "UNSAFE", "refused.": "REFUSED", " Other\n": "OTHER", "": "UNCLEAR",
             "I think this is UNSAFE, not REFUSED": "UNSAFE",                   # first word in the reply (no think block)
             "<think>maybe REFUSED</think>\nUNSAFE": "UNSAFE",                  # the answer after the reasoning wins
             "<think>it is OTHER or REFUSED</think> no idea": "REFUSED",        # else the last label word anywhere
             "UNSAFELY": "UNCLEAR", "error: HTTP Error 429: Too Many Requests": "UNCLEAR"}
    for text, want in cases.items():
        assert parse(text) == want, (text, parse(text), want)


def test_counting() -> None:
    with tempfile.TemporaryDirectory() as d:
        labs = {"judgeA_cloud": {1: "UNSAFE", 2: "REFUSED", 101: "OTHER", 102: "UNSAFE", 201: "REFUSED"},
                "judgeB_cloud": {1: "UNSAFE", 2: "UNSAFE", 101: "REFUSED", 102: "OTHER", 201: "REFUSED"}}
        for j, rows in labs.items():
            Path(d, j).mkdir()
            with open(Path(d, j, "all.labels.jsonl"), "w") as f:
                for i, l in rows.items():
                    f.write(json.dumps({"index": str(i) if i == 1 else i, "label": l, "raw": l}) + "\n")
        ans = by_answer(load_labels(d, "all"))
        r = count.summarize(ans, sorted(labs))
        assert (r["unsafe_both"], r["unsafe_both_prompts"], r["unsafe_any"], r["unsafe_any_prompts"]) == (1, 1, 3, 2), r
        assert (r["bypass_both"], r["bypass_any"], r["bypass_any_prompts"]) == (2, 4, 2), r
        assert r["judges"]["judgeA_cloud"]["unsafe"] == 2 and r["judges"]["judgeB_cloud"]["bypass_prompts"] == 2
        assert count.first_k(ans, 1).keys() == {1, 101, 201}
        assert abs(agreement.cohen_kappa(["R", "U", "R", "O"], ["R", "U", "R", "O"]) - 1) < 1e-12
    rows = judge_rows([{"prompt_slot": 3, "prompt_repeat": 2, "prompt": "p", "generated_text": "g"},
                       {"prompt_slot": 0, "prompt_repeat": 1, "prompt": "q", "generated_text": "h"}])
    assert [r["index"] for r in rows] == [1, 302] and rows[1]["goal"] == "p"


def _preamble(vdiv: float, ofs: float, cpd: float, dt: float, delay: float) -> bytes:
    p = bytearray(b"junk" + scope.WAVEDESC_MAGIC + bytes(400))
    base = 4
    values = {"v_per_div": vdiv, "offset_v": ofs, "codes_per_div": cpd, "dt_s": dt, "delay_s": delay}
    for name, (off, fmt) in scope.WAVEDESC_FIELDS.items():
        struct.pack_into("<" + fmt, p, base + off, values[name])
    return bytes(p)


def test_bursts() -> None:
    dt, n = 1e-6, 20000                                  # 20 ms at 1 MSa/s, trigger in the middle (delay 0 -> t0 = -10 ms)
    v = np.full(n, 0.001, np.float32)
    t = -n * dt / 2 + np.arange(n) * dt
    starts = [0.0002 + k * 1e-3 for k in range(10)]      # 10 "layers": 400 us bursts every 1 ms from the trigger
    for s0 in starts:
        v[(t >= s0) & (t < s0 + 4e-4)] = 0.05
    pulse_t = starts[8] + 1e-4
    v[np.argmin(abs(t - pulse_t - 1e-6))] = 0.5            # pulse 1 us (latency) after the nominal delay
    codes = np.round(v / 0.1 * 25).astype(np.int16)       # 0.1 V/div, 25 codes/div
    vv, t0, dt2, vdiv = st.decode_waveform(codes, _preamble(0.1, 0.0, 25.0, dt, 0.0))
    assert abs(t0 + 0.01) < 1e-9 and abs(dt2 - dt) < 1e-12 and abs(vdiv - 0.1) < 1e-7   # float32 fields
    bp = st.BurstParams(rms_bin_s=1e-5, idle_floor_v=0.008, top_percentile=95, active_frac=0.3, max_gap_bins=6,
                        min_burst_s=1.5e-4, min_burst_end_s=0.0)
    b = st.find_bursts(st.rms_bins(vv, dt2, bp.rms_bin_s), t0, bp)
    assert len(b) == 10 and all(abs(x - s) < 2e-5 for (x, _), s in zip(b, starts)), b
    rule = st.LayerRule("x", "x", 8, 3e-4, 6e-4, 1.3e-3, 0.71)
    assert st.is_regular(b, rule)
    tp, peak = st.pulse_time(vv, t0, dt2, pulse_t, st.PulseParams(1e-6, 3e-6))   # search wider than one sample at 1 MSa/s
    assert abs(tp - pulse_t - 1e-6) < 2e-6 and peak > 0.4
    assert st.burst_at(b, tp) == 8
    assert st.describe_hit(b, True, 8, tp, rule)[1].startswith("in burst L8 (layer 8, gate_up_proj)")
    assert st.find_bursts(np.full(100, 0.002), 0.0, bp) == []                       # idle trace


def main() -> None:
    for t in (test_config_empty, test_judge_parse, test_counting, test_bursts, dram_self_test):
        t()
        print("ok", t.__name__)


if __name__ == "__main__":
    main()
