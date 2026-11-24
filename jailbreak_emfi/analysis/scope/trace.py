"""Near-field scope traces of an EMFI run: loading, layer bursts, and where the EM pulse lands.

One trial saves scope/tNNNNN.npz: `c2` (int16 ADC codes of the near-field probe over the GPU), `preamble`
(the scope's raw WAVEDESC block) and `meta` (JSON: trial_index, prompt_slot, prompt_repeat, primary_delay, ...).
The trace is centred on the scope's horizontal delay; t = 0 is the scope trigger (first GPU kernel of the
request).

Layer bursts: during prefill every transformer layer shows up as one high-EM burst (its MLP matmuls). We take the
rms of the trace in short bins, call a bin active when it is above a fraction of the trace's high (95th
percentile) rms level, join active runs separated by short gaps, and drop slivers. A trace is "regular" when the
bursts form a clean layer sequence (enough bursts, the first at the trigger, a plausible start-to-start spacing):
only then is burst k layer k. Otherwise the layer number is unknown (GPU stalls, idle captures, merged bursts).

Used by: pulse_layer_report.py, layer_timing_table.py, trigger_test.py.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from typing import Iterator

import numpy as np

from common.config import Config
from host.emfi import scope
from host.emfi.glitcher import NS_PER_COUNT


# VC Glitcher: one delay count is 2 ns (instrument fact).
# controller.log line that gives the delay of a fixed-time run (fallback when a trace has no primary_delay)
DELAY_LINE = re.compile(r"primary delay range (\d+)")


# ---------------------------------------------------------------- settings
@dataclass(frozen=True)
class BurstParams:
    """Burst finder settings (config section [scope])."""
    rms_bin_s: float          # rms bin width
    idle_floor_v: float       # trace with top rms below this has no GPU activity
    top_percentile: float     # percentile of the bin rms that defines the trace's high level
    active_frac: float        # bin is active above active_frac * high level
    max_gap_bins: int         # an inactive run longer than this ends a burst
    min_burst_s: float        # shorter bursts are slivers, not layers
    min_burst_end_s: float    # drop bursts that end before this time (relative to the trigger)

    @classmethod
    def from_config(cls, cfg: Config, **override) -> "BurstParams":
        p = dict(rms_bin_s=cfg.get("scope.rms_bin_s", float), idle_floor_v=cfg.get("scope.idle_floor_v", float),
                 top_percentile=cfg.get("scope.top_percentile", float),
                 active_frac=cfg.get("scope.active_frac_of_top", float),
                 max_gap_bins=cfg.get("scope.max_gap_bins", int), min_burst_s=cfg.get("scope.min_burst_s", float),
                 min_burst_end_s=cfg.get("scope.min_burst_end_s", float))
        p.update(override)
        return cls(**p)


@dataclass(frozen=True)
class LayerRule:
    """What a clean layer sequence looks like for one served model (config [layer_rules.<name>])."""
    name: str                   # on-screen / report name
    served_model: str           # vLLM model name as in results.jsonl
    min_bursts: int             # at least this many bursts
    first_burst_max_s: float    # |start of the first burst| below this (it starts at the trigger)
    spacing_min_s: float        # start-to-start spacing of consecutive bursts strictly inside (min, max)
    spacing_max_s: float
    gate_up_frac: float | None  # fraction of a burst that is gate_up_proj (None: do not name the matrix)

    @classmethod
    def for_model(cls, cfg: Config, served_model: str) -> "LayerRule":
        rules = cfg.section("layer_rules")
        for key, r in rules.items():
            if r.get("served_model") == served_model:
                g = cfg.get(f"layer_rules.{key}.gate_up_frac", float, default=None)
                return cls(name=cfg.get(f"layer_rules.{key}.name"), served_model=served_model,
                           min_bursts=cfg.get(f"layer_rules.{key}.min_bursts", int),
                           first_burst_max_s=cfg.get(f"layer_rules.{key}.first_burst_max_s", float),
                           spacing_min_s=cfg.get(f"layer_rules.{key}.spacing_min_s", float),
                           spacing_max_s=cfg.get(f"layer_rules.{key}.spacing_max_s", float), gate_up_frac=g)
        raise KeyError(f"no [layer_rules.*] with served_model = '{served_model}' in {cfg.sources}")


@dataclass(frozen=True)
class PulseParams:
    latency_s: float    # the pulse sits this long after the nominal glitcher delay (scope -> glitcher path)
    search_s: float     # look for the |v| peak within +- this of the expected time

    @classmethod
    def from_config(cls, cfg: Config) -> "PulseParams":
        return cls(cfg.get("scope.pulse_latency_s", float), cfg.get("scope.pulse_search_s", float))


# ---------------------------------------------------------------- loading
@dataclass
class Trace:
    v: np.ndarray       # volts, float32
    t0: float           # time of sample 0 relative to the trigger, s
    dt: float           # s per sample
    vdiv: float         # V per division
    meta: dict


def decode_waveform(codes: np.ndarray, preamble: bytes) -> tuple[np.ndarray, float, float, float]:
    """(volts, t0, dt, V/div) from raw ADC codes and the scope's preamble (decoding shared with the controller,
    host/emfi/scope.py, which saved the trace)."""
    desc = scope.parse_wavedesc(bytes(preamble))
    v, t0, dt = scope.to_volts(codes, desc)
    return v, t0, dt, desc["v_per_div"]


def load_trace(path: str | Path) -> Trace:
    """Read one saved trace (npz with c2, preamble, meta)."""
    with np.load(path) as z:
        v, t0, dt, vdiv = decode_waveform(z["c2"], bytes(z["preamble"]))
        meta = json.loads(str(z["meta"])) if "meta" in z.files else {}
    return Trace(v, t0, dt, vdiv, meta)


def trace_files(run_dir: str | Path) -> list[Path]:
    return sorted(Path(run_dir, "scope").glob("t*.npz"))


def served_model(run_dir: str | Path) -> str:
    """Model name of the run (first row of results.jsonl)."""
    with open(Path(run_dir, "results.jsonl")) as f:
        for line in f:
            if line.strip():
                return json.loads(line)["model"]
    raise ValueError(f"{run_dir}/results.jsonl is empty")


def nominal_delay_s(meta: dict, run_dir: str | Path | None = None) -> float:
    """Glitcher delay of this trial (meta primary_delay), else the run's fixed delay from controller.log."""
    if meta.get("primary_delay") is not None:
        return int(meta["primary_delay"]) * NS_PER_COUNT * 1e-9
    if run_dir is None:
        raise ValueError("trace has no primary_delay and no run folder was given")
    m = DELAY_LINE.search(Path(run_dir, "controller.log").read_text(errors="replace"))
    if not m:
        raise ValueError(f"no '{DELAY_LINE.pattern}' line in {run_dir}/controller.log")
    return int(m.group(1)) * NS_PER_COUNT * 1e-9


# ---------------------------------------------------------------- bursts
def rms_bins(v: np.ndarray, dt: float, bin_s: float) -> np.ndarray:
    nb = int(round(bin_s / dt))
    return np.sqrt((v[: len(v) // nb * nb].reshape(-1, nb) ** 2).mean(1))


def find_bursts(rms: np.ndarray, t0: float, p: BurstParams) -> list[tuple[float, float]]:
    """(start, end) in s of the high-EM bursts (one per layer) of one trace, from its binned rms."""
    top = np.percentile(rms, p.top_percentile)
    if top < p.idle_floor_v:
        return []
    act = rms > p.active_frac * top
    runs, b0, gap, last = [], None, 0, 0
    for q, on in enumerate(act):
        if on:
            if b0 is None:
                b0 = q
            gap, last = 0, q
        elif b0 is not None:
            gap += 1
            if gap > p.max_gap_bins:
                runs.append((b0, last))
                b0 = None
    if b0 is not None:
        runs.append((b0, last))
    min_bins = int(round(p.min_burst_s / p.rms_bin_s))
    w = p.rms_bin_s
    return [(t0 + x * w, t0 + (y + 1) * w) for x, y in runs
            if y - x + 1 >= min_bins and t0 + (y + 1) * w > p.min_burst_end_s]


def is_regular(bursts: list[tuple[float, float]], rule: LayerRule) -> bool:
    """A clean layer sequence: burst k is layer k."""
    if len(bursts) < max(rule.min_bursts, 2):
        return False
    s = np.array([x for x, _ in bursts])
    d = np.diff(s)
    return bool(abs(s[0]) < rule.first_burst_max_s and d.min() > rule.spacing_min_s and d.max() < rule.spacing_max_s)


def burst_at(bursts: list[tuple[float, float]], t: float) -> int | None:
    """Index of the burst that contains time t, or None."""
    return next((i for i, (x, y) in enumerate(bursts) if x <= t <= y), None)


def pulse_time(v: np.ndarray, t0: float, dt: float, delay_s: float, pp: PulseParams) -> tuple[float, float]:
    """(time, |v| peak) of the EM pulse: the largest |v| within +- search_s of delay + latency (the scope
    triggers the glitcher, so the pulse sits a fixed latency after the nominal delay)."""
    a = np.abs(v)
    kc = int(round((delay_s + pp.latency_s - t0) / dt))
    kw = int(round(pp.search_s / dt))
    k = kc - kw + int(np.argmax(a[kc - kw:kc + kw]))
    return t0 + k * dt, float(a[k])


# ---------------------------------------------------------------- per-trace summary (small, cacheable)
@dataclass
class Summary:
    name: str           # trace file stem, e.g. t00042
    rms: np.ndarray     # binned rms (scope.rms_bin_s)
    t0: float
    dt: float
    tp: float           # pulse time, s (nan when unknown)
    peak: float         # pulse |v| peak, V
    meta: dict

    def save(self, path: Path) -> None:
        np.savez(path, rms=self.rms, t0=self.t0, dt=self.dt, tp=self.tp, peak=self.peak,
                 meta=json.dumps(self.meta, default=str))

    @classmethod
    def load(cls, path: Path) -> "Summary":
        with np.load(path) as z:
            return cls(path.stem, z["rms"], float(z["t0"]), float(z["dt"]), float(z["tp"]), float(z["peak"]),
                       json.loads(str(z["meta"])))


def summarize(path: Path, bp: BurstParams, pp: PulseParams | None, run_dir: Path | None = None) -> Summary:
    tr = load_trace(path)
    tp, peak = float("nan"), float("nan")
    if pp is not None:
        tp, peak = pulse_time(tr.v, tr.t0, tr.dt, nominal_delay_s(tr.meta, run_dir), pp)
    return Summary(Path(path).stem, rms_bins(tr.v, tr.dt, bp.rms_bin_s), tr.t0, tr.dt, tp, peak, tr.meta)


def _summary_job(path: Path, bp: BurstParams, pp: PulseParams | None, run_dir: Path, cache: Path | None):
    out = cache / (path.stem + ".npz") if cache else None
    if out is not None and out.exists():
        s = Summary.load(out)
        if pp is None or np.isfinite(s.tp):   # a summary cached without the pulse is redone when it is needed
            return s
    try:
        s = summarize(path, bp, pp, run_dir)
    except Exception as e:   # truncated / unreadable capture: report and skip
        print(f"skip {path}: {e}")
        return None
    if out is not None:
        s.save(out)
    return s


def run_summaries(run_dir: str | Path, bp: BurstParams, pp: PulseParams | None, workers: int,
                  cache: str | Path | None = None, step: int = 1) -> Iterator[Summary]:
    """Summaries of every step-th trace of a run, in trace order. With cache, summaries are stored there
    (never inside the run folder) and reused next time; delete the cache after changing scope.rms_bin_s or
    the pulse settings."""
    run_dir = Path(run_dir)
    cache_dir = Path(cache) if cache else None
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
    files = trace_files(run_dir)[::step]
    job = partial(_summary_job, bp=bp, pp=pp, run_dir=run_dir, cache=cache_dir)
    with Pool(workers) as pool:
        for s in pool.imap(job, files, chunksize=4):
            if s is not None:
                yield s


def layer_info(s: Summary, bp: BurstParams, rule: LayerRule) -> tuple[list, bool, int | None]:
    """(bursts, regular, index of the burst that contains the pulse or None)."""
    b = find_bursts(s.rms, s.t0, bp)
    hit = burst_at(b, s.tp) if np.isfinite(s.tp) else None
    return b, is_regular(b, rule), hit


def describe_hit(bursts, regular: bool, hit: int | None, tp: float, rule: LayerRule) -> tuple[str, str]:
    """(category, text) of where the pulse lands; category is one of no_activity, irregular, in_burst,
    between, outside."""
    if not bursts:
        return "no_activity", "no GPU activity captured"
    if not regular:
        return "irregular", "irregular GPU activity, layer unknown" + (" (pulse inside a burst)" if hit is not None else "")
    if hit is not None:
        b0, b1 = bursts[hit]
        into, blen = (tp - b0) * 1e6, (b1 - b0) * 1e6
        part = ""
        if rule.gate_up_frac is not None:
            part = ", gate_up_proj" if into < rule.gate_up_frac * blen else ", down_proj"
        return "in_burst", f"in burst L{hit} (layer {hit}{part}): {into:.0f} of {blen:.0f} us"
    k = sum(1 for _, b1 in bursts if b1 < tp)    # bursts that ended before the pulse
    if 0 < k < len(bursts):
        return "between", f"between the L{k - 1} and L{k} bursts"
    return "outside", "outside the layer bursts"
