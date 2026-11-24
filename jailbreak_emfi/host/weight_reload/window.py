"""Glitch window (ms after the scope edge) for wiping layer L's down_proj, from every full wipe so far.

vLLM's layerwise reload quantizes modules in checkpoint order (hunt.window.checkpoint_order; per layer
down_proj, gate_up, o_proj, qkv), so each full wipe dates its layer's gate_up: G = t + shift(module).
G(L) is the median for L, else interpolated/extrapolated from the nearest dated layers. The old CSV is an
earlier hunt; the load timing drifts between sessions, so its G values are shifted by the median offset of
the newer hits, and within a session each layer is dated by its last few full wipes.

Usage: window.py EXPERIMENT_TOML LAYER HITLESS_TRIALS -> prints "start,end" (ms)."""

import csv
import os
import re
import statistics
import sys

from common import config

TENSOR_RE = re.compile(r"layers\.(\d+)\.(?:mlp|self_attn)\.(\w+?)\.weight")


def dated(path, pos, shift, min_elems):
    """{checkpoint position: [gate_up times]} from the full-tensor wipes in a hits CSV. Only full wipes date
    a module; small scattered hits (8-48 elements) land anywhere over ~30 ms after it."""
    g = {}
    if not os.path.exists(path):
        return g
    for r in csv.DictReader(open(path)):
        m = TENSOR_RE.match(r["tensors"])
        if m and m.group(2) in shift and int(r["changed_elems"].split(";")[0]) > min_elems:
            g.setdefault(pos[int(m.group(1))], []).append(float(r["primary_ms"]) + shift[m.group(2)])
    return g


def predict(known, p, per_layer, slope_min, slope_max):
    if p in known:
        return known[p]
    lo = max((q for q in known if q < p), default=None)
    hi = min((q for q in known if q > p), default=None)
    if lo is not None and hi is not None:
        return known[lo] + (known[hi] - known[lo]) * (p - lo) / (hi - lo)
    q = lo if lo is not None else hi
    # slope from the two nearest dated layers on that side (clamped), else per_layer
    side = sorted((r for r in known if (r < p) == (q < p)), key=lambda r: abs(r - p))
    slope = per_layer
    if len(side) > 1 and side[0] != side[1]:
        slope = min(slope_max, max(slope_min, (known[side[0]] - known[side[1]]) / (side[0] - side[1])))
    return known[q] + slope * (p - q)


def main():
    cfg = config.load("host.toml", sys.argv[1])
    layer, hitless = int(sys.argv[2]), int(sys.argv[3])
    w = "hunt.window."
    pos = {l: i for i, l in enumerate(cfg.get(w + "checkpoint_order", int))}
    # ms to add to a module's wipe time to get its layer's gate_up time (down_proj is read before gate_up)
    shift = {k: float(v) for k, v in cfg.get(w + "gate_up_shift_ms").items()}
    min_elems = cfg.get(w + "dated_min_elems", int)
    pred = lambda known, p: predict(known, p, cfg.get(w + "per_layer_ms", float),
                                    cfg.get(w + "slope_min_ms", float), cfg.get(w + "slope_max_ms", float))
    old = {q: statistics.median(v) for q, v in dated(cfg.get("hunt.old_hits_csv"), pos, shift, min_elems).items()}
    new = dated(cfg.get("hunt.hits_csv"), pos, shift, min_elems)
    shift = statistics.median([t - pred(old, q) for q, v in new.items() for t in v]) if new else 0.0
    known = {q: g + shift for q, g in old.items()}
    known.update({q: statistics.median(v[-cfg.get(w + "recent_hits", int):]) for q, v in new.items()})
    g = pred(known, pos[layer])
    # down_proj wipes landed 2-30 ms before their layer's gate_up median, and narrow windows that tried to
    # dodge the neighbouring gate_ups starved some layers; aim [G - start_before, G - end_before], widened
    # at the start while no hit comes.
    widen = cfg.get(w + "widen_ms", float) * (hitless // cfg.get(w + "widen_every_trials", int))
    start, end = g - cfg.get(w + "start_before_ms", float), g - cfg.get(w + "end_before_ms", float)
    override = os.path.join(cfg.get(w + "override_dir"), f"window_L{layer}.txt")
    if os.path.exists(override):
        print(open(override).read().strip())
    else:
        print(f"{start - widen:.0f},{end:.0f}")


if __name__ == "__main__":
    main()
