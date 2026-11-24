#!/usr/bin/env python3
"""The block simulation on the Jetson's TensorRT engines (jetson/trt_blocksim.py) against the PyTorch
one (swfi2.py): Table 6's rows and the numbers Section 6.3 quotes, side by side.
usage (from host/): trt_blocksim_compare.py TRT.npz SWFI2.npz
"""
import argparse

import numpy as np

SOURCES = (("w-vision", "weights, vision encoder (1 camera)"), ("w-prefill", "weights, prefill"),
           ("w-expert", "weights, one denoising step"), ("w-expert-all", "weights, all 10 steps (stored)"),
           ("kv-write", "KV cache, as written"), ("kv-read", "KV cache, one step's read"))


def load_trt(path):
    d = dict(np.load(path))
    eng = np.array([r.split(":")[0] for r in d["regions"]])
    where = np.where(d["region"] >= 0, eng[np.maximum(d["region"], 0)], "")
    # trt_blocksim.py marked every expert K/V projection x_*; only the prefill engine's are the cross-attention ones
    wrong = np.char.startswith(d["kind"], "x_") & (where != "prefill")
    d["kind"] = np.where(wrong, np.char.replace(d["kind"], "x_", ""), d["kind"])
    return d


def rates(d, m):
    o, dm = d["outcome"][m], d["dmax"][m]
    big = (dm >= 100) & np.isin(o, ("silent", "erratic"))
    return [100 * np.mean(o == c) for c in ("nan", "silent", "small", "erratic")] + [100 * np.mean(big)]


def pair(t, p, fmt="{:5.1f}"):
    return f"{fmt.format(t):>6s} {fmt.format(p):>6s}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trt")
    ap.add_argument("swfi2")
    a = ap.parse_args()
    t, p = load_trt(a.trt), dict(np.load(a.swfi2))
    print(f"TensorRT: {len(t['site'])} faults on {len(t['clean'])} observations; PyTorch: {len(p['site'])} faults. "
          "Each cell: TensorRT, PyTorch (%)\n")
    print(f"{'source':34s} {'n':>11s}  " + "  ".join(f"{c:>13s}" for c in ("detected", "silent", "small", "erratic", ">=100 deg")))
    for s, label in SOURCES:
        mt, mp = t["site"] == s, p["site"] == s
        print(f"{label:34s} {mt.sum():5d} {mp.sum():5d}  " + "  ".join(pair(x, y) for x, y in zip(rates(t, mt), rates(p, mp))))

    print("\nNaN rate by whether the random block itself holds a NaN or inf")
    for s, label in SOURCES[:4]:
        cells = []
        for has in (True, False):
            sel = [(d["site"] == s) & ((d["new_nonfinite"] > 0) == has) for d in (t, p)]
            cells.append(pair(*(100 * np.mean(d["outcome"][m] == "nan") for d, m in zip((t, p), sel))))
        print(f"  {label:34s} holds NaN/inf {cells[0]}   all finite {cells[1]}")

    print("\nSection 6.3")
    for s in ("w-vision", "w-prefill"):
        med = [np.median(d["dmax"][(d["site"] == s) & (d["outcome"] == "silent")]) for d in (t, p)]
        print(f"  median silent change, {s[2:]} weights (deg): {pair(*med, '{:.0f}')}")
    for lo in (0, 4, 8, 12):
        r = [rates(d, (d["site"] == "w-prefill") & (d["layer"] >= lo) & (d["layer"] < lo + 4)
                   & ~np.char.startswith(d["kind"], "x_")) for d in (t, p)]
        print(f"  prefill weights, layers {lo:2d}-{lo + 3:2d}: silent {pair(r[0][1], r[1][1])}   >=100 deg {pair(r[0][4], r[1][4])}")
    for k in ("v_proj", "k_proj"):
        cnt = []
        for d in (t, p):
            m = (d["site"] == "w-prefill") & (d["kind"] == k)
            cnt.append(f"{int(((d['dmax'][m] >= 100) & np.isin(d['outcome'][m], ('silent', 'erratic'))).sum())} of {int(m.sum())}")
        print(f"  prefill {k} blocks giving >=100 deg: TensorRT {cnt[0]}, PyTorch {cnt[1]}")
    for d, name in ((t, "TensorRT"), (p, "PyTorch")):
        m = d["site"] == "w-expert-all"
        er = d["dmax"][m & (d["outcome"] == "erratic")]
        print(f"  stored expert weights, {name}: erratic {100 * len(er) / m.sum():.1f}%, {100 * np.mean(er >= 100):.0f}% of them >=100 deg")
    print("  stored expert weights by type, erratic / detected (%):")
    for kinds in (("o_proj",), ("down_proj",), ("gate_proj+up_proj", "gate_proj", "up_proj"), ("q_proj",)):
        cells = []
        for d in (t, p):
            m = (d["site"] == "w-expert-all") & np.isin(d["kind"], kinds)
            cells.append(f"{100 * np.mean(d['outcome'][m] == 'erratic'):5.1f} / {100 * np.mean(d['outcome'][m] == 'nan'):5.1f} (n {m.sum()})")
        print(f"    {kinds[-1] if len(kinds) > 1 else kinds[0]:10s} TensorRT {cells[0]}   PyTorch {cells[1]}")
    for d, name in ((t, "TensorRT"), (p, "PyTorch")):
        m = (d["site"] == "w-expert") & (d["idx"] == 9) & np.isin(d["outcome"], ("silent", "erratic", "small"))
        print(f"  one-step weights, last step, changed plans, {name}: median {np.median(d['dmax'][m]):.0f} deg, "
              f"p75 {np.percentile(d['dmax'][m], 75):.0f} deg (n {m.sum()})")


if __name__ == "__main__":
    main()
