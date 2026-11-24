#!/usr/bin/env python3
"""Summary of swfi2.py: outcome per fault site in the EM campaigns' classes, which weights and
layouts matter, and the fingerprint each source leaves in the KV cache (what a dump of the
faulted inference's cache would show). Table 6's PT columns are the first nine rows of the first
table (>=100 there as a count of the 2,000 faults, in the paper in %).
usage (from host/): swfi2_analyze.py out/swfi2/swfi2.npz
"""
import argparse
import collections

import numpy as np

CLASSES = ("unchanged", "small", "silent", "erratic", "nan")
NAMES = {"w-vision": "weight, vision (1 camera)", "w-prefill": "weight, prefill", "w-expert": "weight, expert (1 step)",
         "w-expert-all": "weight, expert (all steps)", "a-vision": "activation, vision", "a-prefill": "activation, prefill",
         "a-expert": "activation, expert (1 step)", "kv-write": "KV cache, written", "kv-read": "KV cache, one step's read",
         "kv-burst": "KV burst (not in paper)", "kv-colbit": "KV column bit (not in paper)"}


def rates(d, m, label, width=32):
    n = int(m.sum())
    if not n:
        return
    out = d["outcome"][m]
    dm = d["dmax"][m]
    sil = dm[out == "silent"]
    big = int(((dm >= 100) & np.isin(out, ("silent", "erratic"))).sum())
    ge1 = 100 * np.mean(np.nan_to_num(dm) >= 1)
    cells = " ".join(f"{100 * np.mean(out == c):6.1f}" for c in CLASSES)
    med = f"{np.median(sil):5.0f} {sil.max():5.0f}" if len(sil) else "    -     -"
    print(f"{label:<{width}s} {n:5d} {cells} {ge1:6.1f} {big:5d}  {med}")


def header(width=32, first="site"):
    print(f"{first:<{width}s} {'n':>5s} " + " ".join(f"{c[:6]:>6s}" for c in CLASSES)
          + f" {'>=1deg':>6s} {'>=100':>5s}  {'silent med/max':>11s}")


def fingerprint(tok, feat, nonfin):
    """Shape of the change in the first KV layer that differs, and whether later layers differ. Odd
    (cross) layers store expert-projected K/V, so a VLM K/V-projection fault there changes every
    feature, but still only K or only V."""
    if nonfin:
        return "non-finite", False
    ch = np.flatnonzero(tok)
    if not len(ch):
        return "none", False
    first = ch[0] // 2
    sel = ch[ch // 2 == first]
    t, f = tok[sel].max(), feat[sel].max()
    if t == 1:
        shape = "block" if f <= 8 and len(sel) == 1 else "one token"
    elif first == 0 and t <= 64:
        shape = "one camera"
    else:
        shape = "K or V only" if len(sel) == 1 else "K and V"
    return shape, bool((ch // 2 > first).any())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    a = ap.parse_args()
    d = dict(np.load(a.npz))
    sites = list(dict.fromkeys(d["site"]))
    print(f"{len(d['site'])} faults, {len(d['clean'])} observations; rates in % of faults; "
          f">=100: silent or erratic chunks changed by >= 100 deg\n")
    print("== outcome per site (classes as in the EM campaigns; nan = detected)")
    header()
    for s in sites:
        rates(d, d["site"] == s, NAMES.get(s, s))

    print("\n== source signatures that EM rates can be compared with (changed = not unchanged)")
    print(f"{'site':<32s} {'NaN/changed':>11s} {'silent/(silent+small)':>21s} {'>=100/silent':>12s} "
          f"{'silent med':>10s} {'erratic p25/50/75':>17s}")
    for s in sites:
        m = d["site"] == s
        out, dm = d["outcome"][m], d["dmax"][m]
        sil, sm, er = dm[out == "silent"], (out == "small").sum(), dm[out == "erratic"]
        big = int(((dm >= 100) & np.isin(out, ("silent", "erratic"))).sum())
        q = "/".join(f"{np.percentile(er, p):.0f}" for p in (25, 50, 75)) if len(er) else "-"
        print(f"{NAMES.get(s, s):<32s} {np.mean(out[out != 'unchanged'] == 'nan'):11.2f} "
              f"{len(sil) / max(len(sil) + sm, 1):21.2f} {big / max(len(sil), 1):12.2f} "
              f"{np.median(sil) if len(sil) else float('nan'):10.0f} {q:>17s}")

    print("\n== weight faults by weight type (all stages)")
    header(40, "stage / type")
    for s in ("w-vision", "w-prefill", "w-expert", "w-expert-all"):
        m = d["site"] == s
        kinds = collections.Counter(d["kind"][m])
        for k, _ in kinds.most_common():
            rates(d, m & (d["kind"] == k), f"{s[2:]} / {k}", 40)

    print("\n== weight faults by block layout and by whether the random block holds a NaN or inf")
    header(40, "stage / layout")
    for s in ("w-vision", "w-prefill", "w-expert", "w-expert-all"):
        m = d["site"] == s
        for lay in ("row", "col", "flat"):
            rates(d, m & (d["layout"] == lay), f"{s[2:]} / {lay}", 40)
        rates(d, m & (d["new_nonfinite"] > 0), f"{s[2:]} / block has NaN/inf", 40)
        rates(d, m & (d["new_nonfinite"] == 0), f"{s[2:]} / block all finite", 40)

    print("\n== prefill weight faults by layer")
    header(40, "layer")
    m = d["site"] == "w-prefill"
    for lo, hi in ((0, 4), (4, 8), (8, 12), (12, 16)):
        rates(d, m & (d["layer"] >= lo) & (d["layer"] < hi) & ~np.char.startswith(d["kind"], "x_"), f"VLM layers {lo}-{hi - 1}", 40)
    rates(d, m & np.char.startswith(d["kind"], "x_"), "expert cross K/V projection", 40)

    print("\n== activation faults by GEMM")
    header(40, "stage / GEMM input of")
    for s in ("a-vision", "a-prefill", "a-expert"):
        m = d["site"] == s
        for k, _ in collections.Counter(d["kind"][m]).most_common():
            rates(d, m & (d["kind"] == k), f"{s[2:]} / {k}", 40)

    print("\n== expert faults by denoising step")
    header(40, "site / step")
    for s in ("w-expert", "a-expert", "kv-read"):
        m = d["site"] == s
        for lo, hi in ((0, 5), (5, 8), (8, 9), (9, 10)):
            rates(d, m & (d["idx"] >= lo) & (d["idx"] < hi), f"{s} / steps {lo}-{hi - 1}", 40)

    print("\n== KV-cache fingerprint of vision and prefill faults (first KV layer that changes;"
          " '+later': later layers change too; a kv-write fault is one block by construction)")
    shapes = ("none", "non-finite", "block", "one token", "one camera", "K or V only", "K and V")
    print(f"{'site / type':<34s} {'n':>5s} " + " ".join(f"{s[:11]:>11s}" for s in shapes) + f" {'+later':>7s}")
    groups = []
    for s in ("w-vision", "a-vision", "w-prefill", "a-prefill"):
        m = d["site"] == s
        groups.append((s, m))
        if s == "w-prefill":
            kk = np.isin(d["kind"], ("k_proj", "v_proj"))
            xx = np.char.startswith(d["kind"], "x_")
            groups += [(" K/V projection (VLM)", m & kk), (" K/V projection (expert)", m & xx),
                       (" other weights", m & ~kk & ~xx)]
        if s == "a-prefill":
            kk = np.isin(d["kind"], ("k_proj", "v_proj"))
            groups += [(" input of a K/V projection", m & kk), (" other GEMM inputs", m & ~kk)]
    for label, m in groups:
        fp = [fingerprint(t, f, n) for t, f, n in zip(d["kv_tok"][m], d["kv_feat"][m], d["kv_nonfinite"][m])]
        cnt = collections.Counter(s for s, _ in fp)
        later = 100 * np.mean([l for s, l in fp if s not in ("none", "non-finite")] or [0])
        print(f"{label:<34s} {len(fp):5d} " + " ".join(f"{100 * cnt[s] / len(fp):11.1f}" for s in shapes) + f" {later:7.1f}")


if __name__ == "__main__":
    main()
