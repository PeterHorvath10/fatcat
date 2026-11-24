#!/usr/bin/env python3
"""Summarize gpumem_run.py campaigns: outcome per shot, and what each corrupted load looked like.

The Jetson's DRAM is AES-XTS encrypted (16-byte blocks) and decrypted on the way into L2. A
fault on the encrypted side (DRAM, its bus, the controller before decryption) turns the whole
16-byte block into random bytes, whatever it was: a flipped ciphertext bit and a read from the
wrong address (wrong tweak) look the same. A fault after decryption (L2, fabric, SM) keeps its
shape: bit flips, or another word of the buffer.

Write mode (gpumem --mode write): the words are those read back wrong after the shot, timed when
they were stored, and a word still holding the previous shot's pattern is "stale" (write lost).

Each corrupted word (first record per word and shot; a word wrong in DRAM repeats every pass):
  stale     write mode: the previous shot's value, the store never reached DRAM
  block16   all 4 words of its aligned 16-byte block wrong, none of the kinds below: a garbled AES block
  compare   loaded == expected: the compare or branch was faulted, not the load
  flip-N    loaded = expected with N <= 3 bits flipped (bit positions and direction counted)
  address   loaded = the hash pattern of another buffer word: the load read the wrong address.
            Only counted when that word is near (< 256 KB) or one address bit away; a random
            value falls into a 256 MB buffer 1 time in 64
  zero/ones loaded 0x00000000 / 0xffffffff
  other     anything else
and the re-read 20 us later: right (transient: only that load), the same wrong value (sticky:
wrong in L2 or DRAM), or another value.

Timing: first_bad_ms (GPU clock, since the kernel start) against pulse_in_kernel_ms (Husky
offset minus the kernel onset measured in the probe RMS): the same pulse seen from both sides.

usage: gpumem_analyze.py RUN.jsonl [RUN2.jsonl ...]
"""

import collections
import json
import sys

import numpy as np

M = 0xFFFFFFFF


def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, c - h), min(1.0, c + h)


def classify(w, want, got, ready, seed, seed_prev):
    if got == want:
        return "compare", None
    if seed_prev != seed and got == (w * ready["k"] ^ seed_prev) & M:
        return "stale", None
    x = got ^ want
    hd = bin(x).count("1")
    if hd <= 3:
        return f"flip-{hd}", None
    if ready["pattern"] == "hash":
        j = ((got ^ seed) * pow(ready["k"], -1, 1 << 32)) & M
        if j < ready["words"] and (abs(j - w) < 65536 or bin(j ^ w).count("1") == 1):
            return "address", j - w
    if got == 0:
        return "zero", None
    if got == M:
        return "ones", None
    return "other", None


def main():
    shots, ready, crashes = [], None, 0
    for path in sys.argv[1:]:
        for line in open(path):
            r = json.loads(line)
            if r.get("ready"):
                ready = ready or r
            elif r.get("crash"):
                crashes += 1
            elif "outcome" in r:
                shots.append(r)
    if not shots:
        sys.exit("no shots")
    pulses = [r for r in shots if r.get("delay_ms") is not None]
    controls = [r for r in shots if r.get("control")]
    fired = [r for r in pulses if r.get("emfi") == "FIRED"]
    v = sorted({(r.get("voltage"), r.get("width_ns")) for r in pulses})
    write = ready.get("mode") == "write"
    print(f"gpumem {ready.get('mode', 'read')}: {ready['mb']:.0f} MB {ready['pattern']}, ld.{ready['load']}, {ready['passes']} passes "
          f"({ready['kernel_ms']:.0f} ms); EMFI {', '.join(f'{a} V {b} ns' for a, b in v) or 'none'}")
    print(f"{len(pulses)} pulses ({len(fired)} confirmed fired), {len(controls)} controls, {crashes} crash/restart rows")

    print("\nOutcomes")
    for name, rows in (("pulses fired", fired), ("controls", controls)):
        c = collections.Counter(r["outcome"] for r in rows)
        bad = sum(n for o, n in c.items() if o != "OK")
        lo, hi = wilson(bad, len(rows))
        print(f"  {name:12s} n={len(rows):4d}  " + ", ".join(f"{o} {n}" for o, n in c.most_common())
              + (f"  | not OK {100 * bad / len(rows):.1f}% [{100 * lo:.1f}-{100 * hi:.1f}]" if rows else ""))

    words, flips_bit, flips_dir, sector_pos, per_shot, lag = [], np.zeros(32, int), collections.Counter(), \
        np.zeros(8, int), [], []
    for r in fired + controls:
        first = {}
        seed = r.get("seed", ready["seed"])
        for t_us, w, want, got, again, pss, sm in r.get("verify_recs" if write else "recs") or []:
            if w not in first or t_us < first[w][0]:
                first[w] = (t_us, want, got, again, pss, sm)
        if not first:
            continue
        in_dram = {rec[1] for rec in r.get("verify_recs") or []}
        per_block = collections.Counter(w // 4 for w in first)
        for w, (t_us, want, got, again, pss, sm) in first.items():
            kind, off = classify(w, want, got, ready, seed, r.get("seed_prev", seed))
            if per_block[w // 4] == 4 and kind == "other":
                kind, off = "block16", None
            reread = "transient" if again == want else "sticky" if again == got else "changed"
            words.append({"kind": kind, "off": off, "reread": reread, "dram": w in in_dram, "control": r.get("control")})
            if kind.startswith("flip"):
                x = got ^ want
                for b in range(32):
                    if x >> b & 1:
                        flips_bit[b] += 1
                        flips_dir["0->1" if got >> b & 1 else "1->0"] += 1
            sector_pos[w % 8] += 1
        ws = sorted(first)
        per_shot.append({"n": len(ws), "sectors": len({w // 8 for w in ws}), "lines": len({w // 32 for w in ws}),
                         "span": ws[-1] - ws[0], "passes": len({rec[5] for rec in r.get("recs") or []}),
                         "sms": len({v[5] for v in first.values()}), "outcome": r["outcome"]})
        if r.get("first_bad_ms") is not None and r.get("pulse_in_kernel_ms") is not None:
            lag.append(r["first_bad_ms"] - r["pulse_in_kernel_ms"])

    print(f"\n{'Words wrong in DRAM after the shot' if write else 'Corrupted words'}: {len(words)} in {len(per_shot)} shots"
          f" ({sum(w['control'] for w in words)} in controls)")
    if words:
        for key in ("kind", "reread", "dram"):
            c = collections.Counter(w[key] for w in words)
            print(f"  {key:7s} " + ", ".join(f"{k} {n}" for k, n in c.most_common()))
        offs = collections.Counter(w["off"] for w in words if w["kind"] == "address")
        if offs:
            print("  address offsets: " + ", ".join(f"{4 * o:+d} B x{n}" for o, n in offs.most_common(12)))
        if flips_bit.any():
            print(f"  flipped bits: {dict(flips_dir)}; by bit 0..31: {flips_bit.tolist()}")
            print(f"    by byte lane: {flips_bit.reshape(4, 8).sum(axis=1).tolist()}")
        print(f"  word position in its 32 B sector (0..7): {sector_pos.tolist()}")
        n = np.array([s["n"] for s in per_shot])
        print(f"  per shot: words {np.median(n):.0f} median (max {n.max()}), sectors "
              f"{np.median([s['sectors'] for s in per_shot]):.0f}, 128 B lines {np.median([s['lines'] for s in per_shot]):.0f},"
              f" passes {max(s['passes'] for s in per_shot)} max, SMs {max(s['sms'] for s in per_shot)} max")
    if lag:
        q = np.percentile(lag, [0, 25, 50, 75, 100])
        print(f"\nGPU {'store' if write else 'fault'} time - pulse position: median {q[2]:+.2f} ms, IQR {q[1]:+.2f}..{q[3]:+.2f}, "
              f"range {q[0]:+.2f}..{q[4]:+.2f} (n={len(lag)})")

    print("\nBy pulse position in the kernel (measured; 10 ms bins)")
    pos = [(r["pulse_in_kernel_ms"], r["outcome"] != "OK") for r in fired if r.get("pulse_in_kernel_ms") is not None]
    for a in range(-10, int(ready["kernel_ms"]) + 10, 10):
        sel = [b for p, b in pos if a <= p < a + 10]
        if sel:
            print(f"  {a:4d}-{a + 10:3d} ms  n={len(sel):3d}  not OK {sum(sel):3d} ({100 * sum(sel) / len(sel):.0f}%)")
    unplaced = sum(1 for r in fired if r.get("pulse_in_kernel_ms") is None)
    if unplaced:
        print(f"  {unplaced} fired pulses without a measured kernel onset")


if __name__ == "__main__":
    main()
