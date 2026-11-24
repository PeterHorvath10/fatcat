#!/usr/bin/env python3
"""Live view of a running gpumem_run.py / arm_load_run.py --live FILE: the Husky probe
capture of the latest pulse (the scan holds the Husky, so probe_monitor.py cannot run beside it), with
the expected kernel/copy onset and the pulse marked, plus the outcome of every shot so far (faulty in red).

usage: live_view.py FILE.npz [RUN.jsonl]
"""
import json
import sys
from collections import deque

import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np


def faulty(outcome, n_bad):
    """gpumem: LOAD/MEM/...; loading/robot runs: changed weight blocks or an erratic/NaN plan."""
    return bool(n_bad) or outcome in ("LOAD", "MEM", "MEM+COUNT", "LOAD+COUNT", "erratic", "nan", "silent",
                                      "NO_REPLY", "ERROR", "TIMEOUT")


def main():
    live = sys.argv[1]
    run = sys.argv[2] if len(sys.argv) > 2 else None
    plt.ion()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7.5), gridspec_kw={"height_ratios": [3, 1.3]})
    fig.canvas.manager.set_window_title("EMFI live: Husky probe + faults")
    state = {"on": True, "mtime": 0.0}
    fig.canvas.mpl_connect("close_event", lambda e: state.update(on=False))
    hist = deque(maxlen=200)
    while state["on"]:
        try:
            mt = __import__("os").path.getmtime(live)
        except OSError:
            plt.pause(0.3)
            continue
        if mt != state["mtime"]:
            state["mtime"] = mt
            try:
                z = np.load(live)
                tr, dt, row = z["trace"].astype(float), float(z["dt_ms"]), json.loads(str(z["row"]))
            except Exception:
                plt.pause(0.1)
                continue
            tr -= np.median(tr)
            t = np.arange(len(tr)) * dt
            ax1.clear()
            ax1.plot(t, tr, lw=0.4, color="#1f77b4")
            if row.get("onset_est_ms") is not None:
                ax1.axvline(row["onset_est_ms"], color="#888", ls="--", lw=1, label=f"kernel start (est) {row['onset_est_ms']:.1f} ms")
            if row.get("offset_ms") is not None:
                ax1.axvline(row["offset_ms"], color="r", lw=1.5, label=f"pulse {row['offset_ms']:.1f} ms")
            bad = faulty(row.get("outcome"), row.get("n_bad"))
            ax1.set_title(f"shot {row.get('n')}  |  {row.get('voltage', '-')} V, {row.get('delay_ms')} ms into kernel  |  "
                          f"{row.get('outcome')}  bad words {row.get('n_bad', '-')}"
                          + (f"  first@{row['first_bad_ms']:.2f} ms" if row.get("first_bad_ms") is not None else "")
                          + (f"  {row['extra']}" if row.get("extra") else ""),
                          color="#c0392b" if bad else "black", fontsize=11)
            ax1.set_xlabel("ms after the Husky trigger")
            ax1.set_ylabel("probe (centered)")
            ax1.grid(alpha=0.3)
            ax1.legend(loc="upper right", fontsize=9)
            if run:
                try:
                    rows = [json.loads(l) for l in open(run)]
                except Exception:
                    rows = []
                hist.clear()
                # gpumem rows count shots in "n" and bad words in "n_bad"; arm_load_run.py rows use
                # "attempt" and "n_bad_blocks"
                hist.extend((r.get("n", r.get("attempt")), r.get("outcome"), r.get("n_bad", r.get("n_bad_blocks")) or 0)
                            for r in rows if r.get("delay_ms") is not None and "outcome" in r)
            elif not hist or hist[-1][0] != row.get("n"):
                hist.append((row.get("n"), row.get("outcome"), row.get("n_bad") or 0))
            ax2.clear()
            if hist:
                n, o, b = zip(*hist)
                fl = [faulty(x, v) for x, v in zip(o, b)]
                col = ["#c0392b" if f else "#2ca02c" for f in fl]
                ax2.bar(n, [max(v, 0.5) for v in b], color=col, width=0.9)
                nf = sum(fl)
                ax2.set_title(f"last {len(o)} pulses: {nf} faulty ({100 * nf / len(o):.0f}%)  "
                              f"|  last 20: {sum(fl[-20:])}/{len(fl[-20:])}", fontsize=10)
            ax2.set_xlabel("shot")
            ax2.set_ylabel("bad words")
            ax2.grid(alpha=0.3)
            fig.tight_layout()
        plt.pause(0.2)


if __name__ == "__main__":
    main()
