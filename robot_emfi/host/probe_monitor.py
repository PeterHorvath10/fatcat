#!/usr/bin/env python3
"""Live Husky near-field probe monitor: position the probe and pick a trigger level.

Free-runs repeated captures from the Husky (software trigger, glitch off) and plots them live while
probe_load.py reloads the decode engine on the Jetson. Move the probe over the chip and watch the
peak-to-peak meter climb to find the spot where the ~37 ms weight-copy burst couples best; read the
suggested ADC trigger level off the trace. No HV, no glitch.

usage: probe_monitor.py [--gain-db 10] [--decimate 150] [--window-ms 400]
Close the window (or Ctrl-C) to stop and release the scope.
"""
import argparse
from collections import deque

import matplotlib
matplotlib.use("TkAgg")
import chipwhisperer as cw
import matplotlib.pyplot as plt
import numpy as np

from husky import CLK, setup


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gain-db", type=float, default=10)
    ap.add_argument("--decimate", type=int, default=150)
    ap.add_argument("--window-ms", type=float, default=400)
    a = ap.parse_args()

    scope = cw.scope()
    setup(scope, a.gain_db)
    scope.trigger.module = "basic"
    scope.trigger.triggers = "tio4"
    scope.adc.basic_mode = "high"
    scope.adc.presamples = 0
    scope.adc.decimate = a.decimate
    scope.adc.samples = int(a.window_ms * 1e-3 * CLK / a.decimate)
    scope.adc.timeout = 1.0
    dt_ms = a.decimate / CLK * 1e3
    t_ax = np.arange(scope.adc.samples) * dt_ms

    plt.ion()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7.5))
    fig.canvas.manager.set_window_title("Husky near-field probe monitor")
    (ln,) = ax1.plot(t_ax, np.zeros_like(t_ax), lw=0.5, color="#1f77b4")
    lvl_hi = ax1.axhline(0, color="r", ls="--", lw=1)
    lvl_lo = ax1.axhline(0, color="r", ls="--", lw=1)
    ax1.set_xlabel("ms after capture start")
    ax1.set_ylabel("probe (centered)")
    ax1.grid(alpha=0.3)
    hist = deque(maxlen=240)
    (hln,) = ax2.plot([], [], lw=1.2, color="#2ca02c")
    best_ln = ax2.axhline(0, color="#888", ls=":", lw=1)
    ax2.set_xlabel("capture #")
    ax2.set_ylabel("burst peak-to-peak")
    ax2.grid(alpha=0.3)

    state = {"on": True, "best": 0.0}
    fig.canvas.mpl_connect("close_event", lambda e: state.update(on=False))

    n = 0
    try:
        while state["on"]:
            scope.io.tio4 = "gpio_high"
            scope.arm()
            to = scope.capture()
            scope.io.tio4 = "high_z"
            if to:
                continue
            tr = scope.get_last_trace()
            tr = tr - np.median(tr)
            p2p = float(tr.max() - tr.min())
            rms = float(tr.std())
            lvl = round(0.5 * float(np.abs(tr).max()), 3)
            state["best"] = max(state["best"], p2p)
            hist.append(p2p)
            n += 1

            ln.set_ydata(tr)
            m = max(0.05, 1.15 * float(np.abs(tr).max()))
            ax1.set_ylim(-m, m)
            lvl_hi.set_ydata([lvl, lvl])
            lvl_lo.set_ydata([-lvl, -lvl])
            ax1.set_title(f"peak-to-peak {p2p:.3f}   rms {rms:.3f}   suggested trigger level  ±{lvl:.3f}",
                          fontsize=11)
            hln.set_data(range(len(hist)), list(hist))
            ax2.set_xlim(0, max(20, len(hist)))
            ax2.set_ylim(0, max(0.05, 1.2 * max(hist)))
            best_ln.set_ydata([state["best"], state["best"]])
            ax2.set_title(f"move the probe to maximize   |   best so far {state['best']:.3f}", fontsize=11)
            fig.canvas.draw_idle()
            fig.canvas.flush_events()
            if n % 10 == 0:
                print(f"  {n} captures, p2p {p2p:.3f}, best {state['best']:.3f}, suggested level ±{lvl:.3f}",
                      flush=True)
            plt.pause(0.01)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            scope.io.tio4 = "high_z"
            scope.dis()
        except Exception:
            pass
        print("scope released", flush=True)


if __name__ == "__main__":
    main()
