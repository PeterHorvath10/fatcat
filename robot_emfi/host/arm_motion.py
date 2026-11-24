#!/usr/bin/env python3
"""What the arm executed in a faulted episode (arm_load_run.py / fi_v3_run.py --move motion log).

Reads the episode's motion log (arm_load_fi.py / fi_v3.py OUT.npz: commands, measured joint states at the
re-plans, received chunks) and reports its length, how many plans it received, how many of them were NaN
(with the NaN drop, a NaN plan is replaced by the previous one before it is logged; the worker reports the
drops in the rollout log; worker errors are e.g. the NaN plan that stops the stock rollout),
jumped by more than 10 deg between consecutive steps or left the joint limits (+-0.5 deg, the outcome
classes' limits), and how far the measured joints moved from the rest pose. With --golden, also the
largest step-to-step change of the fault-free reference plans (arm_load_run.py's arm_golden.npz).

usage: arm_motion.py MOTION.npz [MOTION.npz ...] [--golden arm_golden.npz]
"""
import argparse

import numpy as np

LIM = np.array([(-120.4, 120.4), (-105, 105), (-97.9, 97.9), (-103.9, 103.9), (-180, 180), (0, 100)])
REST = np.array([-3.2, -103.7, 96.7, 73.8, 1.4, 22.8])      # arm_load_fi.py
JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("motion", nargs="+")
    ap.add_argument("--golden", help="fault-free reference plans (chunks: obs x 50 x 6)")
    a = ap.parse_args()
    for path in a.motion:
        d = np.load(path)
        c = d["res_chunk"].astype(float)[..., :6]
        fin = np.isfinite(c).all(axis=(1, 2))
        cf = c[fin]
        jump = np.abs(np.diff(cf, axis=1)).max(axis=(1, 2)) > 10
        out = ((cf < LIM[:, 0] - 0.5) | (cf > LIM[:, 1] + 0.5)).any(axis=(1, 2))
        t = d["cmd_t"]
        print(f"{path}: {t[-1] - t[0]:.1f} s, {len(d['cmd'])} commands, {len(c)} plans; NaN plans received "
              f"{int((~fin).sum())}, worker errors {len(d['err'])}; finite plans with a step > 10 deg: "
              f"{int(jump.sum())}, outside the joint limits: {int(out.sum())}")
        obs = d["obs"]
        for j, name in enumerate(JOINTS):
            lo, hi = obs[:, j].min(), obs[:, j].max()
            print(f"    {name:13s} measured {lo:7.1f} .. {hi:6.1f} deg, up to {np.abs(obs[:, j] - REST[j]).max():5.1f} "
                  f"from rest")
    if a.golden:
        g = np.load(a.golden)["chunks"]
        print(f"fault-free reference plans: largest step-to-step change {np.abs(np.diff(g, axis=1)).max():.2f} deg")


if __name__ == "__main__":
    main()
