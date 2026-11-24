#!/usr/bin/env python3
"""Run the v3 rollout for EMFI: by default the arm never moves, every would-be command is logged.

usage: [FI_V3_MOVE=1] fi_v3.py OUT_PREFIX SCRIPT [script args...]

Parent process (this one):
  - every Goal_Position write is dropped at the bus, so robot.send_action
    runs its full path but the motors only hold their position under torque;
    FI_V3_MOVE=1 lets them through (the arm moves for real) and, when the rollout disconnects
    (finished or aborted, e.g. on NaN), first ramps the arm back to REST,
    so releasing torque does not drop it;
  - every motor command, state read, inference request and received chunk is
    logged and saved to OUT_PREFIX.npz on exit (also after an abort);
  - the TensorRT worker is started through fi_worker.py (fault-free recompute + canary
    after each reply, one "FIV3 {json}" line per inference on stdout).
"""

import atexit
import os
import runpy
import subprocess
import sys
import time
from multiprocessing.connection import Connection

import numpy as np
from lerobot.motors.motors_bus import SerialMotorsBus
from lerobot.robots.so_follower import SO101Follower

WORKER_WRAPPER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fi_worker.py")
MOVE = os.environ.get("FI_V3_MOVE") == "1"
KEYS = ("shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos", "wrist_flex.pos", "wrist_roll.pos", "gripper.pos")
REST = np.array([-3.2, -103.7, 96.7, 73.8, 1.4, 22.8], np.float32)   # measured at rest, 2026-11-01
FPS = 30

out, script = sys.argv[1], sys.argv[2]
sys.argv = sys.argv[2:]
t0 = time.perf_counter()
log = {"cmd_t": [], "cmd": [], "obs_t": [], "obs": [], "req_t": [], "req_step": [], "req_state": [],
       "res_t": [], "res_step": [], "res_latency": [], "res_chunk": [], "err_t": [], "err": []}
dropped = {"Goal_Position": 0}

orig_write, orig_sync_write = SerialMotorsBus.write, SerialMotorsBus.sync_write
orig_send_action, orig_get_obs = SO101Follower.send_action, SO101Follower.get_observation
orig_disconnect = SO101Follower.disconnect
orig_send, orig_recv = Connection.send, Connection.recv
orig_popen_init = subprocess.Popen.__init__


def write(self, data_name, motor, value, *args, **kwargs):
    if data_name == "Goal_Position" and not MOVE:
        dropped["Goal_Position"] += 1
        return None
    return orig_write(self, data_name, motor, value, *args, **kwargs)


def sync_write(self, data_name, values, *args, **kwargs):
    if data_name == "Goal_Position" and not MOVE:
        dropped["Goal_Position"] += 1
        return None
    return orig_sync_write(self, data_name, values, *args, **kwargs)


def ramp(robot, start, target, seconds):
    steps = max(1, int(seconds * FPS))
    for i in range(1, steps + 1):
        orig_send_action(robot, {k: float(v) for k, v in zip(KEYS, start + (target - start) * (i / steps))})
        time.sleep(1.0 / FPS)


def go_rest(robot):
    """Shoulder_lift back by up to 40 deg (gripper off the table) with the gripper
    at its rest opening, then every joint to REST."""
    obs = orig_get_obs(robot)
    start = np.array([obs[k] for k in KEYS], np.float32)
    raised = start.copy()
    raised[1] = max(REST[1], start[1] - 40.0)
    raised[5] = REST[5]
    ramp(robot, start, raised, 1.5)
    ramp(robot, raised, REST, 3.0)
    time.sleep(0.5)
    obs = orig_get_obs(robot)
    return np.array([obs[k] for k in KEYS], np.float32)


def disconnect(self, *args, **kwargs):
    if MOVE and self.is_connected:
        try:
            print(f"\n[fi_v3] arm back at rest: {np.round(go_rest(self), 1).tolist()}", flush=True)
        except Exception as exc:                  # bus failure: torque goes off where the arm is
            print(f"\n[fi_v3] could not return the arm to rest: {exc}", flush=True)
    return orig_disconnect(self, *args, **kwargs)


def send_action(self, action):
    log["cmd_t"].append(time.perf_counter() - t0)
    log["cmd"].append([float(v) for v in action.values()])
    return orig_send_action(self, action)


def get_observation(self):
    obs = orig_get_obs(self)
    log["obs_t"].append(time.perf_counter() - t0)
    log["obs"].append([float(v) for k, v in obs.items() if k.endswith(".pos")])
    return obs


def send(self, msg):
    if isinstance(msg, dict) and msg.get("type") == "infer":
        log["req_t"].append(time.perf_counter() - t0)
        log["req_step"].append(int(msg["source_step"]))
        log["req_state"].append(np.asarray(msg["state"], np.float32))
    return orig_send(self, msg)


def recv(self):
    msg = orig_recv(self)
    if isinstance(msg, dict) and msg.get("type") == "result":
        log["res_t"].append(time.perf_counter() - t0)
        log["res_step"].append(int(msg.get("source_step", -1)))
        log["res_latency"].append(float(msg.get("latency", np.nan)))
        log["res_chunk"].append(np.asarray(msg["chunk"], np.float32))
    elif isinstance(msg, dict) and msg.get("type") == "error":
        log["err_t"].append(time.perf_counter() - t0)
        log["err"].append(str(msg.get("detail", ""))[-2000:])
    return msg


def popen_init(self, args, *a, **kw):
    # [AOT_PYTHON, "-u", SCRIPT, "--worker", ...] -> [AOT_PYTHON, "-u", WRAPPER, SCRIPT, "--worker", ...]
    if isinstance(args, (list, tuple)) and "--worker" in args and len(args) > 2 and args[1] == "-u":
        args = [args[0], "-u", WORKER_WRAPPER, *args[2:]]
    return orig_popen_init(self, args, *a, **kw)


def save():
    np.savez_compressed(out + ".npz", **{k: np.asarray(v) for k, v in log.items()})
    print(f"\n[fi_v3] {'ARM MOVED' if MOVE else 'Goal_Position writes dropped: %d' % dropped['Goal_Position']}; "
          f"{len(log['cmd'])} {'' if MOVE else 'would-be '}commands, {len(log['req_t'])} requests, "
          f"{len(log['res_t'])} chunks -> {out}.npz", flush=True)


SerialMotorsBus.write, SerialMotorsBus.sync_write = write, sync_write
SO101Follower.send_action, SO101Follower.get_observation = send_action, get_observation
SO101Follower.disconnect = disconnect
Connection.send, Connection.recv = send, recv
subprocess.Popen.__init__ = popen_init
atexit.register(save)

runpy.run_path(script, run_name="__main__")
