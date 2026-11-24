#!/usr/bin/env python3
"""Main/arm side of the load campaigns (Section 6.6), forked from fi_v3.py.

Runs smolvla_212_v3_rollout_async.py unchanged, with its TensorRT worker wrapped by
arm_load_worker.py (which pulses the decode load and classifies the faulted model). The one difference
from fi_v3.py: the arm is gated on the worker's load verdict. Every Goal_Position write is DROPPED until
FI_LOAD_VERDICT reports outcome "erratic"; from then on the raw policy actions reach the motors (the
rollout is run with --max-tick-deg 0 --force, so no slew limit and no first-action check). On an erratic
launch the arm is ramped back to REST on disconnect; on a non-erratic launch nothing ever moves.

usage (spawned by arm_load_run.py on the Jetson):
  arm_load_fi.py OUT_PREFIX smolvla_212_v3_rollout_async.py --max-tick-deg 0 --force [args...]
"""
import atexit
import json
import os
import runpy
import subprocess
import sys
import time
from multiprocessing.connection import Connection
from pathlib import Path

import numpy as np
from lerobot.motors.motors_bus import SerialMotorsBus
from lerobot.robots.so_follower import SO101Follower

HERE = Path(__file__).resolve().parent
WORKER_WRAPPER = os.environ.get("FI_LOAD_WRAPPER", str(HERE / "arm_load_worker.py"))
VERDICT_JSON = Path(os.environ.get("FI_LOAD_VERDICT", HERE / "arm_verdict.json"))
KEYS = ("shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos", "wrist_flex.pos", "wrist_roll.pos", "gripper.pos")
REST = np.array([-3.2, -103.7, 96.7, 73.8, 1.4, 22.8], np.float32)   # measured at rest, 2026-11-01
FPS = 30

out, script = sys.argv[1], sys.argv[2]
sys.argv = sys.argv[2:]
t0 = time.perf_counter()
log = {"cmd_t": [], "cmd": [], "obs_t": [], "obs": [], "req_t": [], "req_step": [], "req_state": [],
       "res_t": [], "res_step": [], "res_latency": [], "res_chunk": [], "err_t": [], "err": []}
dropped = {"Goal_Position": 0}
_moving = {"on": False}

orig_write, orig_sync_write = SerialMotorsBus.write, SerialMotorsBus.sync_write
orig_send_action, orig_get_obs = SO101Follower.send_action, SO101Follower.get_observation
orig_disconnect = SO101Follower.disconnect
orig_send, orig_recv = Connection.send, Connection.recv
orig_popen_init = subprocess.Popen.__init__


def moving():
    """True once the worker's load verdict is 'erratic' (latched). Until then the arm is held."""
    if _moving["on"]:
        return True
    try:
        v = json.loads(VERDICT_JSON.read_text())
    except (OSError, ValueError):
        return False
    if v.get("outcome") == "erratic":
        _moving["on"] = True
        print(f"\n[arm_load_fi] erratic load verdict (max_dev {v.get('max_dev_deg')} deg) "
              f"-> ARM ENABLED, executing the faulted policy", flush=True)
    return _moving["on"]


def write(self, data_name, motor, value, *args, **kwargs):
    if data_name == "Goal_Position" and not moving():
        dropped["Goal_Position"] += 1
        return None
    return orig_write(self, data_name, motor, value, *args, **kwargs)


def sync_write(self, data_name, values, *args, **kwargs):
    if data_name == "Goal_Position" and not moving():
        dropped["Goal_Position"] += 1
        return None
    return orig_sync_write(self, data_name, values, *args, **kwargs)


def ramp(robot, start, target, seconds):
    steps = max(1, int(seconds * FPS))
    for i in range(1, steps + 1):
        orig_send_action(robot, {k: float(v) for k, v in zip(KEYS, start + (target - start) * (i / steps))})
        time.sleep(1.0 / FPS)


def go_rest(robot):
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
    if _moving["on"] and self.is_connected:                 # only an erratic launch ever moved the arm
        try:
            print(f"\n[arm_load_fi] arm back at rest: {np.round(go_rest(self), 1).tolist()}", flush=True)
        except Exception as exc:
            print(f"\n[arm_load_fi] could not return the arm to rest: {exc}", flush=True)
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
    print(f"\n[arm_load_fi] {'ARM MOVED' if _moving['on'] else 'held (verdict not erratic); dropped %d' % dropped['Goal_Position']}; "
          f"{len(log['cmd'])} commands, {len(log['req_t'])} requests, {len(log['res_t'])} chunks -> {out}.npz", flush=True)


SerialMotorsBus.write, SerialMotorsBus.sync_write = write, sync_write
SO101Follower.send_action, SO101Follower.get_observation = send_action, get_observation
SO101Follower.disconnect = disconnect
Connection.send, Connection.recv = send, recv
subprocess.Popen.__init__ = popen_init
atexit.register(save)

runpy.run_path(script, run_name="__main__")
