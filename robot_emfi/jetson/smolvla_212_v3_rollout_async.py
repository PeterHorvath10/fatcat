#!/usr/bin/env python3
"""SmolVLA 212_v3 SO-ARM101 rollout.

Control logic is intentionally based on an earlier known-good rollout script:

- 30 Hz control loop
- 50-action chunks
- FIRST chunk is inferred synchronously before the control clock starts
- request next inference when <= 30 actions remain
- align chunks by absolute timestep
- overlapping future actions: 30% old + 70% new
- discard already-executed actions and unrelated old tail
- queue underrun => hold measured robot position
- policy action goes to the robot through a per-tick slew limit (--max-tick-deg,
  the same 12 deg default as V4; 0 sends it unclipped as before)
- the first action must lie within --max-first-delta of the measured state
- the follower is opened by its USB serial number, and its motor calibration is
  checked against follower.json before torque is enabled

Only the pieces required for 212_v3 are changed:

- validated AOT TensorRT backend of the runtime, in its own .venv worker process
- 212_v3 FP16 bundle/cache
- unchanged task text
- 212_v3 physical camera paths (camera names already match training)

212_v3 was recorded/fine-tuned directly as:
    observation.images.camera1 = TOP  (USB 2.3)
    observation.images.camera2 = SIDE (USB 2.4)

Backend image order remains [TOP, SIDE]; no camera-name remapping is needed.
"""

from __future__ import annotations

import argparse
import os
from multiprocessing.connection import Connection
from pathlib import Path
import socket
import subprocess
import sys
import threading
import time
import traceback

import numpy as np


# ============================================================
# SETTINGS -- 212_v3
# ============================================================

# Paths and device names come from site_config.py next to this script.
from site_config import AOT_CACHE, AOT_PYTHON, BUNDLE, CAMERA_SIDE, CAMERA_TOP  # noqa: E402
from site_config import FOLLOWER_PORT as PORT  # noqa: E402
from site_config import VLA_REPO as REPO  # noqa: E402

# V4 model (212_v3_bs64_s30000_FP16 missed the grasp in every trial)

# Follower controller board by USB serial number: /dev/ttyACM* numbers follow
# enumeration order and can swap with the leader after a reboot or replug.

TASK = "pick up the red cube and place it in the transparent box"


# Fine-tune/export labels (already correct in v3):
#   TOP  = camera1
#   SIDE = camera2
# Backend image order stays [TOP, SIDE].

FPS = 30.0
CHUNK_SIZE = 50
REPLAN_REMAINING = 30
OLD_WEIGHT = 0.30
NEW_WEIGHT = 0.70

JOINTS = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)
ACTION_KEYS = tuple(f"{name}.pos" for name in JOINTS)
WRIST_FLEX_INDEX = JOINTS.index("wrist_flex")

# camera2 (SIDE) ships with continuous autofocus; camera1 already runs manual.
SIDE_FOCUS_ABSOLUTE = 200


# ============================================================
# AOT WORKER (TensorRT runtime)
# ============================================================

def worker_main(fd: int, seed: int):
    """Run only inside the runtime's .venv. Never touches robot or cameras."""

    conn = Connection(fd)

    try:
        sys.path.insert(0, str(REPO))
        sys.path.insert(0, str(REPO / "bench/aot/smolvla"))

        from candidate_cache import verify_candidate
        from runtime_memory import install_lean_tokenizer

        install_lean_tokenizer()

        from bench.vendor import groot_trt
        from bench.backends.trt_split_smolvla import TrtSplitSmolVLABackend
        from bench.obs import Observation

        manifest = verify_candidate(AOT_CACHE, BUNDLE)

        groot_trt.ACCUMULATE = (
            manifest.get("expert_accumulate")
            or manifest.get("accumulate")
            or "auto"
        )

        # The AOT cache is already built and verified. Never rebuild it here.
        def verified_prebuild(bundle, cache_dir, *args, **kwargs):
            if Path(cache_dir).resolve() != AOT_CACHE.resolve():
                raise RuntimeError(
                    "Refusing build in unexpected AOT cache directory"
                )
            if bundle.root.resolve() != BUNDLE.resolve():
                raise RuntimeError(
                    "Selected AOT cache does not belong to selected bundle"
                )
            return {}

        groot_trt.prebuild_engines = verified_prebuild

        backend = TrtSplitSmolVLABackend(
            BUNDLE,
            str(AOT_CACHE),
            action_dim=6,
            chain="graph",
        )
        backend.load()

        if backend.fixture_parity.get("status") != "PASS":
            raise RuntimeError(
                f"AOT fixture parity failed: {backend.fixture_parity}"
            )

        info = backend.bundle.info

        contract = (
            int(info["n_cam_slots"]),
            int(info["action_dim"]),
            int(info["chunk_size"]),
            int(info["n_action_steps"]),
        )

        if contract != (2, 6, 50, 50):
            raise RuntimeError(
                "Unexpected model contract. "
                f"Expected (2 cams, 6 actions, 50 chunk, 50 steps), got {contract}"
            )

        rng = np.random.default_rng(seed)

        conn.send(
            {
                "type": "ready",
                "parity": "PASS",
                "contract": contract,
            }
        )

        while True:
            msg = conn.recv()

            if msg.get("type") == "stop":
                break

            if msg.get("type") != "infer":
                raise RuntimeError(f"Unknown worker message: {msg}")

            source_step = int(msg["source_step"])
            state = np.asarray(msg["state"], dtype=np.float32)
            images = msg["images"]

            if state.shape != (6,):
                raise RuntimeError(f"Bad state shape: {state.shape}")

            if len(images) != 2:
                raise RuntimeError(
                    f"Expected 2 camera images, got {len(images)}"
                )

            # SmolVLA flow-matching noise: [1, 50, max_action_dim=32]
            noise = rng.standard_normal((1, CHUNK_SIZE, 32)).astype(
                np.float32
            )

            # IMPORTANT: images already arrive in MODEL order:
            #   slot 1 / camera1 = TOP
            #   slot 2 / camera2 = SIDE
            obs = Observation(
                index=source_step,
                images=images,
                state=state,
                task=TASK,
                noise=noise,
            )

            t0 = time.perf_counter()
            result = backend.infer(obs)
            latency = time.perf_counter() - t0

            chunk = np.asarray(result.chunk, dtype=np.float32)

            if chunk.shape != (CHUNK_SIZE, 6):
                raise RuntimeError(
                    f"Wrong action shape: {chunk.shape}; expected (50, 6)"
                )

            if not np.isfinite(chunk).all():
                raise RuntimeError("TensorRT returned NaN/Inf")

            conn.send(
                {
                    "type": "result",
                    "source_step": source_step,
                    "chunk": chunk,
                    "latency": latency,
                }
            )

    except (EOFError, BrokenPipeError):
        pass

    except Exception:
        detail = traceback.format_exc()
        print(detail, file=sys.stderr, flush=True)

        try:
            conn.send({"type": "error", "detail": detail})
        except Exception:
            pass

    finally:
        conn.close()


# ============================================================
# CAMERA -- keep latest MJPG frame available while TRT runs
# ============================================================

class LatestCamera:
    def __init__(self, name: str, dev: str):
        import cv2

        self.cv2 = cv2
        self.name = name
        self.dev = dev

        self.cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)

        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open {name}: {dev}")

        self.cap.set(
            cv2.CAP_PROP_FOURCC,
            cv2.VideoWriter_fourcc(*"MJPG"),
        )
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.lock = threading.Lock()
        self.frame = None
        self.stamp = 0.0
        self.stop_event = threading.Event()

        self.thread = threading.Thread(
            target=self._loop,
            daemon=True,
        )
        self.thread.start()

    def _loop(self):
        while not self.stop_event.is_set():
            ok, frame = self.cap.read()

            if ok and frame is not None and frame.shape == (480, 640, 3):
                with self.lock:
                    self.frame = frame
                    self.stamp = time.monotonic()
            else:
                self.stop_event.wait(0.05)

    def rgb(self, max_age: float = 0.6):
        with self.lock:
            frame = None if self.frame is None else self.frame.copy()
            stamp = self.stamp

        age = time.monotonic() - stamp

        if frame is None or age > max_age:
            raise RuntimeError(
                f"{self.name} frame missing/stale ({age:.2f}s)"
            )

        return np.ascontiguousarray(
            self.cv2.cvtColor(frame, self.cv2.COLOR_BGR2RGB)
        )

    def close(self):
        self.stop_event.set()
        self.thread.join(timeout=1.0)
        self.cap.release()


# ============================================================
# ROBOT / OBSERVATION HELPERS
# ============================================================

def read_state(robot):
    raw_obs = robot.get_observation()

    missing = [key for key in ACTION_KEYS if key not in raw_obs]

    if missing:
        raise RuntimeError(f"Missing robot state keys: {missing}")

    state = np.asarray(
        [raw_obs[key] for key in ACTION_KEYS],
        dtype=np.float32,
    )

    if state.shape != (6,) or not np.isfinite(state).all():
        raise RuntimeError(f"Invalid robot state: {state}")

    return state


def verify_follower_arm(robot):
    """Refuse to start unless the arm on PORT holds the follower calibration.

    robot.connect() enables torque, so the motors' homing offsets and position
    limits are read first, read-only. The leader arm carries its own values.
    """

    if not robot.calibration:
        raise RuntimeError(
            f"No follower calibration file: {robot.calibration_fpath}"
        )

    robot.bus.connect()

    try:
        on_motors = robot.bus.read_calibration()
    finally:
        robot.bus.disconnect(disable_torque=False)

    mismatched = [
        name
        for name, expected in robot.calibration.items()
        if (
            on_motors[name].homing_offset,
            on_motors[name].range_min,
            on_motors[name].range_max,
        )
        != (expected.homing_offset, expected.range_min, expected.range_max)
    ]

    if mismatched:
        raise RuntimeError(
            f"The arm on {robot.config.port} does not hold the follower "
            f"calibration in {robot.calibration_fpath} (mismatched motors: "
            f"{', '.join(mismatched)}). Wrong arm or wrong port?"
        )


def slew_limit(action, previous, max_delta: float):
    """Limit the per-tick change of the commanded action (per joint)."""

    if max_delta <= 0:
        return np.asarray(action, dtype=np.float32), False

    delta = np.asarray(action, dtype=np.float32) - previous
    clipped = np.clip(delta, -max_delta, max_delta)

    return (previous + clipped).astype(np.float32), bool(np.any(clipped != delta))


def check_first_action(first_action, state, limit: float, force: bool):
    """Refuse a first action further than `limit` degrees from the measured state."""

    delta = float(np.abs(np.asarray(first_action) - state).max())

    if delta <= limit:
        return

    message = (
        f"first action differs from the measured state by {delta:.0f} deg "
        f"(limit {limit:.0f})"
    )

    if force:
        print(f"WARNING: {message}")
        return

    raise RuntimeError(
        message + ". Refusing to move; check the gripper/scene, or use --force."
    )


def capture_policy_observation(robot, top, side):
    """Capture one fresh logical observation in MODEL camera order.

    camera1 = TOP  = USB 2.3
    camera2 = SIDE = USB 2.4
    """

    state = read_state(robot)

    # MODEL ORDER, not USB numerical order.
    images = [
        top.rgb(),   # camera1
        side.rgb(),  # camera2
    ]

    return state, images


def lock_side_camera_focus():
    """Turn off continuous autofocus on camera2 (SIDE).

    Left on, it hunts: the sharpness of a static scene swings ~2x within
    seconds and the policy reacts to that blur. The setting resets whenever
    the camera is replugged, so it is applied on every run.
    """

    try:
        subprocess.run(
            [
                "v4l2-ctl",
                "-d",
                CAMERA_SIDE,
                "-c",
                "focus_automatic_continuous=0",
                "-c",
                f"focus_absolute={SIDE_FOCUS_ABSOLUTE}",
            ],
            check=True,
            capture_output=True,
            timeout=5.0,
        )
    except Exception as exc:
        print(f"WARNING: could not lock SIDE camera focus: {exc}")


def wake_nudge(robot, degrees, duration=0.5):
    """Start the policy the way every training episode starts.

    All 50 recorded episodes begin with wrist_flex dropping ~12 deg, then
    ~30 deg, before anything else moves. From a perfectly still rest pose the
    policy never sees that cue and keeps predicting "hold", so ramp wrist_flex
    down by `degrees` and let the policy take over from there.
    """

    start = read_state(robot)
    target = start.copy()
    target[WRIST_FLEX_INDEX] -= degrees

    steps = max(1, int(duration * FPS))

    for i in range(1, steps + 1):
        command = start + (target - start) * (i / steps)

        robot.send_action(
            {
                key: float(value)
                for key, value in zip(
                    ACTION_KEYS,
                    command,
                    strict=True,
                )
            }
        )

        time.sleep(1.0 / FPS)

    # Let the follower settle before the next observation.
    time.sleep(0.2)


# ============================================================
# KNOWN-GOOD ACTION QUEUE LOGIC
# ============================================================

def merge_chunk(
    action_buffer,
    new_chunk,
    source_step,
    latest_executed,
):
    """Same semantics as the known-good rollout."""

    old_buffer = dict(action_buffer)
    new_buffer = {}

    stale = 0
    overlap = 0
    added = 0

    for i in range(CHUNK_SIZE):
        absolute_step = int(source_step) + i

        if absolute_step <= latest_executed:
            stale += 1
            continue

        new_action = np.asarray(new_chunk[i], dtype=np.float32)

        if absolute_step in old_buffer:
            old_action = old_buffer[absolute_step]

            new_buffer[absolute_step] = (
                OLD_WEIGHT * old_action
                + NEW_WEIGHT * new_action
            ).astype(np.float32)

            overlap += 1

        else:
            new_buffer[absolute_step] = new_action.copy()
            added += 1

    # Exactly like known-good code: incoming plan replaces unrelated old tail.
    action_buffer.clear()
    action_buffer.update(new_buffer)

    return stale, overlap, added


# ============================================================
# IPC HELPERS
# ============================================================

def check_paths():
    required = [
        ("TensorRT runtime repo", REPO),
        ("TensorRT runtime Python", AOT_PYTHON),
        ("ONNX bundle", BUNDLE),
        ("AOT engine cache", AOT_CACHE),
        ("follower port", Path(PORT)),
        ("SIDE camera / USB 2.4", Path(CAMERA_SIDE)),
        ("TOP camera / USB 2.3", Path(CAMERA_TOP)),
    ]

    for name, path in required:
        if not path.exists():
            raise FileNotFoundError(f"Missing {name}: {path}")

    if not (AOT_CACHE / "candidate.json").is_file():
        raise FileNotFoundError(
            f"Missing verified AOT candidate.json in {AOT_CACHE}"
        )


def start_worker(script_path: Path, seed: int):
    host_sock, child_sock = socket.socketpair()

    env = os.environ.copy()
    env.update(
        {
            "PYTHONNOUSERSITE": "1",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        }
    )

    proc = subprocess.Popen(
        [
            str(AOT_PYTHON),
            "-u",
            str(script_path),
            "--worker",
            "--fd",
            str(child_sock.fileno()),
            "--seed",
            str(seed),
        ],
        pass_fds=(child_sock.fileno(),),
        cwd=str(REPO),
        env=env,
    )

    child_sock.close()
    conn = Connection(host_sock.detach())

    if not conn.poll(180.0):
        proc.terminate()
        raise TimeoutError("AOT worker did not initialize within 180 s")

    ready = conn.recv()

    if ready.get("type") != "ready":
        proc.terminate()
        raise RuntimeError(
            f"AOT worker failed to initialize: {ready}"
        )

    return proc, conn


def send_inference_request(conn, source_step, state, images):
    conn.send(
        {
            "type": "infer",
            "source_step": int(source_step),
            "state": np.asarray(state, dtype=np.float32),
            "images": images,
        }
    )


def receive_result_blocking(conn, proc, timeout: float):
    deadline = time.monotonic() + timeout

    while time.monotonic() < deadline:
        if conn.poll(0.05):
            reply = conn.recv()

            if reply.get("type") == "error":
                raise RuntimeError(
                    "AOT worker error:\n" + reply.get("detail", "unknown")
                )

            if reply.get("type") != "result":
                raise RuntimeError(
                    f"Unexpected worker reply: {reply}"
                )

            return reply

        if proc.poll() is not None:
            raise RuntimeError(
                f"AOT worker exited with code {proc.returncode}"
            )

    raise TimeoutError("Timed out waiting for AOT inference")


def infer_first_chunk(robot, top, side, conn, proc, timeout: float):
    """Capture one observation and infer a 50-action chunk synchronously."""

    state, images = capture_policy_observation(robot, top, side)

    send_inference_request(conn, source_step=0, state=state, images=images)

    reply = receive_result_blocking(conn, proc, timeout)

    chunk = np.asarray(reply["chunk"], dtype=np.float32)

    if chunk.shape != (CHUNK_SIZE, 6):
        raise RuntimeError(f"Wrong first chunk shape: {chunk.shape}")

    if not np.isfinite(chunk).all():
        raise RuntimeError("First chunk contains NaN/Inf")

    return state, chunk, float(reply["latency"])


# ============================================================
# MAIN -- KNOWN-GOOD CONTROL STRUCTURE
# ============================================================

def main(args):
    check_paths()

    total_steps = int(args.duration * FPS)

    print("============================================")
    print(" SmolVLA 212_v3 FP16 TensorRT ASYNC")
    print("============================================")
    print(f"Duration:           {args.duration:.1f} s")
    print(f"Control FPS:        {FPS}")
    print(f"Total steps:        {total_steps}")
    print(f"Chunk size:         {CHUNK_SIZE}")
    print(f"Replan remaining:   {REPLAN_REMAINING}")
    print(f"Old overlap weight: {OLD_WEIGHT}")
    print(f"New overlap weight: {NEW_WEIGHT}")
    print(f"Task:               {TASK}")
    print()
    print("212_v3 CAMERA ORDER (DIRECT LABELS):")
    print(f"  camera1 = TOP  = USB 2.3 = {CAMERA_TOP}")
    print(f"  camera2 = SIDE = USB 2.4 = {CAMERA_SIDE}")
    print("============================================")

    from lerobot.robots.so_follower import (
        SO101Follower,
        SO101FollowerConfig,
    )

    robot = None
    top = None
    side = None
    proc = None
    conn = None
    inference_inflight = False

    try:
        # Per-tick clipping happens in this script (slew_limit), not in lerobot.
        cfg = SO101FollowerConfig(
            port=PORT,
            id="follower",
            cameras={},
            use_degrees=True,
        )

        robot = SO101Follower(cfg)
        verify_follower_arm(robot)
        print("Follower arm verified: motor calibration matches follower.json.")
        robot.connect(calibrate=False)

        print("\nRobot connected.")

        initial_state = read_state(robot)
        print("Initial state:", np.round(initial_state, 2))

        # CAMERA1 is TOP / 2.3; CAMERA2 is SIDE / 2.4.
        lock_side_camera_focus()
        top = LatestCamera("TOP / model camera1", CAMERA_TOP)
        side = LatestCamera("SIDE / model camera2", CAMERA_SIDE)

        deadline = time.monotonic() + 8.0

        while time.monotonic() < deadline:
            try:
                top.rgb()
                side.rgb()
                break
            except RuntimeError:
                time.sleep(0.1)
        else:
            raise RuntimeError("Camera streams did not become ready")

        print("Cameras ready: MJPG 640x480 @ 30 FPS")
        print("Launching validated AOT worker...")

        proc, conn = start_worker(
            Path(__file__).resolve(),
            args.seed,
        )

        print("AOT worker ready; fixture parity PASS.")

        # ====================================================
        # FIRST CHUNK -- SYNCHRONOUS, AS IN KNOWN-GOOD SCRIPT
        # ====================================================

        print("\n===== INITIAL INFERENCE =====")

        initial_state, initial_images = capture_policy_observation(
            robot,
            top,
            side,
        )

        send_inference_request(
            conn,
            source_step=0,
            state=initial_state,
            images=initial_images,
        )

        first_reply = receive_result_blocking(
            conn,
            proc,
            args.inference_timeout,
        )

        first_chunk = np.asarray(
            first_reply["chunk"],
            dtype=np.float32,
        )

        if first_chunk.shape != (CHUNK_SIZE, 6):
            raise RuntimeError(
                f"Wrong first chunk shape: {first_chunk.shape}"
            )

        if not np.isfinite(first_chunk).all():
            raise RuntimeError("First chunk contains NaN/Inf")

        print(
            f"Initial inference: "
            f"{float(first_reply['latency']) * 1000:.1f} ms"
        )
        print("First action:", np.round(first_chunk[0], 2))
        print(
            "Difference from initial state:",
            np.round(first_chunk[0] - initial_state, 2),
        )

        # Exactly like known-good script: first 50 actions are ready BEFORE
        # rollout timing starts.
        action_buffer = {
            i: first_chunk[i].copy()
            for i in range(CHUNK_SIZE)
        }

        latest_executed = -1
        inference_inflight = False
        request_time = 0.0

        inference_latencies = [float(first_reply["latency"])]
        inference_count = 1
        underrun_count = 0
        queue_history = []

        print("\n============================================")
        print(" READY")
        print("============================================")
        print(f"Rollout duration: {args.duration:.1f} seconds")
        print(f"Replan at: <= {REPLAN_REMAINING} remaining actions")
        print("Overlap: 30% old + 70% new")
        print(
            f"Slew limit: {args.max_tick_deg:g} deg/tick"
            + (" (off: direct policy actions)" if args.max_tick_deg <= 0 else "")
        )
        print(f"First-action limit: {args.max_first_delta:g} deg")
        print("camera1=TOP/2.3, camera2=SIDE/2.4")
        print()
        print("Keep your hand near power/emergency stop.")

        input("\nPress ENTER to start...")

        if args.wake_nudge > 0:
            print(
                f"\nWake nudge: wrist_flex -{args.wake_nudge:.1f} deg, "
                "then re-inferring from the nudged pose..."
            )

            wake_nudge(robot, args.wake_nudge)

            initial_state, first_chunk, latency = infer_first_chunk(
                robot,
                top,
                side,
                conn,
                proc,
                args.inference_timeout,
            )

            print("State after nudge:", np.round(initial_state, 2))
            print("First action:", np.round(first_chunk[0], 2))
            print(
                "Difference from state:",
                np.round(first_chunk[0] - initial_state, 2),
            )

            action_buffer = {
                i: first_chunk[i].copy()
                for i in range(CHUNK_SIZE)
            }
            inference_latencies = [latency]

        # The first action must start near where the arm is now, and every
        # command after it moves at most --max-tick-deg from the previous one.
        last_command = read_state(robot)
        check_first_action(
            action_buffer[0],
            last_command,
            args.max_first_delta,
            args.force,
        )
        clipped_ticks = 0

        rollout_start = time.perf_counter()

        # ====================================================
        # 30 Hz CONTROL LOOP -- SAME SHAPE AS KNOWN-GOOD CODE
        # ====================================================

        for step in range(total_steps):
            loop_start = time.perf_counter()

            # ------------------------------------------------
            # RECEIVE COMPLETED ASYNC INFERENCE
            # ------------------------------------------------

            if conn.poll(0):
                reply = conn.recv()
                inference_inflight = False

                if reply.get("type") == "error":
                    raise RuntimeError(
                        "AOT worker error:\n"
                        + reply.get("detail", "unknown")
                    )

                if reply.get("type") != "result":
                    raise RuntimeError(
                        f"Unexpected worker reply: {reply}"
                    )

                source_step = int(reply["source_step"])
                new_chunk = np.asarray(
                    reply["chunk"],
                    dtype=np.float32,
                )
                latency = float(reply["latency"])

                if new_chunk.shape != (CHUNK_SIZE, 6):
                    raise RuntimeError(
                        f"Wrong action shape: {new_chunk.shape}"
                    )

                if not np.isfinite(new_chunk).all():
                    raise RuntimeError("TensorRT returned NaN/Inf")

                inference_count += 1
                inference_latencies.append(latency)

                stale, overlap, added = merge_chunk(
                    action_buffer,
                    new_chunk,
                    source_step,
                    latest_executed,
                )

                print(
                    f"\n[ASYNC RESULT]"
                    f" source={source_step}"
                    f" now={step}"
                    f" infer={latency * 1000:.0f}ms"
                    f" stale={stale}"
                    f" overlap={overlap}"
                    f" added={added}"
                    f" queue={len(action_buffer)}"
                )

            if proc.poll() is not None:
                raise RuntimeError(
                    f"AOT worker exited with code {proc.returncode}"
                )

            if (
                inference_inflight
                and time.monotonic() - request_time
                > args.inference_timeout
            ):
                raise TimeoutError(
                    "AOT inference worker stalled during rollout"
                )

            # ------------------------------------------------
            # ACTION FOR CURRENT TIMESTEP
            # ------------------------------------------------

            action = action_buffer.pop(step, None)

            if action is None:
                underrun_count += 1

                print(f"\n[QUEUE UNDERRUN] step={step}")

                # Same fallback as known-good rollout: hold the ACTUAL
                # measured robot position.
                action = read_state(robot)

            # ------------------------------------------------
            # POLICY ACTION -> SLEW LIMIT -> ROBOT
            # ------------------------------------------------

            action, was_clipped = slew_limit(
                action,
                last_command,
                args.max_tick_deg,
            )
            clipped_ticks += int(was_clipped)

            command = {
                key: float(value)
                for key, value in zip(
                    ACTION_KEYS,
                    action,
                    strict=True,
                )
            }

            robot.send_action(command)
            last_command = action
            latest_executed = step

            # ------------------------------------------------
            # FUTURE QUEUE SIZE
            # ------------------------------------------------

            remaining = sum(
                1
                for timestep in action_buffer
                if timestep > step
            )

            queue_history.append(remaining)

            # ------------------------------------------------
            # REQUEST NEXT CHUNK AT <= 30 REMAINING
            # ------------------------------------------------

            if (
                remaining <= REPLAN_REMAINING
                and not inference_inflight
                and step + 1 < total_steps
            ):
                # Capture ONE fresh logical observation after the current
                # action command, preserving the known-good sequencing.
                measured_state, images = capture_policy_observation(
                    robot,
                    top,
                    side,
                )

                source_step = max(latest_executed, 0)

                send_inference_request(
                    conn,
                    source_step=source_step,
                    state=measured_state,
                    images=images,
                )

                inference_inflight = True
                request_time = time.monotonic()

                print(
                    f"\n[ASYNC REQUEST]"
                    f" source={source_step}"
                    f" remaining={remaining}"
                    f" pan={measured_state[0]:.2f}"
                    f" grip={measured_state[5]:.2f}"
                )

            # ------------------------------------------------
            # 30 Hz TIMING
            # ------------------------------------------------

            elapsed = time.perf_counter() - loop_start
            sleep_time = (1.0 / FPS) - elapsed

            if sleep_time > 0:
                time.sleep(sleep_time)

        # ====================================================
        # FINISHED
        # ====================================================

        actual_duration = time.perf_counter() - rollout_start

        print("\n============================================")
        print(" ROLLOUT FINISHED")
        print("============================================")
        print(f"Requested duration: {args.duration:.2f} s")
        print(f"Actual duration:    {actual_duration:.2f} s")
        print(f"Actions sent:       {total_steps}")
        print(
            f"Effective rate:     "
            f"{total_steps / actual_duration:.2f} Hz"
        )
        print(f"Inferences:         {inference_count}")
        print(
            f"Mean inference:     "
            f"{np.mean(inference_latencies) * 1000:.1f} ms"
        )
        print(
            f"Max inference:      "
            f"{np.max(inference_latencies) * 1000:.1f} ms"
        )

        if queue_history:
            print(f"Queue minimum:      {min(queue_history)}")
            print(f"Queue maximum:      {max(queue_history)}")

        print(f"Queue underruns:    {underrun_count}")
        print(f"Slew-limited ticks: {clipped_ticks}")

        if underrun_count == 0:
            print("\nASYNC SUCCESS: no queue underruns.")

    finally:
        if conn is not None:
            try:
                if not inference_inflight:
                    conn.send({"type": "stop"})
            except Exception:
                pass

            try:
                conn.close()
            except Exception:
                pass

        if proc is not None:
            try:
                proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                proc.terminate()
                try:
                    proc.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    proc.kill()

        if top is not None:
            top.close()

        if side is not None:
            side.close()

        if robot is not None and robot.is_connected:
            print("\nDisconnecting robot...")
            robot.disconnect()

        print("Done.")


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "212_v3 SmolVLA AOT TensorRT rollout using the "
            "known-good async control logic"
        )
    )

    parser.add_argument(
        "--duration",
        type=float,
        default=20.0,
        help="Rollout duration in seconds",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="SmolVLA flow-matching RNG seed",
    )

    parser.add_argument(
        "--inference-timeout",
        type=float,
        default=8.0,
        help="Abort if one AOT inference exceeds this many seconds",
    )

    parser.add_argument(
        "--wake-nudge",
        type=float,
        default=12.0,  # same as V4
        help=(
            "Degrees to ramp wrist_flex down after ENTER, before the first "
            "inference, to trigger the policy's start from a still rest pose "
            "(without it the policy holds at rest; 0 disables)"
        ),
    )

    parser.add_argument(
        "--max-tick-deg",
        type=float,
        default=12.0,
        help=(
            "Slew limit: max change of any commanded joint per 30 Hz tick, as "
            "in V4; 0 sends policy actions unclipped like the original v3"
        ),
    )

    parser.add_argument(
        "--max-first-delta",
        type=float,
        default=20.0,
        help=(
            "Refuse to start if the first action is further than this many "
            "degrees from the measured state"
        ),
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Start despite a large first action (prints a warning instead)",
    )

    # Internal worker options.
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--fd", type=int, default=-1, help=argparse.SUPPRESS)

    args = parser.parse_args()

    if args.worker:
        if args.fd < 0:
            parser.error("--worker requires --fd")

        worker_main(args.fd, args.seed)

    else:
        if args.duration <= 0:
            parser.error("--duration must be > 0")

        if args.inference_timeout <= 0:
            parser.error("--inference-timeout must be > 0")

        if not 0 <= args.wake_nudge <= 30:
            parser.error("--wake-nudge must be between 0 and 30 degrees")

        if args.max_tick_deg < 0:
            parser.error("--max-tick-deg must be >= 0 (0 disables the slew limit)")

        if args.max_first_delta <= 0:
            parser.error("--max-first-delta must be > 0")

        try:
            main(args)
        except KeyboardInterrupt:
            print("\nCTRL+C received. Stopping rollout.")
        except Exception as exc:
            print(f"ROLLOUT ABORTED: {exc}", file=sys.stderr)
            sys.exit(1)
