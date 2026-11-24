"""Settings of the Jetson for the Jetson-side scripts and the rollout.

Edit the values below, or put your own values in site_local.py next to this file (it is read last and
is not part of the repository; the host's deploy copies it along).
"""
from pathlib import Path

HOME = Path.home()
HERE = Path(__file__).resolve().parent

# The TensorRT runtime (github.com/eetmie/jetson-orin-nano-vla, commit 6e2d857) with patches/vla-runtime.patch applied.
VLA_REPO = HOME / "vla-runtime"
AOT_PYTHON = VLA_REPO / ".venv/bin/python"
# ONNX bundle exported from the fine-tuned SmolVLA, and the TensorRT engines the runtime built from it.
BUNDLE = HOME / "onnx_bundles/212_v4_bs64_s40000_FP16"
AOT_CACHE = HOME / ".cache/vla-runtime" / f"{BUNDLE.name}__aot-engines" / "optimized-cache"   # where the runtime built them

# Inputs deployed with the scripts (../data on the host).
TRACES = HERE / "data" / "traces"              # the 8 logged observations, one trace.npz per rollout
CANARY_RUN = HERE / "data" / "canary"          # first_obs_{top,side}.png of a fault-free rollout
CANARY_STATE = [-2.9, -103.7, 96.8, 62.8, 2.9, 24.5]   # the arm's joint state in those frames (deg)
# Outputs.
FI_LOGS = HOME / "fi_logs"
GOLDEN = FI_LOGS / "golden.npy"                # the canary's fault-free chunk, from capture_golden.py

# The SO-101 follower and the two cameras.
FOLLOWER_PORT = "/dev/serial/by-id/usb-1a86_USB_Single_Serial_SERIAL-if00"
CAMERA_SIDE = "/dev/v4l/by-path/platform-3610000.usb-usb-0:2.4:1.0-video-index0"
CAMERA_TOP = "/dev/v4l/by-path/platform-3610000.usb-usb-0:2.3:1.0-video-index0"

try:
    from site_local import *  # noqa: F401,F403  -- lab-specific overrides, not in the repository
except ImportError:
    pass
