"""Settings of the lab setup for the host-side scripts.

Edit the values below, or put your own values in site_local.py next to this file (it is read last and
is not part of the repository). Every host script imports these names instead of hard-coding them.
"""
import glob
import subprocess
from pathlib import Path


def _first(pattern, fallback):
    hits = sorted(glob.glob(pattern))
    return hits[0] if hits else fallback


# ---- the Jetson -------------------------------------------------------------------------------------
# ssh target of the Jetson. Log-in must work without a password prompt (ssh-copy-id).
JETSON = "user@jetson"
# Absolute folder on the Jetson that receives a copy of ../jetson and ../data (deploy_jetson() below).
JETSON_DIR = "/home/user/emfi_robot"
# Python of the TensorRT runtime's virtual environment on the Jetson (see ../jetson/README.md).
AOT_PYTHON = "/home/user/vla-runtime/.venv/bin/python"
# Shell command that activates the LeRobot environment on the Jetson (the rollout's main process).
CONDA_ACTIVATE = "source ~/miniforge3/etc/profile.d/conda.sh && conda activate lerobot"

# ---- devices on the host ----------------------------------------------------------------------------
# NewAE ChipSHOUTER (USB serial).
SHOUTER_PORT = _first("/dev/serial/by-id/usb-NewAE_ChipSHOUTER*", "/dev/serial/by-id/usb-NewAE_ChipSHOUTER")
# Optional Arduino that pulls the Jetson's reset line when it receives "10\n" (crash recovery during long
# campaigns). None: no hard reset, a hung Jetson stops the campaign.
ARDUINO_PORT = None

# ---- simulation (swfi2.py) --------------------------------------------------------------------------
# The fine-tuned SmolVLA checkpoint (LeRobot "pretrained_model" folder).
SMOLVLA_CKPT = "/path/to/smolvla/checkpoints/040000/pretrained_model"
# The 8 logged observations the simulations and the Jetson-side classifications use.
TRACES = Path(__file__).resolve().parents[1] / "data" / "traces"

try:
    from site_local import *  # noqa: F401,F403  -- lab-specific overrides, not in the repository
except ImportError:
    pass


def deploy_jetson(timeout=180):
    """Copy ../jetson/* (scripts, site settings) and ../data to JETSON_DIR on the Jetson."""
    root = Path(__file__).resolve().parents[1]
    files = [str(p) for p in sorted((root / "jetson").iterdir()) if p.is_file()]
    ssh = ["ssh", "-o", "BatchMode=yes", JETSON]
    subprocess.run(ssh + [f"mkdir -p {JETSON_DIR}/data ~/fi_logs/v3"], check=True, timeout=30)
    subprocess.run(["scp", "-pq", *files, f"{JETSON}:{JETSON_DIR}/"], check=True, timeout=timeout)
    subprocess.run(["scp", "-pqr", str(root / "data" / "traces"), str(root / "data" / "canary"),
                    f"{JETSON}:{JETSON_DIR}/data/"], check=True, timeout=timeout)
