"""Power-cycle a hung Jetson and bring the host's Ethernet link back.

Orin: an Arduino on the J14 header (board/arduino_ctr/arduino_ctr.ino) shorts the power pins when it reads
the configured message on its USB serial port (method "serial").
Thor: NVIDIA's boardctl (method "command", e.g. `boardctl -t thor-jetson-devkit power_on`), which needs root.

Commands run under `sudo -S` when the env var named by network.sudo_password_env holds this PC's sudo
password (never stored in a file); otherwise they run as they are (sudo may then prompt or use cached
credentials). After a reset the USB-Ethernet adapter to the Jetson comes back without its address, so the
interface is set up and the address added again.

    python -m host.emfi.reset EXPERIMENT_TOML      # one reset of that experiment's board, by hand
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from dataclasses import dataclass

SERIAL_CLOSE_DELAY_S = 0.05   # let the message leave the USB-serial buffer before the port closes


@dataclass(frozen=True)
class ResetSettings:
    method: str                      # "serial" or "command"
    command: list[str]
    serial_port: str | None
    serial_baud: int | None
    serial_message: str | None
    serial_open_wait_s: float | None
    boot_wait_s: float
    host_interface: str
    host_address: str
    sudo_password_env: str | None


def sudo_run(argv: list[str], password_env: str | None, quiet: bool = False) -> int:
    """Run argv under `sudo -S` with the password from $password_env, or plainly if that is unset."""
    password = os.environ.get(password_env) if password_env else None
    out = subprocess.DEVNULL if quiet else None
    if password:
        return subprocess.run(["sudo", "-S", "-p", "", *argv], input=password + "\n", text=True,
                              stdout=out, stderr=out).returncode
    return subprocess.run(argv, stdout=out, stderr=out).returncode


def serial_reset(port: str, baud: int, message: str, open_wait_s: float) -> None:
    import serial
    dev = serial.Serial(port=port, baudrate=baud)
    try:
        dev.reset_input_buffer()
        print("About to reset the Jetson via the Arduino...")
        time.sleep(open_wait_s)      # opening the port resets the Arduino; wait until its sketch runs
        dev.write(message.encode() + b"\n")
        dev.flush()
        dev.reset_input_buffer()
        time.sleep(SERIAL_CLOSE_DELAY_S)
    finally:
        dev.close()
    print("Reset sent from Arduino.")


def power_cycle(r: ResetSettings) -> None:
    if r.method == "serial":
        serial_reset(r.serial_port, r.serial_baud, r.serial_message, r.serial_open_wait_s)
    elif r.method == "command":
        sudo_run(r.command, r.sudo_password_env)
    else:
        raise ValueError(f"unknown reset method {r.method!r} (serial or command)")


def ethernet_up(r: ResetSettings) -> None:
    sudo_run(["ip", "link", "set", r.host_interface, "up"], r.sudo_password_env)
    sudo_run(["ip", "addr", "add", r.host_address, "dev", r.host_interface], r.sudo_password_env)


def reset_board(r: ResetSettings) -> None:
    """Power-cycle, wait for the boot, re-up the link. Prints 'Sent reset.' (counted by the hunt scripts)."""
    power_cycle(r)
    print("Sent reset.", flush=True)
    time.sleep(r.boot_wait_s)
    ethernet_up(r)


def settings_from_config(cfg) -> ResetSettings:
    board = f"boards.{cfg.get('experiment.board')}"
    method = cfg.get(f"{board}.reset_method")
    serial = method == "serial"
    return ResetSettings(
        method=method,
        command=[] if serial else cfg.get(f"{board}.reset_command", str),
        serial_port=cfg.get(f"{board}.serial_port") if serial else None,
        serial_baud=cfg.get(f"{board}.serial_baud", int) if serial else None,
        serial_message=cfg.get(f"{board}.serial_message", str) if serial else None,
        serial_open_wait_s=cfg.get(f"{board}.serial_open_wait_s", float) if serial else None,
        boot_wait_s=cfg.get(f"{board}.boot_wait_s", float),
        host_interface=cfg.get("network.host_interface"),
        host_address=cfg.get("network.host_address"),
        sudo_password_env=cfg.get("network.sudo_password_env", default=None),
    )


if __name__ == "__main__":
    from common import config
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    reset_board(settings_from_config(config.load("host.toml", sys.argv[1])))
