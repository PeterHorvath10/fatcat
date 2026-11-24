"""Connection to the Jetson worker and the host side of one trial handshake (common.protocol)."""

from __future__ import annotations

import socket
import time
from dataclasses import dataclass

from common import protocol

# How the host answers COLLECT (which fields the worker expects):
#   short        attempt index + collection id                   (memory-check workers)
#   slot         + prompt slot + prompt repeat                    (Orin Ollama query worker)
#   slot_weights + prompt slot + prompt repeat + check-weights    (Thor vLLM query/reload workers)
ACK_FORMS = ("short", "slot", "slot_weights")


@dataclass(frozen=True)
class TargetSettings:
    host: str
    port: int
    socket_timeout_s: float
    connect_retry_s: float
    connect_settle_s: float
    ack_form: str
    weight_report: bool


def connect(t: TargetSettings) -> socket.socket:
    """Connect, retrying until the worker is up (it may be rebooting after a reset)."""
    while True:
        try:
            print(f"Connecting to {t.host}:{t.port}...")
            sock = socket.create_connection((t.host, t.port))
            time.sleep(t.connect_settle_s)
            sock.settimeout(t.socket_timeout_s)
            print("Socket connected.")
            return sock
        except OSError as exc:
            print(f"Connect failed: {exc}; retrying in {t.connect_retry_s:g}s")
            time.sleep(t.connect_retry_s)


def acknowledgement(t: TargetSettings, attempt_index: int, collection_id: str,
                    prompt_slot: int | None, prompt_repeat: int | None) -> bytes:
    if t.ack_form == "short":
        return protocol.acknowledgement(attempt_index, collection_id)
    check = None
    if t.ack_form == "slot_weights":
        check = protocol.CHECK_WEIGHTS_REPORT if t.weight_report else 0
    return protocol.acknowledgement(attempt_index, collection_id, prompt_slot, prompt_repeat, check_weights=check)


def print_weight_report(sock: socket.socket) -> None:
    """Print the worker's end-of-trial check of the loaded weights. weight_reload/until_hit.py matches the
    'Weight check (attempt N): CHANGED' line."""
    attempt, status, changed, summary = protocol.recv_weight_report(sock)
    line = f"Weight check (attempt {attempt}): {status}"
    if status == "TRANSIENT":
        line += f" -- misread during the check, stored weights intact: {summary}"
    elif status != "MATCH":
        line += f" -- {changed} tensor(s) changed: {summary}"
    print(line, flush=True)
