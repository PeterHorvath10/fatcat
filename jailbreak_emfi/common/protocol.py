"""Wire protocol between the host controller and the Jetson worker (both sides import this file).

The Jetson worker is the TCP server, the host connects. One trial:

    worker -> host   COLLECT                       (u32)  "ready for a trial"
    host   -> worker COLLECT + acknowledgement            (one sendall)
                     attempt index                 (u64)
                     collection id                 (16 bytes, UUID)
                     [prompt slot                  (u64)  prompt-slot form only
                      prompt repeat, 1-based       (u64)
                      check-weights flag           (u64)]
    worker runs the workload; the scope edge triggers the glitcher
    worker -> host   [weight report]                      only if check-weights == CHECK_WEIGHTS_REPORT

All integers are little-endian and unsigned. These values define the protocol; they are not settings.
"""

from __future__ import annotations

import select
import socket
import uuid
from dataclasses import dataclass

BYTE_ORDER = "little"
U32 = 4
U64 = 8
UUID_BYTES = 16

COLLECT_TOKEN = 3354
COLLECT_COMMAND = COLLECT_TOKEN.to_bytes(U32, BYTE_ORDER)

# check-weights value asking the worker to verify the loaded weights after the trial and send a weight report
CHECK_WEIGHTS_REPORT = 2
WEIGHT_REPORT_TOKEN = 3355
WEIGHT_REPORT_STATUS = {0: "MATCH", 1: "CHANGED", 2: "ERROR", 3: "TRANSIENT"}
WEIGHT_REPORT_CODE = {name: code for code, name in WEIGHT_REPORT_STATUS.items()}

PROMPT_FIELDS_BYTES = 2 * U64
# The prompt-slot fields arrive in the same sendall() as the short acknowledgement; this is how long the
# worker waits for them before it treats the acknowledgement as the short form.
PROMPT_FIELDS_WAIT_S = 0.05


@dataclass(frozen=True)
class Handshake:
    attempt_index: int
    collection_id: str
    prompt_slot: int | None = None
    prompt_repeat: int | None = None
    check_weights: int | None = None


def u32(value: int) -> bytes:
    return value.to_bytes(U32, BYTE_ORDER)


def u64(value: int) -> bytes:
    return value.to_bytes(U64, BYTE_ORDER, signed=False)


def recv_exact(sock: socket.socket, size: int) -> bytes:
    data = bytearray()
    while len(data) < size:
        chunk = sock.recv(size - len(data))
        if not chunk:
            raise ConnectionError("peer closed the connection")
        data.extend(chunk)
    return bytes(data)


def recv_int(sock: socket.socket, size: int) -> int:
    return int.from_bytes(recv_exact(sock, size), BYTE_ORDER, signed=False)


# ---- host side -------------------------------------------------------------------------------------------

def wait_for_collect(sock: socket.socket) -> None:
    while recv_int(sock, U32) != COLLECT_TOKEN:
        pass


def acknowledgement(attempt_index: int, collection_id: str, prompt_slot: int | None = None,
                    prompt_repeat: int | None = None, check_weights: int | None = 0) -> bytes:
    """The host's reply to COLLECT. With prompt_slot it is the prompt-slot form (all three trailing fields);
    check_weights=None leaves the check-weights field out (slot + repeat only, as the Orin Ollama worker reads
    it with request_trial_any)."""
    msg = COLLECT_COMMAND + u64(attempt_index) + uuid.UUID(collection_id).bytes
    if prompt_slot is not None:
        msg += u64(prompt_slot) + u64(prompt_repeat or 0)
        if check_weights is not None:
            msg += u64(check_weights)
    return msg


def recv_weight_report(sock: socket.socket) -> tuple[int, str, int, str]:
    """(attempt index, status name, changed-tensor count, JSON summary)."""
    token = recv_int(sock, U32)
    if token != WEIGHT_REPORT_TOKEN:
        raise ConnectionError(f"expected weight report token {WEIGHT_REPORT_TOKEN}, got {token}")
    attempt = recv_int(sock, U64)
    status = recv_exact(sock, 1)[0]
    changed = recv_int(sock, U32)
    summary = recv_exact(sock, recv_int(sock, U32)).decode("utf-8", errors="replace")
    return attempt, WEIGHT_REPORT_STATUS.get(status, f"STATUS {status}"), changed, summary


# ---- worker side -----------------------------------------------------------------------------------------

def _request(sock: socket.socket) -> tuple[int, str]:
    sock.sendall(COLLECT_COMMAND)
    while recv_exact(sock, U32) != COLLECT_COMMAND:
        pass
    attempt = recv_int(sock, U64)
    collection = uuid.UUID(bytes=recv_exact(sock, UUID_BYTES)).hex
    return attempt, collection


def request_trial(sock: socket.socket) -> Handshake:
    """Short form: attempt index and collection id only."""
    return Handshake(*_request(sock))


def request_prompt_trial(sock: socket.socket) -> Handshake:
    """Prompt-slot form: also prompt slot, prompt repeat and the check-weights flag."""
    attempt, collection = _request(sock)
    return Handshake(attempt, collection, recv_int(sock, U64), recv_int(sock, U64), recv_int(sock, U64))


def request_trial_any(sock: socket.socket) -> Handshake:
    """Either form: prompt fields are present exactly when more bytes follow the short acknowledgement."""
    attempt, collection = _request(sock)
    readable, _, _ = select.select([sock], [], [], PROMPT_FIELDS_WAIT_S)
    if not readable:
        return Handshake(attempt, collection)
    if not sock.recv(PROMPT_FIELDS_BYTES, socket.MSG_PEEK | socket.MSG_DONTWAIT):
        return Handshake(attempt, collection)
    return Handshake(attempt, collection, recv_int(sock, U64), recv_int(sock, U64))


def send_weight_report(sock: socket.socket, attempt_index: int, status: str, changed: int, summary: bytes) -> None:
    sock.sendall(u32(WEIGHT_REPORT_TOKEN) + u64(attempt_index) + bytes([WEIGHT_REPORT_CODE[status]])
                 + u32(changed) + u32(len(summary)) + summary)


def serve_one(host: str, port: int) -> socket.socket:
    """Listen on host:port and return the first connection (the caller closes it)."""
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)   # rebind at once after a restart
    server.bind((host, port))
    server.listen(1)
    print(f"Waiting for connection on {host}:{port}", flush=True)
    conn, addr = server.accept()
    server.close()
    print(f"Connection from {addr}", flush=True)
    return conn
