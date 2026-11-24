"""JSON over HTTP (urllib only: the workers run with the system Python, outside the vLLM venv)."""

from __future__ import annotations

import json
import time
from typing import Any, Callable, TypeVar
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

T = TypeVar("T")
# Errors that mean "the server is not (yet) answering properly"; poll() retries on them.
TRANSIENT_ERRORS = (HTTPError, URLError, TimeoutError, ConnectionError, json.JSONDecodeError, ValueError)


def get(url: str, timeout: float) -> bytes:
    with urlopen(Request(url, method="GET"), timeout=timeout) as response:
        return response.read()


def get_json(url: str, timeout: float) -> Any:
    with urlopen(Request(url, headers={"Accept": "application/json"}, method="GET"), timeout=timeout) as response:
        return json.load(response)


def post(url: str, body: Any, timeout: float) -> bytes:
    """POST body as JSON (None sends an empty body) and return the raw response."""
    data = b"" if body is None else json.dumps(body).encode("utf-8")
    request = Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    with urlopen(request, timeout=timeout) as response:
        return response.read()


def post_json(url: str, body: Any, timeout: float) -> Any:
    payload = post(url, body, timeout)
    return json.loads(payload) if payload else None


def http_error_detail(error: HTTPError) -> str:
    return f"HTTP {error.code}: {error.read().decode('utf-8', errors='replace')}"


def poll(what: str, probe: Callable[[], T | None], retry_s: float, timeout_s: float = 0.0) -> T:
    """Call probe() until it returns something other than None; timeout_s 0 waits forever.

    probe may raise one of TRANSIENT_ERRORS (or return None after setting nothing) while the server
    starts; the last reason is printed with every retry.
    """
    deadline = time.monotonic() + timeout_s if timeout_s else None
    reason = "not ready"
    while True:
        try:
            value = probe()
            if value is not None:
                return value
        except HTTPError as error:
            reason = f"HTTP {error.code}"
        except TRANSIENT_ERRORS as error:
            reason = str(error)
        remaining = deadline - time.monotonic() if deadline is not None else float("inf")
        if remaining <= 0:
            raise TimeoutError(f"timed out waiting for {what}: {reason}")
        print(f"Waiting for {what} ({reason})", flush=True)
        time.sleep(min(retry_s, remaining))
