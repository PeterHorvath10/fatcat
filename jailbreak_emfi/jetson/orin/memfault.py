"""Orin memory-check worker (memfault.service): locate the spots where the EM pulse corrupts GPU memory.

At start it allocates the buffers (memcheck.py) and records the physical pages behind each one
(pa_capture.py). Only then, at the first verification pass, it accepts the host controller on
network.listen_host:listen_port; every pass waits for one trial handshake and runs one glitched
read-and-checksum pass of the whole buffer with the GPIO line high.

    python3 -m jetson.orin.memfault
"""

from __future__ import annotations

import socket
import sys
from contextlib import ExitStack
from pathlib import Path

from common import config, protocol
from jetson.common.gpio import GpioOutput
from jetson.common.results import new_run_id
from jetson.orin import memcheck
from jetson.orin.cuda_driver import CudaError, CudaModule
from jetson.orin.pa_capture import AllocationMappingCapture

GIB = 1024 ** 3
JOIN_LOG = "run_capture_join.tsv"


def workload_from_config(cfg: config.Config) -> memcheck.Workload:
    return memcheck.Workload(
        nbytes=int(cfg.get("memfault.gib", float) * GIB),
        blocks=cfg.get("memfault.blocks", int),
        threads=cfg.get("memfault.threads", int),
        seed=cfg.get("memfault.seed", lambda v: int(v, 0) if isinstance(v, str) else int(v)),
        passes=cfg.get("memfault.passes", int),
        data_mismatch_capacity=cfg.get("memfault.data_mismatch_capacity", int),
        log_dir=Path(cfg.get("memfault.log_dir")),
    )


def main() -> int:
    cfg = config.load("orin.toml")
    workload = workload_from_config(cfg)
    run_id = new_run_id()
    host, port = cfg.get("network.listen_host"), cfg.get("network.listen_port", int)
    with ExitStack() as stack:
        capture = AllocationMappingCapture(Path(cfg.get("memfault.pa_query_tool")), Path(cfg.get("memfault.map_dir")),
                                           workload.log_dir / JOIN_LOG, len(memcheck.BUFFER_LABELS), run_id)
        print(f"CUDA mapping run directory: {capture.run_dir}", flush=True)
        gpio = stack.enter_context(GpioOutput(cfg.get("gpio.chip"), cfg.get("memfault.gpio_line", int),
                                              cfg.get("gpio.consumer")))
        cuda = stack.enter_context(CudaModule(memcheck.KERNEL_SOURCE.read_text(), memcheck.KERNEL_SOURCE.name,
                                              on_alloc=capture))
        conn: socket.socket | None = None

        def before_pass(pass_index: int) -> int:
            # Accept the controller lazily: allocation, PA capture and warm-up happen before it connects.
            nonlocal conn
            if conn is None:
                conn = stack.enter_context(protocol.serve_one(host, port))
            handshake = protocol.request_trial_any(conn)
            capture.record_handshake(pass_index, handshake.attempt_index, handshake.collection_id)
            print(f"Handshake complete: pass={pass_index} attempt_index={handshake.attempt_index} "
                  f"collection_id={handshake.collection_id}", flush=True)
            return handshake.attempt_index

        return memcheck.run(workload, cuda, run_id=run_id, before_pass=before_pass, gpio=gpio)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (CudaError, OSError, RuntimeError, ValueError, config.ConfigError) as error:
        print(f"error: {error}", flush=True)
        sys.exit(1)
