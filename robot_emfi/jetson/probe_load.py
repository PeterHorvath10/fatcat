#!/usr/bin/env python3
"""Continuously reload the decode engine so a near-field EMFI probe can be positioned over the chip and
a hardware trigger level found.

Each cycle idles the GPU (a clean quiet stretch on the probe), then deserializes decode.engine -- the
~193 MB host->device copy of the expert's o_proj/down/v/k/q + gate/up weights, the same burst the arm
experiment must pulse -- then frees it. So the probe sees, repeating every ~period_s: a quiet gap, then
a sharp rising edge into a ~30-40 ms busy burst. Position the probe for the biggest burst and set the
Husky/scope trigger on that rising edge. No HV, no arm, no inference.

usage (AOT venv on the Jetson, next to swfi.py): probe_load.py [idle_ms=120] [period_s=0.6]
stop with Ctrl-C or kill.
"""
import gc
import sys
import time

import tensorrt as trt

import swfi

idle_ms = float(sys.argv[1]) if len(sys.argv) > 1 else 120.0
period_s = float(sys.argv[2]) if len(sys.argv) > 2 else 0.6
blob = (swfi.AOT_CACHE / "decode.engine").read_bytes()
logger = trt.Logger(trt.Logger.WARNING)
print(f"probe_load: reload decode.engine ({len(blob)/1e6:.0f} MB) every ~{period_s*1e3:.0f} ms, "
      f"{idle_ms:.0f} ms idle before each copy -> rising edge then ~30-40 ms burst. Ctrl-C to stop.",
      flush=True)

i, t_report = 0, time.monotonic()
try:
    while True:
        t_cycle = time.monotonic()
        time.sleep(idle_ms / 1e3)                     # GPU idle: clean quiet before the copy edge
        t0 = time.perf_counter()
        rt = trt.Runtime(logger)
        eng = rt.deserialize_cuda_engine(blob)        # the host->device decode weight copy (the burst)
        des_ms = (time.perf_counter() - t0) * 1e3
        if eng is None:
            print("  deserialize returned None", flush=True)
        eng = None
        rt = None
        gc.collect()                                  # free the weights, so the next copy is a fresh edge
        i += 1
        if time.monotonic() - t_report >= 2.0:
            print(f"  {i} reloads, last copy {des_ms:.1f} ms", flush=True)
            t_report = time.monotonic()
        time.sleep(max(0.0, period_s - (time.monotonic() - t_cycle)))
except KeyboardInterrupt:
    print(f"stopped after {i} reloads", flush=True)
