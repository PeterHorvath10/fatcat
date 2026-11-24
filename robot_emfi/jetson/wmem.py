#!/usr/bin/env python3
"""GPU weight memory of the deployed TensorRT engines, for the Jetson-side tools (imported, not run).

TensorRT's weight memory is found with a recording GPU allocator on the runtime: what an engine
allocates while it deserializes is its weights (the chain's scratch and I/O buffers come from its own
cudaMalloc). weight_map() looks the ONNX initializers up in that memory, and locate() names the weight
a 16-byte block falls in; d2h() and h2d() read and write device memory. outcome() is the class of
Section 6.2 of a chunk against the fault-free one, and load_obs() reads the 8 reference observations
(OBS, the same as swfi2.py's) from ../data/traces.
"""

import ctypes
import itertools
from pathlib import Path

import numpy as np

import onnx_inits

from site_config import TRACES as LOGS           # one trace.npz per logged rollout
OBS = [("20261027_190527", 0), ("20261027_190527", 418), ("20261027_191033_seed1", 247),
       ("20261027_191155_seed2", 57), ("20261027_191155_seed2", 513), ("20261027_191301_seed3", 323),
       ("20261027_191352_seed4", 152), ("20261027_191352_seed4", 589)]     # swfi2.py's (run, step)
EXPERT = {"decode": "smolvlm_expert_decode.onnx", "time_in": "time_in_projector.onnx",
          "time_out": "time_out_projector.onnx", "action_in": "action_in_projector.onnx",
          "action_out": "action_out_projector.onnx"}
BLOCK = 16
LIM = [(-120.4, 120.4), (-105, 105), (-97.9, 97.9), (-103.9, 103.9), (-180, 180), (0, 100)]   # move_analyze.py


# ---- weight memory ------------------------------------------------------------------------------
class Recorder:
    """GPU allocator for the TensorRT runtime that remembers each live allocation and which engine
    was deserializing when it was made."""

    def __init__(self, trt):
        class Alloc(trt.IGpuAllocator):
            def allocate(s, size, alignment, flags):
                p = ctypes.c_void_p()
                if cuda().cudaMalloc(ctypes.byref(p), ctypes.c_size_t(max(size, 1))):
                    return None
                self.live[p.value] = (size, self.tag)
                return p.value

            def allocate_async(s, size, alignment, flags, stream):
                return s.allocate(size, alignment, flags)

            def deallocate(s, memory):
                self.live.pop(memory, None)
                return cuda().cudaFree(ctypes.c_void_p(memory)) == 0

            def deallocate_async(s, memory, stream):
                return s.deallocate(memory)

        self.live, self.tag = {}, None             # ptr -> (size, engine), in allocation order
        self.alloc = Alloc()


_cudart = None


def cuda():
    global _cudart
    if _cudart is None:
        _cudart = ctypes.CDLL("libcudart.so.13")
    return _cudart


def install_recorder(cache):
    """Makes every trt.Runtime use the recorder; allocations are tagged with the engine name, found
    from the serialized size (the engine files differ in size)."""
    import tensorrt as trt
    rec = Recorder(trt)
    names = {p.stat().st_size: p.stem for p in Path(cache).glob("*.engine")}
    real = trt.Runtime

    class Runtime:
        def __init__(self, logger):
            self.rt = real(logger)
            self.rt.gpu_allocator = rec.alloc

        def deserialize_cuda_engine(self, blob):
            rec.tag = names.get(len(blob), f"{len(blob)} B")
            try:
                return self.rt.deserialize_cuda_engine(blob)
            finally:
                rec.tag = None

        def __getattr__(self, k):
            return getattr(self.rt, k)

    trt.Runtime = Runtime
    return rec


def d2h(ptr, n):
    out = np.empty(n, np.uint8)
    rc = cuda().cudaMemcpy(out.ctypes.data_as(ctypes.c_void_p), ctypes.c_void_p(ptr), ctypes.c_size_t(n), 2)
    if rc:
        raise RuntimeError(f"D2H failed: cudaError {rc}")
    return out


def h2d(ptr, data):
    data = np.ascontiguousarray(np.frombuffer(bytes(data), np.uint8))
    rc = cuda().cudaMemcpy(ctypes.c_void_p(ptr), data.ctypes.data_as(ctypes.c_void_p), ctypes.c_size_t(data.nbytes), 1)
    if rc:
        raise RuntimeError(f"H2D failed: cudaError {rc}")


def onnx_weights(path):
    """(name, dims, dtype, raw bytes) of each initializer that holds raw data."""
    b = open(path, "rb").read()
    graph = next(v for f, w, v in onnx_inits.fields(b, 0, len(b)) if f == 7 and w == 2)
    for f, w, v in onnx_inits.fields(b, *graph):
        if f != 5 or w != 2:
            continue
        name, dims, dt, raw = "", [], 0, None
        for g, wt, x in onnx_inits.fields(b, *v):
            if g == 8:
                name = b[x[0]:x[1]].decode()
            elif g == 1 and wt == 0:
                dims.append(x)
            elif g == 1 and wt == 2:
                j = x[0]
                while j < x[1]:
                    d, j = onnx_inits.varint(b, j)
                    dims.append(d)
            elif g == 2:
                dt = x
            elif g == 9:
                raw = b[x[0]:x[1]]
        if raw:
            yield name, dims, onnx_inits.DTYPES.get(dt, str(dt)), raw


FUSED = (("q_proj", "k_proj", "v_proj"), ("q_proj", "k_proj"), ("k_proj", "v_proj"), ("gate_proj", "up_proj"))


def weight_map(regions, bundle, onnx_files=EXPERT):
    """Where each ONNX initializer (all [out, in] or 1-D) lies in the engines' weight memory, checked
    byte for byte. Returns entries (region index, start, end, name, layout, parts, n_in, itemsize) and
    the initializers not found. Layouts:
      row    as in ONNX: [out][in], a block is 8 inputs of one output (4 for fp32)
      col    [in][out], a block is 8 outputs of one input
      fused  several projections of one input side by side, [in][part 1 out | part 2 out | ...]
             (the expert's self-attention Q, K and V; the prefill's Q and K, K and V, and gate and up)
      inter  two projections interleaved per output, [in][out][part 1, part 2] (the expert MLP's gate and up)
      flat   1-D (norm weights, biases)
    parts lists the fused projections in memory order as "name:outputs,...". """
    found, missing = [], []
    npdt = {"fp16": np.float16, "fp32": np.float32}
    for eng, onnx in onnx_files.items():
        mems = [(ri, d2h(ptr, size).tobytes()) for ri, (e, ptr, size) in enumerate(regions) if e == eng]
        W = {n: np.frombuffer(raw, npdt[dt]).reshape(dims) for n, dims, dt, raw in onnx_weights(bundle / onnx) if dt in npdt}

        def find(blob):
            for ri, mem in mems:
                at = mem.find(blob[:64])
                while at >= 0:
                    if mem[at:at + len(blob)] == blob:
                        return ri, at
                    at = mem.find(blob[:64], at + 1)
            return None

        done = set()
        for group in FUSED:                             # fused groups first, per layer
            for n in [n for n in W if n.endswith(f".{group[0]}.weight")]:
                names = [n.replace(group[0], g) for g in group]
                if not all(m in W for m in names) or done & set(names):
                    continue
                same = len({W[m].shape for m in names}) == 1
                layouts = ("inter", "fused") if len(group) == 2 and same else ("fused",)
                for layout, order in itertools.product(layouts, itertools.permutations(names)):
                    if layout == "inter":
                        blob = np.stack([W[m].T for m in order], -1).tobytes()
                    else:
                        blob = np.concatenate([W[m].T for m in order], 1).tobytes()
                    hit = find(blob)
                    if hit:
                        parts = ",".join(f"{m.split('.')[-2]}:{W[m].shape[0]}" for m in order)
                        found.append((*hit, hit[1] + len(blob), n.rsplit(".", 2)[0], layout, parts,
                                      W[n].shape[1], W[n].itemsize))
                        done.update(names)
                        break
        for n, w in W.items():
            if n in done:
                continue
            tries = [("row" if w.ndim == 2 else "flat", w.tobytes())]
            if w.ndim == 2:
                tries.append(("col", np.ascontiguousarray(w.T).tobytes()))
            for layout, blob in tries:
                hit = find(blob) if len(blob) >= 64 else None
                if hit:
                    found.append((*hit, hit[1] + len(blob), n, layout, "", w.shape[-1], w.itemsize))
                    break
            else:
                missing.append((eng, n, list(w.shape), w.dtype.name))
        del mems
    return found, missing


def locate(wmap, ri, off):
    """The weight a block falls in and its first element: (name, layout, "out o in i")."""
    for r, s, e, name, layout, parts, n_in, isz in wmap:
        if r != ri or not s <= off < e:
            continue
        el = (off - s) // isz
        if layout == "flat":
            return name, layout, f"elem {el}"
        if layout == "row":
            return name, layout, f"out {el // n_in} in {el % n_in}"
        n_out = (e - s) // isz // n_in
        i, j = divmod(el, n_out)
        if layout == "col":
            return name, layout, f"out {j} in {i}"
        p = [(q.split(":")[0], int(q.split(":")[1])) for q in parts.split(",")]
        if layout == "inter":
            return f"{name}.{'+'.join(q for q, _ in p)}", layout, f"out {j // len(p)} in {i}"
        for q, n in p:
            if j < n:
                return f"{name}.{q}", layout, f"out {j} in {i}"
            j -= n
    return "", "", ""


# ---- policy -------------------------------------------------------------------------------------
def outcome(c, clean):
    """The EM campaigns' classes (move_analyze.py) on a chunk in degrees."""
    if not np.isfinite(c).all():
        return "nan", float("nan")
    d = float(np.abs(c - clean).max())
    if d == 0:
        return "unchanged", d
    if d < 10:
        return "small", d
    inside = all(LIM[j][0] - 0.5 <= c[:, j].min() and c[:, j].max() <= LIM[j][1] + 0.5 for j in range(6))
    return ("silent" if inside and np.abs(np.diff(c, axis=0)).max() <= 10 else "erratic"), d


def load_obs():
    """(run, step, top image, side image, joint state) of each of the 8 reference observations."""
    obs = []
    for run in dict.fromkeys(r for r, _ in OBS):
        d = np.load(LOGS / run / "trace.npz")
        steps = list(d["req_step"])
        for r, s in OBS:
            if r == run:
                i = steps.index(s)
                obs.append((run, s, d["req_top"][i].copy(), d["req_side"][i].copy(), d["req_state"][i].copy()))
    return obs

