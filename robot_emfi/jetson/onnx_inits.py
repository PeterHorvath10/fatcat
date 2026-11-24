#!/usr/bin/env python3
"""List an ONNX graph's initializers (name, dims, dtype) without the onnx package.

Reads the protobuf directly: ModelProto.graph (7) -> GraphProto.initializer (5) -> TensorProto
dims (1), data_type (2), name (8). Shows which weights a graph (and so the TensorRT engine built
from it) actually reads.
usage: onnx_inits.py MODEL.onnx [...]
"""
import mmap
import sys

DTYPES = {1: "fp32", 10: "fp16", 7: "int64", 6: "int32", 9: "bool", 16: "bf16"}


def varint(b, i):
    v = s = 0
    while True:
        c = b[i]
        i += 1
        v |= (c & 0x7F) << s
        s += 7
        if c < 0x80:
            return v, i


def fields(b, i, end):
    """Yield (field number, wire type, value or (start, end)) of one message."""
    while i < end:
        key, i = varint(b, i)
        f, w = key >> 3, key & 7
        if w == 0:
            v, i = varint(b, i)
            yield f, w, v
        elif w == 2:
            n, i = varint(b, i)
            yield f, w, (i, i + n)
            i += n
        elif w == 1:
            yield f, w, b[i:i + 8]
            i += 8
        elif w == 5:
            yield f, w, b[i:i + 4]
            i += 4
        else:
            raise ValueError(f"wire type {w} at {i}")


def initializers(path):
    with open(path, "rb") as fh:
        b = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
        graph = next(v for f, w, v in fields(b, 0, len(b)) if f == 7 and w == 2)
        for f, w, v in fields(b, *graph):
            if f != 5 or w != 2:
                continue
            name, dims, dt = "", [], 0
            for g, wt, x in fields(b, *v):
                if g == 8:
                    name = bytes(b[x[0]:x[1]]).decode()
                elif g == 1 and wt == 0:
                    dims.append(x)
                elif g == 1 and wt == 2:                    # packed dims
                    j = x[0]
                    while j < x[1]:
                        d, j = varint(b, j)
                        dims.append(d)
                elif g == 2:
                    dt = x
            yield name, dims, DTYPES.get(dt, str(dt))


if __name__ == "__main__":
    for p in sys.argv[1:]:
        rows = list(initializers(p))
        n = sum(1 for _ in rows)
        params = sum(eval("*".join(map(str, d)) or "1") for _, d, _ in rows)
        print(f"== {p}: {n} initializers, {params / 1e6:.1f} M elements")
        for name, d, t in rows:
            print(f"{t:5s} {str(d):18s} {name}")
