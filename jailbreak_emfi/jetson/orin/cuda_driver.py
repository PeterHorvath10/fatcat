"""Thin CUDA driver API layer (cuda-python) for the Orin memory-check workload.

The kernels are compiled with NVRTC to native SASS for the device's own architecture (queried at run
time), so a newer NVRTC never hands an older driver PTX it cannot JIT.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np


class CudaError(RuntimeError):
    pass


@dataclass(frozen=True)
class ScalarArg:
    value: int
    dtype: np.dtype


def u32(value: int) -> ScalarArg:
    return ScalarArg(int(value), np.dtype(np.uint32))


def u64(value: int) -> ScalarArg:
    return ScalarArg(int(value), np.dtype(np.uint64))


def _load_bindings():
    try:
        from cuda.bindings import driver, nvrtc
    except ImportError:
        try:
            from cuda import cuda as driver
            from cuda import nvrtc
        except ImportError as exc:
            raise CudaError("cuda-python is not installed or not visible to this Python") from exc
    return driver, nvrtc


def _code(err) -> int:
    return int(getattr(err, "value", err))


def _unpack(values: list):
    if not values:
        return None
    return values[0] if len(values) == 1 else tuple(values)


def check_cuda(driver, result, call: str):
    err, *values = result if isinstance(result, tuple) else (result,)
    if _code(err) != 0:
        name = str(err)
        try:
            name_result = driver.cuGetErrorName(err)
            if isinstance(name_result, tuple) and _code(name_result[0]) == 0:
                raw = name_result[1]
                name = raw.decode() if isinstance(raw, bytes) else str(raw)
        except Exception:
            pass
        raise CudaError(f"{call} failed: {name}")
    return _unpack(values)


def _read_bytes(nvrtc, size_call, read_call, program, call: str) -> bytes:
    """Read an NVRTC output buffer (program log, CUBIN) whatever buffer type the bindings accept."""
    size = int(check_nvrtc(nvrtc, size_call(program), call + "Size"))
    if not size:
        return b""
    buffer = bytearray(size)
    try:
        check_nvrtc(nvrtc, read_call(program, buffer), call)
    except TypeError:
        buffer = b" " * size
        check_nvrtc(nvrtc, read_call(program, buffer), call)
    return bytes(buffer)


def check_nvrtc(nvrtc, result, call: str, program=None):
    err, *values = result if isinstance(result, tuple) else (result,)
    if _code(err) != 0:
        message = str(err)
        try:
            _, raw = nvrtc.nvrtcGetErrorString(err)
            message = raw.decode() if isinstance(raw, bytes) else str(raw)
        except Exception:
            pass
        if program is not None:
            log = _read_bytes(nvrtc, nvrtc.nvrtcGetProgramLogSize, nvrtc.nvrtcGetProgramLog, program,
                              "nvrtcGetProgramLog").decode("utf-8", errors="replace").strip("\x00 \n")
            if log:
                message = f"{message}\n\nNVRTC log:\n{log}"
        raise CudaError(f"{call} failed: {message}")
    return _unpack(values)


def device_pointer(value) -> int:
    try:
        return int(value)
    except TypeError:
        return int(getattr(value, "value", value))


class DeviceBuffer:
    def __init__(self, driver, nbytes: int, label: str):
        self._cuda, self.nbytes, self.label = driver, int(nbytes), label
        self.ptr = check_cuda(driver, driver.cuMemAlloc(self.nbytes), f"cuMemAlloc({label})")

    def copy_to_host(self, host: np.ndarray) -> None:
        host = np.ascontiguousarray(host)
        if host.nbytes > self.nbytes:
            raise ValueError(f"host array has {host.nbytes} bytes, buffer has {self.nbytes}")
        check_cuda(self._cuda, self._cuda.cuMemcpyDtoH(host.ctypes.data, self.ptr, host.nbytes),
                   f"cuMemcpyDtoH({self.label})")

    def copy_from_host(self, host: np.ndarray) -> None:
        host = np.ascontiguousarray(host)
        if host.nbytes > self.nbytes:
            raise ValueError(f"host array has {host.nbytes} bytes, buffer has {self.nbytes}")
        check_cuda(self._cuda, self._cuda.cuMemcpyHtoD(self.ptr, host.ctypes.data, host.nbytes),
                   f"cuMemcpyHtoD({self.label})")

    def copy_from_device(self, other: "DeviceBuffer", nbytes: int) -> None:
        if nbytes > self.nbytes or nbytes > other.nbytes:
            raise ValueError(f"copy of {nbytes} bytes does not fit {other.label}->{self.label}")
        check_cuda(self._cuda, self._cuda.cuMemcpyDtoD(self.ptr, other.ptr, nbytes),
                   f"cuMemcpyDtoD({other.label}->{self.label})")

    def free(self) -> None:
        if self.ptr is not None:
            check_cuda(self._cuda, self._cuda.cuMemFree(self.ptr), f"cuMemFree({self.label})")
            self.ptr = None


AllocHook = Callable[[DeviceBuffer], None]


class CudaModule:
    """Primary context on device 0, one stream, and the kernels of source compiled to native SASS.

    on_alloc is called with every buffer right after allocation (the PA capture hooks in here).
    """

    def __init__(self, source: str, source_name: str, on_alloc: AllocHook | None = None):
        self.cuda, self.nvrtc = _load_bindings()
        self.on_alloc = on_alloc
        self.module = self.stream = None
        check_cuda(self.cuda, self.cuda.cuInit(0), "cuInit")
        self.device = check_cuda(self.cuda, self.cuda.cuDeviceGet(0), "cuDeviceGet")
        self.ctx = check_cuda(self.cuda, self.cuda.cuDevicePrimaryCtxRetain(self.device), "cuDevicePrimaryCtxRetain")
        check_cuda(self.cuda, self.cuda.cuCtxSetCurrent(self.ctx), "cuCtxSetCurrent")
        self.stream = check_cuda(self.cuda, self.cuda.cuStreamCreate(0), "cuStreamCreate")
        self.arch = self._native_arch()
        print("Architecture: ", self.arch, flush=True)
        self._image = np.frombuffer(self._compile(source, source_name), dtype=np.uint8).copy()
        try:
            self.module = check_cuda(self.cuda, self.cuda.cuModuleLoadData(self._image), "cuModuleLoadData")
        except TypeError:
            self.module = check_cuda(self.cuda, self.cuda.cuModuleLoadData(self._image.ctypes.data),
                                     "cuModuleLoadData")

    def _native_arch(self) -> str:
        attr = self.cuda.CUdevice_attribute
        major = check_cuda(self.cuda, self.cuda.cuDeviceGetAttribute(
            attr.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, self.device), "cuDeviceGetAttribute(major)")
        minor = check_cuda(self.cuda, self.cuda.cuDeviceGetAttribute(
            attr.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, self.device), "cuDeviceGetAttribute(minor)")
        return f"sm_{major}{minor}"

    def _compile(self, source: str, name: str) -> bytes:
        nvrtc = self.nvrtc
        program = check_nvrtc(nvrtc, nvrtc.nvrtcCreateProgram(source.encode(), name.encode(), 0, [], []),
                              "nvrtcCreateProgram")
        options = [f"--gpu-architecture={self.arch}".encode(), b"--std=c++17"]
        check_nvrtc(nvrtc, nvrtc.nvrtcCompileProgram(program, len(options), options), "nvrtcCompileProgram",
                    program=program)
        image = _read_bytes(nvrtc, nvrtc.nvrtcGetCUBINSize, nvrtc.nvrtcGetCUBIN, program, "nvrtcGetCUBIN")
        check_nvrtc(nvrtc, nvrtc.nvrtcDestroyProgram(program), "nvrtcDestroyProgram")
        print(f"Compiled native CUDA image for {self.arch}", flush=True)
        return image

    def function(self, name: str):
        return check_cuda(self.cuda, self.cuda.cuModuleGetFunction(self.module, name.encode()),
                          f"cuModuleGetFunction({name})")

    def alloc(self, nbytes: int, label: str) -> DeviceBuffer:
        buffer = DeviceBuffer(self.cuda, nbytes, label)
        if self.on_alloc is not None:
            self.on_alloc(buffer)
        return buffer

    def mem_info(self) -> tuple[int, int]:
        free, total = check_cuda(self.cuda, self.cuda.cuMemGetInfo(), "cuMemGetInfo")
        return int(free), int(total)

    def synchronize(self) -> None:
        check_cuda(self.cuda, self.cuda.cuStreamSynchronize(self.stream), "cuStreamSynchronize")

    def _launch(self, function, grid, block, params: np.ndarray, shared_mem_bytes: int):
        try:
            return self.cuda.cuLaunchKernel(function, *grid, *block, int(shared_mem_bytes), self.stream,
                                            kernelParams=params, extra=0)
        except TypeError:       # older bindings take the parameter array by address, positionally
            return self.cuda.cuLaunchKernel(function, *grid, *block, int(shared_mem_bytes), self.stream,
                                            params.ctypes.data, 0)

    def launch(self, function, grid: tuple[int, int, int], block: tuple[int, int, int],
               args: Sequence[DeviceBuffer | ScalarArg], shared_mem_bytes: int = 0, gpio=None) -> None:
        """Launch a kernel. With gpio, the line is high from just before the launch until the kernel has
        completed."""
        values = []
        for arg in args:
            if isinstance(arg, DeviceBuffer):
                values.append(np.array([device_pointer(arg.ptr)], dtype=np.uint64))
            elif isinstance(arg, ScalarArg):
                values.append(np.array([arg.value], dtype=arg.dtype))
            else:
                raise TypeError(f"unsupported kernel argument: {arg!r}")
        params = np.array([v.ctypes.data for v in values], dtype=np.uint64)
        if gpio is not None:
            gpio.set_value(1)
            try:
                check_cuda(self.cuda, self._launch(function, grid, block, params, shared_mem_bytes), "cuLaunchKernel")
                self.synchronize()
            finally:
                gpio.set_value(0)
        else:
            check_cuda(self.cuda, self._launch(function, grid, block, params, shared_mem_bytes), "cuLaunchKernel")

    def close(self) -> None:
        if self.stream is not None:
            check_cuda(self.cuda, self.cuda.cuStreamDestroy(self.stream), "cuStreamDestroy")
            self.stream = None
        if self.module is not None:
            check_cuda(self.cuda, self.cuda.cuModuleUnload(self.module), "cuModuleUnload")
            self.module = None
        if self.ctx is not None:
            check_cuda(self.cuda, self.cuda.cuDevicePrimaryCtxRelease(self.device), "cuDevicePrimaryCtxRelease")
            self.ctx = None

    def __enter__(self) -> "CudaModule":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
