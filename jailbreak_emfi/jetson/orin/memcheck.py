"""Orin memory-check workload: a large GPU buffer, read and checksummed while the probe glitches.

A warm-up pass stores the expected per-block checksums. Every verification pass (one per trial) repeats
the same read-and-checksum over the whole buffer with the GPIO line high; blocks whose checksum differs
are logged with their addresses, and then an exact scan finds every word that differs from the pattern
(the words the glitch actually changed in memory, as opposed to a transient misread).

Log files (all in memfault.log_dir, tab-separated with a header):
  pass_results.log                every pass;  mismatch.log / no_mismatch.log: the same rows, split
  cuda_mismatch_addresses.tsv     every mismatching block of a failed pass
  data_scan_results.log           one exact-scan summary per failed pass
  cuda_data_mismatch_addresses.tsv every word the exact scan found (up to data_mismatch_capacity)
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

from jetson.common.results import append_tsv
from jetson.orin.cuda_driver import CudaModule, DeviceBuffer, device_pointer, u32, u64

KERNEL_SOURCE = Path(__file__).with_name("memcheck_kernels.cu")
WORD_BYTES = np.dtype(np.uint32).itemsize
U64_BYTES = np.dtype(np.uint64).itemsize
U32_BYTES = np.dtype(np.uint32).itemsize
GIB = 1024 ** 3
DATA_MISMATCH_DTYPE = np.dtype([("word_index", "<u8"), ("expected", "<u4"), ("actual", "<u4")])
# Buffers the workload allocates (the PA capture expects a mapping for each).
BUFFER_LABELS = ("data", "block_sums", "expected_sums", "mismatch_flags", "expected_mismatch_addresses",
                 "actual_mismatch_addresses", "data_scan_count", "data_scan_records")

PASS_RESULT_LOG = "pass_results.log"
MISMATCH_LOG = "mismatch.log"
NO_MISMATCH_LOG = "no_mismatch.log"
MISMATCH_ADDRESS_LOG = "cuda_mismatch_addresses.tsv"
DATA_SCAN_RESULT_LOG = "data_scan_results.log"
DATA_MISMATCH_ADDRESS_LOG = "cuda_data_mismatch_addresses.tsv"

PASS_RESULT_HEADER = ("timestamp\trun_id\tpass\tattempt_index\ttotal_passes\tstatus\tmismatch_count\t"
                      "compute_seconds\tfirst_block\tfirst_expected_checksum\tfirst_actual_checksum\t"
                      "first_actual_address")
MISMATCH_ADDRESS_HEADER = ("timestamp\tpass\tattempt_index\tcompute_seconds\tblock\texpected\tactual\t"
                           "expected_addr\tactual_addr\tdata_first_word\tdata_lane_end_word\tdata_first_addr\t"
                           "data_lane_end_addr\tdata_stride_words")
DATA_SCAN_RESULT_HEADER = ("timestamp\trun_id\tpass\tattempt_index\tcompute_seconds\tchecksum_mismatch_count\t"
                           "status\tdata_mismatch_count\tcaptured_count\tcapacity\tscan_seconds\t"
                           "first_word_index\tfirst_cuda_addr")
DATA_MISMATCH_ADDRESS_HEADER = ("timestamp\trun_id\tpass\tattempt_index\tcompute_seconds\tslot\tword_index\t"
                                "byte_offset\tblock\tthread\titeration\texpected_u32\tactual_u32\txor_u32\t"
                                "cuda_addr\tdata_mismatch_count\tcaptured_count\ttruncated")

PassHook = Callable[[int], int]


@dataclass(frozen=True)
class Workload:
    nbytes: int
    blocks: int
    threads: int
    seed: int
    passes: int
    data_mismatch_capacity: int
    log_dir: Path

    def __post_init__(self):
        if self.nbytes <= 0 or self.nbytes % WORD_BYTES:
            raise ValueError(f"buffer size must be a positive multiple of {WORD_BYTES} bytes")
        if self.blocks <= 0 or self.threads <= 0 or self.threads & (self.threads - 1):
            raise ValueError("blocks must be positive and threads a power of two (for the reduction)")
        if self.passes <= 0 or self.data_mismatch_capacity <= 0:
            raise ValueError("passes and data_mismatch_capacity must be positive")

    @property
    def words(self) -> int:
        return self.nbytes // WORD_BYTES


def timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def na(value: int | None) -> str:
    return "NA" if value is None else str(value)


def hex_or_na(value: int | None) -> str:
    return "NA" if value is None else f"0x{value:016x}"


def format_pass_result(*, timestamp: str, run_id: str, pass_index: int, attempt_index: int | None,
                       total_passes: int, compute_elapsed: float, mismatch_count: int,
                       first_block: int | None = None, first_expected: int | None = None,
                       first_actual: int | None = None, first_actual_address: int | None = None) -> str:
    status = "MISMATCH" if mismatch_count else "NO_MISMATCH"
    return (f"{timestamp}\t{run_id}\t{pass_index}\t{na(attempt_index)}\t{total_passes}\t{status}\t"
            f"{mismatch_count}\t{compute_elapsed:.6f}\t{na(first_block)}\t{hex_or_na(first_expected)}\t"
            f"{hex_or_na(first_actual)}\t{hex_or_na(first_actual_address)}")


def lane_bounds(data_base: int, block: int, blocks: int, threads: int, words: int):
    """First and last word (and address) block's threads read in the first grid-stride iteration."""
    first = block * threads
    stride = blocks * threads
    if first >= words:
        return None, None, None, None, stride
    last = min(first + threads - 1, words - 1)
    return first, last, data_base + first * WORD_BYTES, data_base + last * WORD_BYTES, stride


def mismatch_address_rows(pass_index: int, attempt: int | None, elapsed: float, indices: np.ndarray,
                          expected: np.ndarray, actual: np.ndarray, expected_addr: np.ndarray,
                          actual_addr: np.ndarray, data_base: int, w: Workload) -> Iterable[str]:
    ts = timestamp()
    for raw in indices:
        b = int(raw)
        first, last, first_addr, last_addr, stride = lane_bounds(data_base, b, w.blocks, w.threads, w.words)
        yield (f"{ts}\t{pass_index}\t{na(attempt)}\t{elapsed:.6f}\t{b}\t0x{int(expected[b]):016x}\t"
               f"0x{int(actual[b]):016x}\t0x{int(expected_addr[b]):016x}\t0x{int(actual_addr[b]):016x}\t"
               f"{na(first)}\t{na(last)}\t{hex_or_na(first_addr)}\t{hex_or_na(last_addr)}\t{stride}")


def data_mismatch_rows(ts: str, run_id: str, pass_index: int, attempt: int | None, elapsed: float,
                       records: np.ndarray, data_base: int, w: Workload, total: int) -> Iterable[str]:
    stride = w.blocks * w.threads
    truncated = int(total > records.size)
    for slot, record in enumerate(records):
        word = int(record["word_index"])
        exp, act = int(record["expected"]), int(record["actual"])
        tid = word % stride
        yield (f"{ts}\t{run_id}\t{pass_index}\t{na(attempt)}\t{elapsed:.6f}\t{slot}\t{word}\t"
               f"{word * WORD_BYTES}\t{tid // w.threads}\t{tid % w.threads}\t{word // stride}\t0x{exp:08x}\t"
               f"0x{act:08x}\t0x{exp ^ act:08x}\t0x{data_base + word * WORD_BYTES:016x}\t{total}\t"
               f"{records.size}\t{truncated}")


def scan_status(data_mismatches: int, captured: int) -> str:
    if data_mismatches == 0:
        return "CLEAN"
    return "TRUNCATED" if data_mismatches > captured else "DATA_MISMATCH"


def timed(label: str, cuda: CudaModule, launch: Callable[[], None]) -> float:
    start = time.perf_counter()
    launch()
    cuda.synchronize()
    elapsed = time.perf_counter() - start
    print(f"{label}: {elapsed:.6f} s", flush=True)
    return elapsed


def run(w: Workload, cuda: CudaModule, *, run_id: str, before_pass: PassHook, gpio=None) -> int:
    """Allocate, initialise, warm up, then w.passes verification passes; 0 if no pass mismatched, else 2."""
    words = w.words
    sums_bytes, flags_bytes = w.blocks * U64_BYTES, w.blocks * U32_BYTES
    shared_bytes = w.threads * U64_BYTES
    scan_bytes = w.data_mismatch_capacity * DATA_MISMATCH_DTYPE.itemsize
    logs = {name: w.log_dir / name for name in (PASS_RESULT_LOG, MISMATCH_LOG, NO_MISMATCH_LOG,
                                                 MISMATCH_ADDRESS_LOG, DATA_SCAN_RESULT_LOG,
                                                 DATA_MISMATCH_ADDRESS_LOG)}
    print(f"requested allocation: {w.nbytes} bytes ({w.nbytes / GIB:.3f} GiB); uint32 words: {words}; "
          f"blocks={w.blocks} threads={w.threads} passes={w.passes}; logs in {w.log_dir}; run_id {run_id}",
          flush=True)

    free_before, total = cuda.mem_info()
    print(f"device memory before allocation: free={free_before / GIB:.3f} GiB total={total / GIB:.3f} GiB")
    if w.nbytes + 2 * sums_bytes + flags_bytes + 2 * sums_bytes + U64_BYTES + scan_bytes > free_before:
        print("warning: requested allocation is larger than reported free device memory")
    sizes = (w.nbytes, sums_bytes, sums_bytes, flags_bytes, sums_bytes, sums_bytes, U64_BYTES, scan_bytes)
    buf = {label: cuda.alloc(size, label) for label, size in zip(BUFFER_LABELS, sizes)}
    data: DeviceBuffer = buf["data"]
    print(f"device memory after allocation: free={cuda.mem_info()[0] / GIB:.3f} GiB")

    init_k, compute_k = cuda.function("init_values"), cuda.function("load_compute_reduce")
    compare_k, scan_k = cuda.function("compare_block_sums"), cuda.function("scan_data_values")
    grid, block = (w.blocks, 1, 1), (w.threads, 1, 1)

    def checksum_pass(with_gpio):
        cuda.launch(compute_k, grid, block, [data, buf["block_sums"], u64(words)], shared_bytes, gpio=with_gpio)

    init_s = timed("init_values", cuda, lambda: cuda.launch(init_k, grid, block, [data, u64(words), u32(w.seed)]))
    print(f"init write bandwidth: {w.nbytes / GIB / init_s:.3f} GiB/s")
    warm_s = timed("warmup_expected", cuda, lambda: checksum_pass(None))
    print(f"warmup load bandwidth: {w.nbytes / GIB / warm_s:.3f} GiB/s")
    buf["expected_sums"].copy_from_device(buf["block_sums"], sums_bytes)
    expected = np.zeros(w.blocks, dtype=np.uint64)
    buf["expected_sums"].copy_to_host(expected)
    print(f"expected_checksum_xor: 0x{int(np.bitwise_xor.reduce(expected)):016x}")
    print(f"expected_checksum_sum: 0x{int(np.sum(expected, dtype=np.uint64)):016x}")

    flags = np.zeros(w.blocks, dtype=np.uint32)
    expected_addr = np.zeros(w.blocks, dtype=np.uint64)
    actual_addr = np.zeros(w.blocks, dtype=np.uint64)
    actual = np.zeros(w.blocks, dtype=np.uint64)
    scan_count = np.zeros(1, dtype=np.uint64)
    data_base = device_pointer(data.ptr)
    total_elapsed, total_mismatches = 0.0, 0

    for pass_index in range(1, w.passes + 1):
        attempt = before_pass(pass_index)
        elapsed = timed(f"verification_pass_{pass_index}", cuda, lambda: checksum_pass(gpio))
        print("Compute kernel done.", flush=True)
        total_elapsed += elapsed
        cuda.launch(compare_k, grid, block, [buf["expected_sums"], buf["block_sums"], buf["mismatch_flags"],
                                             buf["expected_mismatch_addresses"], buf["actual_mismatch_addresses"],
                                             u64(w.blocks)])
        cuda.synchronize()
        buf["mismatch_flags"].copy_to_host(flags)
        mismatches = int(np.sum(flags, dtype=np.uint64))
        total_mismatches += mismatches
        first = dict(first_block=None, first_expected=None, first_actual=None, first_actual_address=None)

        if mismatches:
            buf["block_sums"].copy_to_host(actual)
            buf["expected_mismatch_addresses"].copy_to_host(expected_addr)
            buf["actual_mismatch_addresses"].copy_to_host(actual_addr)
            indices = np.flatnonzero(flags)
            b0 = int(indices[0])
            first = dict(first_block=b0, first_expected=int(expected[b0]), first_actual=int(actual[b0]),
                         first_actual_address=int(actual_addr[b0]))
            message = (f"pass {pass_index}: FAIL attempt_index={na(attempt)} mismatches={mismatches} "
                       f"first_block={b0} expected=0x{first['first_expected']:016x} "
                       f"actual=0x{first['first_actual']:016x} actual_addr=0x{first['first_actual_address']:016x}")
            append_tsv(logs[MISMATCH_ADDRESS_LOG], MISMATCH_ADDRESS_HEADER,
                       mismatch_address_rows(pass_index, attempt, elapsed, indices, expected, actual,
                                             expected_addr, actual_addr, data_base, w))

            # Exact scan: which words differ from the pattern (stored corruption vs a misread).
            scan_count.fill(0)
            buf["data_scan_count"].copy_from_host(scan_count)
            scan_s = timed(f"exact_data_scan_pass_{pass_index}", cuda, lambda: cuda.launch(
                scan_k, grid, block, [data, u64(words), u32(w.seed), buf["data_scan_count"],
                                      buf["data_scan_records"], u64(w.data_mismatch_capacity)]))
            buf["data_scan_count"].copy_to_host(scan_count)
            found = int(scan_count[0])
            captured = min(found, w.data_mismatch_capacity)
            records = np.zeros(captured, dtype=DATA_MISMATCH_DTYPE)
            if captured:
                buf["data_scan_records"].copy_to_host(records)
            first_word = int(records[0]["word_index"]) if captured else None
            ts = timestamp()
            if captured:
                append_tsv(logs[DATA_MISMATCH_ADDRESS_LOG], DATA_MISMATCH_ADDRESS_HEADER,
                           data_mismatch_rows(ts, run_id, pass_index, attempt, elapsed, records, data_base, w, found))
            status = scan_status(found, captured)
            append_tsv(logs[DATA_SCAN_RESULT_LOG], DATA_SCAN_RESULT_HEADER, [
                f"{ts}\t{run_id}\t{pass_index}\t{na(attempt)}\t{elapsed:.6f}\t{mismatches}\t{status}\t{found}\t"
                f"{captured}\t{w.data_mismatch_capacity}\t{scan_s:.6f}\t{na(first_word)}\t"
                f"{hex_or_na(None if first_word is None else data_base + first_word * WORD_BYTES)}"])
            message += f" data_scan={status} data_mismatches={found} captured={captured}"
        else:
            message = f"pass {pass_index}: OK attempt_index={na(attempt)}"

        row = format_pass_result(timestamp=timestamp(), run_id=run_id, pass_index=pass_index,
                                 attempt_index=attempt, total_passes=w.passes, compute_elapsed=elapsed,
                                 mismatch_count=mismatches, **first)
        append_tsv(logs[MISMATCH_LOG if mismatches else NO_MISMATCH_LOG], PASS_RESULT_HEADER, [row])
        append_tsv(logs[PASS_RESULT_LOG], PASS_RESULT_HEADER, [row])
        print(message, flush=True)

    if total_elapsed:
        print(f"verification load bandwidth: {w.nbytes * w.passes / GIB / total_elapsed:.3f} GiB/s")
    print(f"total_mismatches: {total_mismatches}")
    for buffer in reversed(list(buf.values())):
        buffer.free()
    return 0 if total_mismatches == 0 else 2
