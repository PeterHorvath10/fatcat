"""Record the physical pages behind every CUDA allocation of the Orin memory check.

Right after each allocation, the orin_gpu_pa query tool (orin_gpu_pa/, needs the kernel module loaded by
orin-gpu-pa.service) walks the GPU page table of our process and prints one row per 4 KiB page: GPU VA,
GMMU page size, physical address, mapping, DRAM channel and package. The rows go gzip'ed into
<map_dir>/<boot id>/<time>_pid<pid>/, summarised in manifest.json. Before each trial the handshake
(collection id, attempt index) is appended to attempt_handshakes.tsv there and to a global join log, so
every logged mismatch can be traced back to the physical pages of the exact process that saw it.
See orin_gpu_pa/HOW_IT_WORKS.md.
"""

from __future__ import annotations

import gzip
import hashlib
import os
import re
import socket
import struct
import subprocess
import time
from collections import Counter
from pathlib import Path

from jetson.common.results import append_tsv, fsync_directory, write_json_atomic
from jetson.orin.cuda_driver import DeviceBuffer, device_pointer

PAGE_SIZE = 4096
PA_DEVICE = Path("/dev/orin_gpu_pa")
MANIFEST_SCHEMA = 2
CHANNELS = 8                 # Orin Nano DRAM channels
PACKAGES = 2
HASH_BLOCK = 1024 * 1024
JOIN_HEADER = ("schema\ttimestamp\tcollection_id\trun_id\tpass\tattempt_index\tboot_id\tpid\t"
               "process_start_ticks\tcapture_dir\tmanifest\tstatus")
QUERY_ROW = re.compile(rb"gpu_va=(0x[0-9a-f]+).*?page=([0-9]+)K.*?pa=(0x[0-9a-f]+)"
                       rb".*?mapping=(0x[0-9a-f]+)\+(0x[0-9a-f]+).*?channel=([0-9]+)"
                       rb".*?candidate_package=([0-9]+)")


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def process_start_ticks() -> int:
    return int(Path(f"/proc/{os.getpid()}/stat").read_text(encoding="utf-8").split()[21])


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(HASH_BLOCK), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_label(label: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", label)


class AllocationMappingCapture:
    def __init__(self, query_tool: Path, map_root: Path, join_log: Path, expected_allocations: int, run_id: str):
        if not query_tool.is_file() or not os.access(query_tool, os.X_OK):
            raise RuntimeError(f"CUDA mapping query tool is unavailable: {query_tool}")
        if not PA_DEVICE.exists():
            raise RuntimeError(f"{PA_DEVICE} is absent; is orin-gpu-pa.service running?")
        self.query_tool, self.join_log = query_tool, join_log
        self.boot_id = Path("/proc/sys/kernel/random/boot_id").read_text(encoding="utf-8").strip()
        self.pid, self.process_ticks, self.run_id = os.getpid(), process_start_ticks(), run_id
        self.run_dir = map_root / self.boot_id / f"{time.strftime('%Y%m%dT%H%M%S%z')}_pid{self.pid}"
        self.run_dir.mkdir(parents=True, mode=0o700, exist_ok=False)
        self.manifest_path = self.run_dir / "manifest.json"
        self.handshake_path = self.run_dir / "attempt_handshakes.tsv"
        self.capture_dir = str(Path(map_root.name) / self.run_dir.relative_to(map_root))
        self.expected_allocations = expected_allocations
        self._recorded_passes: set[int] = set()
        self.document: dict = {
            "schema": MANIFEST_SCHEMA, "status": "capturing", "host": socket.gethostname(),
            "boot_id": self.boot_id, "pid": self.pid, "process_start_ticks": self.process_ticks,
            "run_id": run_id, "capture_dir": self.capture_dir, "attempt_handshake_log": self.handshake_path.name,
            "handshake_count": 0, "attempt_indices_seen": [], "started_at": now(), "query_tool": str(query_tool),
            "sampling_page_bytes": PAGE_SIZE, "expected_explicit_allocations": expected_allocations,
            "allocations": [],
        }
        self._write_manifest()

    def _write_manifest(self) -> None:
        write_json_atomic(self.manifest_path, self.document)

    def record_handshake(self, pass_index: int, attempt_index: int, collection_id: str) -> None:
        """Persist the run/capture/attempt join before the glitched pass starts."""
        if pass_index <= 0 or attempt_index < 0 or not re.fullmatch(r"[0-9a-f]{32}", collection_id):
            raise ValueError(f"bad handshake: pass {pass_index}, attempt {attempt_index}, id {collection_id}")
        if pass_index in self._recorded_passes:
            raise ValueError(f"pass {pass_index} handshake was already recorded")
        previous = self.document.get("collection_id")
        if previous not in (None, collection_id):
            raise ValueError(f"run collection changed from {previous} to {collection_id}")
        self.document["collection_id"] = collection_id
        ts = now()
        record = (f"{MANIFEST_SCHEMA}\t{ts}\t{collection_id}\t{self.run_id}\t{pass_index}\t{attempt_index}\t"
                  f"{self.boot_id}\t{self.pid}\t{self.process_ticks}\t{self.capture_dir}\t"
                  f"{self.capture_dir}/manifest.json\tHANDSHAKE_RECEIVED")
        # The capture-local copy travels with its mapping; the global copy allows a one-file join.
        append_tsv(self.handshake_path, JOIN_HEADER, [record])
        append_tsv(self.join_log, JOIN_HEADER, [record])
        handshake = {"timestamp": ts, "collection_id": collection_id, "pass": pass_index,
                     "attempt_index": attempt_index}
        if self.document["handshake_count"] == 0:
            self.document["first_handshake"] = handshake
        self.document["latest_handshake"] = handshake
        self.document["handshake_count"] += 1
        self.document["attempt_indices_seen"] = sorted({*self.document["attempt_indices_seen"], attempt_index})
        self._write_manifest()
        self._recorded_passes.add(pass_index)

    def __call__(self, buffer: DeviceBuffer) -> None:
        """CudaModule on_alloc hook."""
        try:
            self._capture(buffer)
        except Exception as exc:
            self.document["status"] = "capture_failed"
            self.document["capture_error"] = str(exc)
            self._write_manifest()
            raise

    def _capture(self, buffer: DeviceBuffer) -> None:
        index = len(self.document["allocations"])
        gpu_va, nbytes = device_pointer(buffer.ptr), int(buffer.nbytes)
        expected_pages = (nbytes + PAGE_SIZE - 1) // PAGE_SIZE
        stem = f"{index:02d}_{safe_label(buffer.label)}"
        final_path = self.run_dir / f"{stem}_gpu_pa_4k.txt.gz"
        temporary = self.run_dir / f".{final_path.name}.tmp"
        stderr_path = self.run_dir / f"{stem}_query.stderr"
        command = [str(self.query_tool), "scan", str(self.pid), hex(gpu_va), hex(nbytes), hex(PAGE_SIZE)]
        print(f"capturing CUDA mapping: label={buffer.label} gpu_va=0x{gpu_va:x} bytes={nbytes} "
              f"pages={expected_pages}", flush=True)

        digest = hashlib.sha256()
        pages, extents, pa_min, pa_max, previous = 0, 0, None, 0, None
        intervals: set[tuple[int, int]] = set()
        page_kib: set[int] = set()
        channels: Counter[int] = Counter()
        packages: Counter[int] = Counter()
        with stderr_path.open("wb") as errors:
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=errors)
            try:
                with gzip.open(temporary, "wb", compresslevel=6) as out:
                    for line in process.stdout:
                        match = QUERY_ROW.search(line)
                        if match is None:
                            raise RuntimeError(f"unrecognized mapping row for {buffer.label}: "
                                               f"{line.decode(errors='replace').rstrip()}")
                        va, kib, pa, map_start, map_size, channel, package = match.groups()
                        if int(va, 16) != gpu_va + pages * PAGE_SIZE:
                            raise RuntimeError(f"mapping VA sequence mismatch for {buffer.label}: expected "
                                               f"0x{gpu_va + pages * PAGE_SIZE:x}, got {va.decode()}")
                        out.write(line)
                        pa = int(pa, 16)
                        digest.update(struct.pack("<Q", pa))
                        if previous is None or pa != previous + PAGE_SIZE:
                            extents += 1
                        previous = pa
                        pa_min = pa if pa_min is None else min(pa_min, pa)
                        pa_max = max(pa_max, pa + PAGE_SIZE)
                        intervals.add((int(map_start, 16), int(map_size, 16)))
                        page_kib.add(int(kib))
                        channels[int(channel)] += 1
                        packages[int(package)] += 1
                        pages += 1
            finally:
                process.stdout.close()
            status = process.wait()

        stderr_text = stderr_path.read_text(encoding="utf-8", errors="replace")
        if not stderr_text:
            stderr_path.unlink()
        if status != 0:
            raise RuntimeError(f"mapping query failed for {buffer.label} with status {status}: {stderr_text.strip()}")
        if pages != expected_pages:
            raise RuntimeError(f"mapping query returned {pages} pages for {buffer.label}; expected "
                               f"{expected_pages}: {stderr_text.strip()}")
        os.replace(temporary, final_path)
        with final_path.open("rb") as handle:
            os.fsync(handle.fileno())
        fsync_directory(self.run_dir)
        self.document["allocations"].append({
            "index": index, "label": buffer.label, "gpu_va_start": hex(gpu_va),
            "gpu_va_end_exclusive": hex(gpu_va + nbytes), "bytes": nbytes, "captured_pages": pages,
            "mapping_intervals": [{"start": hex(s), "size": hex(n)} for s, n in sorted(intervals)],
            "gmmu_page_kib": sorted(page_kib), "pa_min": hex(pa_min or 0), "pa_max_exclusive": hex(pa_max),
            "physical_extent_count": extents,
            "channel_page_samples": {str(c): channels[c] for c in range(CHANNELS)},
            "candidate_package_page_samples": {str(p): packages[p] for p in range(PACKAGES)},
            "pa_sequence_sha256": digest.hexdigest(), "mapping_dump": final_path.name,
            "mapping_dump_sha256": sha256_file(final_path), "captured_at": now(),
        })
        if len(self.document["allocations"]) >= self.expected_allocations:
            self.document["status"] = "mappings_complete"
            self.document["mappings_completed_at"] = now()
        self._write_manifest()
        print(f"captured CUDA mapping: label={buffer.label} file={final_path} extents={extents}", flush=True)
