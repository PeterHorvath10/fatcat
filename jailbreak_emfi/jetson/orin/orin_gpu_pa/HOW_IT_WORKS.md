# How the Orin CUDA virtual-to-physical mapper works

The Orin mapping stack translates a CUDA device pointer through the GPU's own
page tables. It does not use Linux `/proc/PID/pagemap`, because a CUDA device
pointer is a GPU virtual address rather than a CPU virtual address.

```text
CUDA GPU VA + process PID
        |
        v
process nvgpu address space
        |
        v
GA10B GMMU page-table entry
        |
        v
direct physical address or SMMU IOVA
        |
        v
system physical address
        |
        v
memory-controller channel and candidate DRAM package
```

## Components

| Component | Purpose |
|---|---|
| `orin_gpu_pa.ko` | Kernel-side, read-only nvgpu page-table query |
| `orin_gpu_pa_query` | Userspace ioctl client for one address or a range |
| `orin-gpu-pa.service` | Loads the module through the privileged loader |
| `load-orin-gpu-pa` | Resolves private nvgpu symbols and manages the module |
| `memfault.py` | Captures every page belonging to its CUDA allocations |
| `analysis/dram/dram_map.py` | Interprets a system PA as channel/package/bank candidates |

## 1. Loading the kernel probe

The repository loader is
`jetson/orin/bin/load-orin-gpu-pa`. On the Jetson it is installed as
`/usr/local/sbin/load-orin-gpu-pa` (see docs/jetson.md) and invoked by
`orin-gpu-pa.service`.

The loader first verifies that `<memfault.pa_module>` has a
kernel `vermagic` matching `uname -r`. It then resolves the current addresses
of three private nvgpu symbols from root-only `/proc/kallsyms`:

- `nvgpu_get_pte`
- `gk20a_as_ops`
- `nvgpu_vm_find_mapped_buf_range`

The stock R36.5 nvgpu module does not export these symbols. Their live
addresses are therefore passed to `orin_gpu_pa.ko` as module parameters during
`insmod`. They must be resolved again after every reboot because kernel address
randomization can move them.

After the module loads, it registers the misc character device
`/dev/orin_gpu_pa`. The loader assigns it to the `video` group with mode `0660`,
allowing ordinary queries by the configured Jetson user while keeping module
loading privileged.

The module is read-only: it does not modify GPU page tables, CUDA allocations,
the target process, or memory contents.

## 2. Sending a query

The userspace query tool opens `/dev/orin_gpu_pa` and sends an
`ORIN_GPU_PA_IOC_QUERY` ioctl containing:

- the PID owning the CUDA allocation;
- the CUDA GPU virtual address to translate.

For example:

```bash
<memfault.pa_query_tool> 3639 0x2040ae000
```

The returned structure contains the raw PTE, detected page size, GMMU address,
optional IOVA, system physical address, nvgpu mapping interval, channel, and
candidate package.

The query tool also supports range scanning:

```bash
<memfault.pa_query_tool> \
  scan PID START_GPU_VA LENGTH STEP
```

Its default scan step is 2 MiB. `memfault.py` explicitly supplies a 4 KiB step
to capture every page.

## 3. Finding the process GPU address space

Inside the ioctl handler, the kernel module finds the target process and
iterates its open file descriptors. It identifies the nvgpu address-space file
by comparing each file's operations pointer with the live `gk20a_as_ops`
address supplied when the module was loaded.

The file's private data begins with the verified R36.5
`gk20a_as_share` fields. These expose the nvgpu GPU object and VM required by
`nvgpu_get_pte()`.

The module then:

1. calls `nvgpu_get_pte(g, vm, gpu_va, pte)`;
2. checks the PTE valid bit;
3. calls `nvgpu_vm_find_mapped_buf_range(vm, gpu_va)`;
4. verifies that the requested VA is within the returned live allocation;
5. records its mapping start, size, and nvgpu page-size index.

The mapping-range check prevents a successful PTE lookup from being accepted
without confirming ownership by a live nvgpu allocation.

## 4. Decoding the GA10B GMMU PTE

For the GA10B GPU, the GMMU page base is reconstructed from the two 32-bit PTE
words as follows:

```text
gmmu_base =
    (((pte[0] >> 8) & 0x00ffffff) << 12)
  | ((pte[1] & 0x0000000f) << 36)
```

This uses PTE word 0 bits 31:8 for address bits 35:12 and PTE word 1 bits 3:0
for address bits 39:36. Other bits in PTE word 1 contain memory-kind and
compression-tag metadata.

GA10B user mappings use 4 KiB or 64 KiB pages. The module probes adjacent
4 KiB virtual addresses: if the neighboring address returns the same PTE, the
mapping is treated as a 64 KiB big page; otherwise it is treated as a 4 KiB
page.

The offset within the detected page is:

```text
page_offset = gpu_va & ((1 << page_shift) - 1)
```

## 5. Converting the GMMU address to a system PA

GMMU address bit 37 indicates an IOMMU mapping on this target.

If bit 37 is set, the module clears the marker, adds the page offset, and asks
the Tegra SMMU for the physical translation:

```text
iova = (gmmu_base with bit 37 cleared) + page_offset
physical_address = iommu_iova_to_phys(gpu_domain, iova)
```

If bit 37 is not set, the GMMU entry already contains a direct system physical
address:

```text
physical_address = gmmu_base + page_offset
```

The result is a system physical byte address, not merely a physical page
number. Different offsets within a page therefore produce different physical
byte addresses.

## 6. Capturing an entire CUDA allocation

`memfault.py` wraps each explicit `CudaModule.alloc` call. Immediately after an
allocation is returned, and while it remains live, the wrapper launches:

```text
orin_gpu_pa_query scan PID CUDA_VA ALLOCATION_SIZE 4096
```

For a 4 GiB allocation this produces 1,048,576 translated rows. During the
scan, `memfault.py` verifies that:

- virtual addresses are returned in exact 4 KiB order;
- the row count equals the expected allocation page count;
- each output row follows the expected schema;
- the query process exits successfully.

It also calculates physical extents, minimum and maximum PAs, page counts per
channel and candidate package, a hash of the ordered PA sequence, and a
SHA-256 hash of the compressed raw mapping. Results are written beneath:

```text
<memfault.map_dir>/BOOT_ID/TIMESTAMP_pidPID/
```

The run's `manifest.json` records allocation identity, mapping intervals,
page-size observations, hashes, and capture-completion state.

## 6.1 Directly joining a new collection

New acquisitions do not depend on wall-clock order or syslog reconstruction.
The local XYZ controller creates one random `collection_id` UUID when a scan
starts. Every trial row in `trial_parameters.csv` contains that ID. For each
TCP collect acknowledgement, the controller sends:

```text
4-byte collect token | uint64 attempt_index | 16-byte collection UUID
```

Before launching the corresponding CUDA compute pass, `memfault.py` durably
writes the identity to both:

```text
<memfault.log_dir>/run_capture_join.tsv
<memfault.map_dir>/BOOT_ID/TIMESTAMP_pidPID/attempt_handshakes.tsv
```

Each join row contains `collection_id`, `run_id`, pass, attempt index, boot ID,
PID, process-start ticks, capture directory, and manifest path. The same
`run_id`, capture path, and first/latest handshake metadata are stored in
`manifest.json`. Runtime mismatch/pass/scan rows already contain `run_id`, pass,
and attempt index. The exact join chain is therefore:

```text
trial_parameters.csv
  collection_id + attempt_index
             |
             v
<memfault.log_dir>/run_capture_join.tsv
  collection_id + attempt_index + run_id + capture_dir
             |
             +------ run_id + pass + attempt_index ------+
             |                                            |
             v                                            v
manifest.json and mapping files                 mismatch/pass/scan logs
```

The acquisition identity is the `collection_id + attempt_index`, not the boot
folder count. Pass rows with the same attempt index are repeated trials at one
location; they are not additional cells. For a configured 15-column row-major
scan the logical cell position remains `x = attempt_index % 15` and
`y = attempt_index // 15`, while the controller's recorded XYZ coordinates are
authoritative for arbitrary line or nonuniform scans.

## Mapping files and mismatch lookup

Each memfault startup writes one directory beneath the boot ID:

```text
<memfault.map_dir>/BOOT_ID/TIMESTAMP_pidPID/
```

For example, the capture inspected on 2026-08-11 was:

```text
<memfault.map_dir>/BOOT_ID/
└── 20260811T133947+0200_pid865/
```

The exact path changes after a new boot or memfault process. Always select the
capture whose boot ID, PID, process start time, and allocation lifetime cover
the mismatch event being analyzed.

### Primary data-allocation mapping

`00_data_gpu_pa_4k.txt.gz` is the complete mapping for the 4 GiB allocation
being tested for corruption. It contains 1,048,576 rows, one for every 4 KiB
virtual page. Each row includes the CUDA VA, raw GMMU PTE, detected page size,
GMMU address, optional IOVA, exact system PA, nvgpu mapping interval, channel,
and candidate package. For example:

```text
gpu_va=0x00000002040ae000 page=4K pte=06000000:12f78007 aperture=0x6 gmmu=0x012f780000 iova=0x0000000000 pa=0x012f780000 mapping=0x2040ae000+0x100000000 map_pgsz_idx=0 channel=5 candidate_package=0
gpu_va=0x00000002040af000 page=4K pte=06000000:12f78107 aperture=0x6 gmmu=0x012f781000 iova=0x0000000000 pa=0x012f781000 mapping=0x2040ae000+0x100000000 map_pgsz_idx=0 channel=4 candidate_package=0
```

The inspected allocation covered:

```text
CUDA VA:     0x2040ae000–0x3040ae000
PA envelope: 0x12f780000–0x22f870000
Pages:       1,048,576
```

The PA envelope is only the minimum through maximum observed address. It does
not imply that every physical byte inside that interval belongs to the
allocation; the complete page-by-page mapping file is authoritative.

### Auxiliary allocation mappings

The remaining files describe the explicit CUDA buffers used by the checksum
and exact-data verification machinery:

```text
01_block_sums_gpu_pa_4k.txt.gz
02_expected_sums_gpu_pa_4k.txt.gz
03_mismatch_flags_gpu_pa_4k.txt.gz
04_expected_mismatch_addresses_gpu_pa_4k.txt.gz
05_actual_mismatch_addresses_gpu_pa_4k.txt.gz
06_data_scan_count_gpu_pa_4k.txt.gz
07_data_scan_records_gpu_pa_4k.txt.gz
```

These files use the same row format as the primary data map. They are normally
much smaller because the auxiliary allocations contain summaries, flags,
address buffers, counters, and bounded mismatch records rather than the 4 GiB
test array.

### Manifest

`manifest.json` is the capture index. For every allocation it records:

- allocation label and index;
- CUDA VA start and end;
- corresponding mapping filename;
- expected and captured page counts;
- nvgpu mapping intervals and observed GMMU page sizes;
- minimum and maximum physical addresses;
- physical-extent count;
- page samples per channel and candidate package;
- ordered-PA-sequence and compressed-file SHA-256 hashes.

The manifest summarizes and validates a capture, but it does not replace the
raw `.txt.gz` files when an exact VA-to-PA lookup is required.

### Joining a mismatch to its physical address

The files under `<memfault.log_dir>/` contain pass results, mismatch
events, corrupted values, word indices, and CUDA addresses. They do not contain
the complete VA-to-PA mapping.

To obtain the physical address of a corrupted value:

1. identify the mapping capture through the direct `run_id` join (or, for old
   captures only, from the same boot and memfault process);
2. read the mismatch's CUDA address from the appropriate log;
3. find the 4 KiB mapping row containing that CUDA address in
   `00_data_gpu_pa_4k.txt.gz`;
4. add the mismatch address's offset within that 4 KiB page to the row's PA.

In formula form:

```text
page_va = cuda_address & ~0xfff
offset  = cuda_address & 0xfff
physical_address = mapped_pa_for(page_va) + offset
```

Do not join a mismatch to a mapping solely because its CUDA VA range looks the
same. CUDA virtual ranges are frequently reused, while their physical backing
can change after a process restart or reboot.

To inspect the primary mapping directly on the Jetson:

```bash
gzip -cd <memfault.map_dir>/BOOT_ID/TIMESTAMP_pidPID/00_data_gpu_pa_4k.txt.gz | less
```

## 7. From system PA to DRAM routing fields

VA-to-PA translation and DRAM decoding are separate operations. Once a system
PA is known, `analysis/dram/dram_map.py` applies the P3767 R36.5 memory-BCT
channel masks:

```text
channel bit N = parity(physical_address & CHANNEL_MASK[N])
```

The three channel bits select one of eight x16 memory-controller channels.
EM fault localisation places channels 0, 1, 4, 5 on one package and
2, 3, 6, 7 on the other, so channel bit 1 selects the package:

```text
candidate_package = (channel >> 1) & 1
```

Four additional BCT masks produce a candidate bank address using the same
parity construction.

Channel decoding is grounded in the Orin TRM and memory BCT. Package grouping
and bank decoding remain explicitly labeled candidates because the public
documentation does not provide the complete P3767 package wiring and
row/bank/column decoder.

## Limitations and safety boundaries

- The target process and CUDA allocation must remain alive during a query.
- A new process, model reload, allocator decision, or reboot can produce a new
  physical mapping for the same virtual range.
- The module depends on private R36.5 nvgpu ABI layouts and must be rebuilt and
  revalidated for another BSP or kernel.
- Kernel symbols are resolved on every module load and must never be reused
  blindly across boots.
- `/proc/PID/pagemap` works only for CPU virtual addresses; it cannot translate
  `cudaMalloc` device pointers.
- System PAs are exact for the captured live mapping. Package, bank, and probe
  position interpretations have separate empirical or documentation limits.
- Exposing physical addresses is privileged even though the probe itself is
  read-only.

## Source locations

- Kernel module: `jetson/orin/orin_gpu_pa/orin_gpu_pa.c`
- Shared ioctl structure: `jetson/orin/orin_gpu_pa/orin_gpu_pa_uapi.h`
- Query client: `jetson/orin/orin_gpu_pa/orin_gpu_pa_query.c`
- Loader: `jetson/orin/bin/load-orin-gpu-pa`
- Mapping capture: `jetson/orin/pa_capture.py` (used by `jetson/orin/memfault.py`)
- DRAM decoder: `analysis/dram/dram_map.py` (analysis side, not part of jetson/)
