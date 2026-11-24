# Jetson Orin Nano CUDA virtual-address to physical-address probe

This directory contains a read-only diagnostic for the Jetson Orin Nano
P3767-0005 running Jetson Linux R36.5. It maps a CUDA virtual address owned by
a live process through:

```text
process nvgpu address space
  -> GA10B GMMU PTE
  -> direct system PA on this target
     (or GPU IOVA -> Tegra SMMU if an IOMMU domain is active)
  -> system physical address
  -> P3767 memory-controller channel hash
  -> candidate one of two 4 GiB DRAM packages
```

For a step-by-step description of symbol resolution, nvgpu address-space
lookup, PTE decoding, optional SMMU translation, full-allocation capture, and
DRAM routing interpretation, see [`HOW_IT_WORKS.md`](./HOW_IT_WORKS.md).

The module does not alter the target process, page tables, CUDA allocations,
or memory contents. It must be loaded as root because exposing system physical
addresses is privileged. It is intentionally not installed at boot.


## Build

Build directly on the target so the module matches its running kernel:

```bash
cd <this folder>
make
```

## Load

The stock `nvgpu.ko` keeps its PTE lookup and address-space file operations
private. The loader resolves their randomized runtime addresses from root-only
`/proc/kallsyms` and passes them to the temporary module:

```bash
cd <this folder>
sudo sh -c '
set -eu
get_pte=$(awk '\''$3 == "nvgpu_get_pte" { print "0x" $1; exit }'\'' /proc/kallsyms)
as_ops=$(awk '\''$3 == "gk20a_as_ops" { print "0x" $1; exit }'\'' /proc/kallsyms)
find_mapped=$(awk '\''$3 == "nvgpu_vm_find_mapped_buf_range" { print "0x" $1; exit }'\'' /proc/kallsyms)
test -n "$get_pte"
test -n "$as_ops"
test -n "$find_mapped"
insmod ./orin_gpu_pa.ko get_pte_addr="$get_pte" as_ops_addr="$as_ops" find_mapped_addr="$find_mapped"
chgrp video /dev/orin_gpu_pa
chmod 0660 /dev/orin_gpu_pa
'
```

After loading, ordinary queries by a user in the `video` group do not need sudo.

## Query

```bash
# Translate one CUDA VA belonging to PID 1602.
./orin_gpu_pa_query 1602 0x2040ae000

# Find mapped regions by sampling an 8 GiB CUDA VA interval every 2 MiB.
./orin_gpu_pa_query scan 1602 0x200000000 0x200000000
```

Successful output includes the nvgpu mapping start/size, raw PTE, GMMU
address, optional IOVA, system PA, controller channel 0-7, and candidate
package 0-1. `No such file or directory`
for a sampled address means that address is not mapped in the target GPU VM.

## Remove

Unload the diagnostic when mapping work is complete:

```bash
sudo rmmod orin_gpu_pa
```

## Hardware constants

- GA10B GMMU small and big pages: 4 KiB and 64 KiB.
- GA10B IOMMU marker: GMMU address bit 37.
- P3767-0005 nominal DRAM aperture: `0x80000000` through `0x27fffffff`.
- Enabled memory-controller channels: 8 x16 channels.
- Physical packages: 2 x 4 GiB; candidate grouping is channels 0, 1, 4, 5 and
  2, 3, 6, 7 (channel bit 1 selects the package).
- BCT channel masks: `0xb94f5200`, `0xcae4a400`, `0x2e764800`.

The address-to-channel calculation is the conventional XOR/parity
interpretation of NVIDIA's `McEmemAdrCfgChannelMask0..2` values. It passes the
expected consistency property: every aligned 4 KiB physical span visits all
eight channels in 512-byte steps. The package grouping comes from EM fault
localisation and remains labeled candidate without the P3767 schematic/net
list.
