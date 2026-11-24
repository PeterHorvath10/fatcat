# gpu_memfault: GPU memory-check workload with true physical addresses (Thor)

A CUDA kernel reads and verifies a buffer over and over while the EM probe glitches, and logs every
corrupted word with its **true** DRAM physical address and DRAM channel/bank.

## How it gets the physical address
The buffer is `cuMemHostAlloc(DEVICEMAP)`: pinned host memory that the GPU accesses through a device
pointer and whose pages `/proc/self/pagemap` resolves to the real DRAM PA (0x80000000 + 0..128 GiB). The
DRAM channel and bank are XOR hashes of PA - 0x80000000 with the memory controller's channel/bank mask
registers (constants in the source). Device memory (`cudaMalloc`) only exposes a 0xff.. GPU aperture
address, which is not a PA; `--mem device` therefore logs the GPU VA and NA/-1 for pa/channel/bank.

- `--pattern index`: word i = (i * 0x9E3779B97F4A7C15) ^ 0x55..55 instead of 0x55..55 everywhere, so a
  read served from another word of the buffer decodes to that word (`src_word`, `src_pa`, `src_channel`;
  `src_word` -1 = not from the buffer).
- Each corrupted word is logged once per window and repaired.
- ECC polling (default; `memfault.ecc_polling`): each window snapshots the GPU's ECC status/counter
  registers (L2 slices, SM memories, L1.5/TLBs, falcons) and error-capture registers through BAR0 before
  and after, and writes every unit that changed to `<log>_ecc.tsv`; `<log>_win.tsv` gets one row per
  window (kernel launches, samples of the graphics-engine busy registers, GPU timer at start/end).
  Registers reading 0xbadf... (power-gated unit) are ignored.

## Build (on Thor)
```sh
make ARCH=<memfault.cuda_arch> NVCC=<memfault.nvcc>
```

## Run
Root is required (pagemap PFNs, BAR0). The service `gpu-memfault.service` (rendered by
`jetson/install.py`) runs `python3 -m jetson.thor.gpu_memfault.launch`, which starts the binary with the
`[memfault]` settings of `config/thor.toml`. The vLLM workers use the same TCP port and the vLLM server
holds memory, so stop them first (see docs/jetson.md). The worker is ready at "waiting for controller...";
it accepts a new controller connection after the previous one closes.

By hand: `sudo ./gpu_memfault --host H --port P --log PATH --gib G --window S [--pattern index] [--mem device]`;
`--free-run GAP_S` runs one verify window every GAP_S seconds without a controller (for setting up the
scope trigger).

Caution: freeing a buffer of tens of GiB takes over a minute; if systemd kills the process meanwhile, the
GPU driver can stay broken (CUDA_ERROR_NO_DEVICE) until a reboot, hence the long TimeoutStopSec.

## Log format
`<log>`: timestamp, attempt_index, collection_id, word_index, va, pa, channel, bank, expected, actual,
xor (+ src_word, src_pa, src_channel with `--pattern index`). The file grows across scans; select a scan
by its collection_id. A board reset can tear the last row.
