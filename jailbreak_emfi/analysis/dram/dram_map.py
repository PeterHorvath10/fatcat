"""Physical address -> DRAM channel (and rank, bank, package) on the Jetson Orin Nano and the Jetson AGX Thor
(paper appendix "Address-to-channel mapping", tab:channel-masks).

Both memory controllers select the channel with bit-selector masks: channel bit n = parity(a & M_n), where
a = PA on the Orin Nano and a = PA - 0x80000000 (the offset into DRAM) on the Thor. The masks are hardware facts of
the two boards (read from the live MC registers; equal to the McEmemAdrCfg* entries of the boards' SDRAM BCTs);
they are not settings.

Thor (P3834-0008, 128 GiB LPDDR5X, 8 x32 packages, 16 x16 channels, 2 ranks), MC broadcast aperture 0x81_0802_0000:
  +0x8640 McEmemCfg = 0x20000 MiB, +0x8644 McEmemAdrCfg = 1 (2 ranks), +0x8650/58/60/68 ChannelMask0..3 (low words),
  +0x8654/5c/64/6c their high bytes, +0x8678..0x8684 BankMask0..3.
  rank = off bit 36; bank bit j = parity((off >> 13) & BANK_MASK_j); package = channel >> 1 (sub-channel A/B =
  channel bit 0). 512 B interleave: every aligned 8 KiB holds all 16 channels, a 4 KiB page 8 of them.
  Verified three ways: the model predicts exactly the 1361 row-buffer conflicts among 500,000 timed
  address pairs (no false positives or misses); CPU reads confined to one decoded channel cap at 16.0-16.4 GB/s
  (one x16 LPDDR5X-8533 channel) and two channels give twice that; channels 2k and 2k+1 have identical idle
  latency from every CPU core (one package). Applying the masks to the raw PA is wrong for
  every address.
Orin Nano (P3767-0005, 8 GiB LPDDR5, 2 x64 packages, 8 x16 channels, rank 0 only), MC broadcast 0x2c10000:
  +0x060 CHANNEL_MASK0, +0xdfc CHANNEL_MASK1, +0xdf4 CHANNEL_MASK2, +0x064/068/06c/078 BANK_MASK0..3.
  The channel masks apply to the PA itself. The bank masks act on the channel-local address (the three 512-byte
  channel-stripe bits removed): bank bit n = parity(PA & (BANK_MASK_n << 3)) (fits 749/749 measured row conflicts,
  1 false positive; the unshifted masks fit only 59/749). EM fault localisation places channels 0,1,4,5 on one
  package and 2,3,6,7 on the other: channel bit 1 selects the package, bits 0 and 2 the x16 die.

usage: PYTHONPATH=<jailbreak_emfi> python -m analysis.dram.dram_map {thor|orin} pa ADDR [ADDR ...]
       ... {thor|orin} range START LENGTH [--step 512]      (sizes take K/M/G suffixes)
       ... {thor|orin} pagemap --pid PID VA [VA ...]        (on the Jetson, as root: CPU VAs only)
       ... {thor|orin} annotate LOG.tsv [-o OUT.tsv]        (recompute channel/bank/rank/package from a 'pa' column)
       ... {thor|orin} masks | self-test
A cudaMalloc device pointer has no CPU PTE, so pagemap cannot translate it; the GPU memory-check workloads use
host-mapped pinned memory (cuMemHostAlloc DEVICEMAP) for that reason.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import struct
import sys
from dataclasses import asdict, dataclass
from pathlib import Path


def parity(value: int) -> int:
    return bin(value).count("1") & 1


@dataclass(frozen=True)
class Platform:
    name: str
    dram_base: int
    dram_size: int
    channel_masks: tuple[int, ...]
    channel_on_offset: bool     # masks apply to PA - dram_base (Thor) or to the PA (Orin)
    bank_masks: tuple[int, ...]
    bank_shift_right: int       # Thor: bank masks act on off >> 13
    bank_mask_shift_left: int   # Orin: bank masks act on PA & (mask << 3)
    rank_bit: int | None        # None: single rank

    def channel(self, pa: int) -> int:
        a = pa - self.dram_base if self.channel_on_offset else pa
        return sum(parity(a & m) << i for i, m in enumerate(self.channel_masks))

    def bank(self, pa: int) -> int:
        a = (pa - self.dram_base) >> self.bank_shift_right if self.channel_on_offset else pa
        return sum(parity(a & (m << self.bank_mask_shift_left)) << i for i, m in enumerate(self.bank_masks))

    def rank(self, pa: int) -> int:
        return 0 if self.rank_bit is None else ((pa - self.dram_base) >> self.rank_bit) & 1

    def package(self, channel: int) -> int:
        # Thor: package k = channels 2k, 2k+1. Orin: channel bit 1 selects the package.
        return channel >> 1 if self.name == "thor" else (channel >> 1) & 1

    def in_dram(self, pa: int) -> bool:
        return self.dram_base <= pa < self.dram_base + self.dram_size


THOR = Platform("thor", 0x8000_0000, 128 << 30,
                channel_masks=(0x68B62E3800, 0x4B5A527E00, 0x7FA9A6CC00, 0xE694CD5A00), channel_on_offset=True,
                bank_masks=(0x023331C9, 0x04BC922D, 0x057BD5BE, 0x07AD6AD5), bank_shift_right=13,
                bank_mask_shift_left=0, rank_bit=36)
ORIN_NANO = Platform("orin", 0x8000_0000, 8 << 30,
                     channel_masks=(0xB94F5200, 0xCAE4A400, 0x2E764800), channel_on_offset=False,
                     bank_masks=(0xF26B1000, 0xD19E6000, 0x5FADA400, 0xB57AF800), bank_shift_right=0,
                     bank_mask_shift_left=3, rank_bit=None)
PLATFORMS = {"thor": THOR, "orin": ORIN_NANO}
CHANNEL_STRIPE = 512


@dataclass(frozen=True)
class Mapping:
    physical_address: int
    channel: int
    package: int
    rank: int
    bank: int
    in_dram: bool


def map_pa(p: Platform, pa: int) -> Mapping:
    ch = p.channel(pa)
    return Mapping(pa, ch, p.package(ch), p.rank(pa), p.bank(pa), p.in_dram(pa))


def parse_integer(value: str) -> int:
    """Base-0 integer with an optional binary-size suffix (K, KiB, M, G, ...)."""
    text = value.strip().replace("_", "").lower()
    for suffix, mult in sorted({"k": 1 << 10, "kb": 1 << 10, "kib": 1 << 10, "m": 1 << 20, "mb": 1 << 20,
                                "mib": 1 << 20, "g": 1 << 30, "gb": 1 << 30, "gib": 1 << 30}.items(),
                               key=lambda kv: -len(kv[0])):
        if text.endswith(suffix):
            return int(text[: -len(suffix)], 0) * mult
    return int(text, 0)


PAGE_PRESENT, PAGE_SWAPPED, PFN_MASK = 1 << 63, 1 << 62, (1 << 55) - 1


def virtual_to_physical(pid: int, va: int) -> int:
    """CPU VA -> PA through /proc/PID/pagemap (needs root / CAP_SYS_ADMIN; otherwise the PFN reads as 0)."""
    page = os.sysconf("SC_PAGE_SIZE")
    idx, off = divmod(va, page)
    with open(f"/proc/{pid}/pagemap", "rb", buffering=0) as f:
        f.seek(idx * 8)
        raw = f.read(8)
    if len(raw) != 8:
        raise RuntimeError(f"short pagemap read for VA 0x{va:x}")
    e = struct.unpack("=Q", raw)[0]
    if e & PAGE_SWAPPED or not e & PAGE_PRESENT:
        raise RuntimeError(f"VA 0x{va:x} is swapped or has no CPU PTE (a CUDA device pointer has none)")
    if not e & PFN_MASK:
        raise RuntimeError(f"PFN of VA 0x{va:x} is masked: run as root")
    return (e & PFN_MASK) * page + off


def annotate_tsv(p: Platform, src: Path, dst: Path | None) -> int:
    """Rewrite channel/bank/rank/package of every row of a TSV log from its 'pa' column (-1 outside DRAM)."""
    with src.open(newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if not reader.fieldnames or "pa" not in reader.fieldnames:
            raise ValueError(f"{src} has no 'pa' column")
        fields = list(reader.fieldnames) + [k for k in ("channel", "bank", "rank", "package") if k not in reader.fieldnames]
        out = dst.open("w", newline="") if dst else sys.stdout
        w = csv.DictWriter(out, fieldnames=fields, delimiter="\t", lineterminator="\n")
        w.writeheader()
        n = 0
        for row in reader:
            try:
                pa = int(row["pa"], 0)
            except (TypeError, ValueError):
                pa = 0
            if p.in_dram(pa):
                m = map_pa(p, pa)
                row.update(channel=m.channel, bank=m.bank, rank=m.rank, package=m.package)
            else:
                row.update(channel=-1, bank=-1, rank=-1, package=-1)
            w.writerow(row)
            n += 1
        if dst:
            out.close()
    return n


def self_test() -> None:
    """Stripe property (every aligned 8 KiB / 4 KiB holds all channels) and the Thor answers measured on the box."""
    for base in (THOR.dram_base, 0x3_2A80_0000, 0x1F_0000_E000):
        chans = [THOR.channel(base + o) for o in range(0, 8192, CHANNEL_STRIPE)]
        assert sorted(chans) == list(range(16)), (hex(base), chans)
    # (channel, rank, bank): 0x1d80400000 row-conflicts with 0x136b842040 and 0x126cce3d00; 0xd8ce90e00 shares
    # channel and bank but not rank and does not conflict.
    known = {0x80000000: (0, 0, 0), 0x80000200: (10, 0, 0), 0x1D80400000: (3, 1, 5), 0x136B842040: (3, 1, 5),
             0x126CCE3D00: (3, 1, 5), 0xD8CE90E00: (3, 0, 5), 0x2000000000: (10, 1, 2)}
    for pa, want in known.items():
        m = map_pa(THOR, pa)
        assert (m.channel, m.rank, m.bank) == want, (hex(pa), m, want)
    for base in (0, ORIN_NANO.dram_base, 0x2_4900_0000):
        assert len({ORIN_NANO.channel(base + o) for o in range(0, 4096, CHANNEL_STRIPE)}) == 8, hex(base)
        assert {ORIN_NANO.bank(base + o) for o in range(0, 128 << 10, CHANNEL_STRIPE)} == set(range(16)), hex(base)
    print("dram_map self-test passed (Thor: 16 channels per 8 KiB, known answers; Orin: 8 channels per 4 KiB, "
          "16 banks per 128 KiB)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("platform", choices=sorted(PLATFORMS))
    ap.add_argument("--json", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("pa")
    s.add_argument("addresses", nargs="+", type=parse_integer)
    s = sub.add_parser("range")
    s.add_argument("start", type=parse_integer)
    s.add_argument("length", type=parse_integer)
    s.add_argument("--step", type=parse_integer, default=CHANNEL_STRIPE)
    s = sub.add_parser("pagemap")
    s.add_argument("--pid", type=int, required=True)
    s.add_argument("addresses", nargs="+", type=parse_integer)
    s = sub.add_parser("annotate")
    s.add_argument("tsv", type=Path)
    s.add_argument("-o", "--output", type=Path)
    sub.add_parser("masks")
    sub.add_parser("self-test")
    a = ap.parse_args()
    p = PLATFORMS[a.platform]
    if a.cmd == "self-test":
        self_test()
        return 0
    if a.cmd == "masks":
        print(f"{p.name}: a = {'PA - 0x%x' % p.dram_base if p.channel_on_offset else 'PA'}")
        for i, m in enumerate(p.channel_masks):
            print(f"  channel bit {i}: parity(a & 0x{m:x})")
        return 0
    if a.cmd == "annotate":
        print(f"annotated {annotate_tsv(p, a.tsv, a.output)} rows", file=sys.stderr)
        return 0
    if a.cmd == "pa":
        pas = a.addresses
    elif a.cmd == "range":
        pas = list(range(a.start, a.start + a.length, a.step))
    else:
        pas = [virtual_to_physical(a.pid, va) for va in a.addresses]
    rows = [map_pa(p, pa) for pa in pas]
    if a.json:
        print(json.dumps([asdict(r) | {"physical_address_hex": f"0x{r.physical_address:x}"} for r in rows], indent=1))
    else:
        print("physical address     ch  pkg  rank  bank  in DRAM")
        for r in rows:
            print(f"0x{r.physical_address:016x}  {r.channel:2d}  {r.package:3d}  {r.rank:4d}  {r.bank:4d}  "
                  f"{'yes' if r.in_dram else 'NO'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
