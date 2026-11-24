"""Teledyne LeCroy T3DSO3000(HD) oscilloscope over VISA: open, arm, read a channel, save captures.

The scope watches the near-field probe over the GPU (channel `scope.channel`); its trigger (an interval
trigger on that channel) drives the VC Glitcher through the Aux output. Per trial the controller arms a
single acquisition, and after the trial it can save the screenshot and the raw channel codes.

Also used by analysis scripts (e.g. the trigger-setting test): open_resource(), arm_single(),
wait_stopped(), read_waveform(), save_capture(), trigger_state()/restore_trigger_state().
"""

from __future__ import annotations

import contextlib
import json
import struct
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# SCPI (instrument facts, not settings)
ARM_SINGLE = ":TRIGger:MODE SINGle"
TRIGGER_STATUS = ":TRIGger:STATus?"
SCREENSHOT_PNG = ":PRINt? PNG"
PNG_MAGIC = b"\x89PNG"
STATUS_POLL_S = 0.1

# WAVEDESC block (LeCroy-style preamble returned by :WAVeform:PREamble?): byte offsets of the fields used
WAVEDESC_MAGIC = b"WAVEDESC"
WAVEDESC_FIELDS = {"v_per_div": (156, "f"), "offset_v": (160, "f"), "codes_per_div": (164, "f"),
                   "dt_s": (176, "f"), "delay_s": (180, "d")}
WAVE_DTYPE = "<i2"   # :WAVeform:WIDTh WORD -> signed 16-bit codes

# trigger-state queries for save/restore (interval and Nth-edge trigger)
TRIGGER_STATE_QUERIES = (
    ":TRIGger:TYPE?", ":TRIGger:MODE?", ":TRIGger:INTerval:SOURce?", ":TRIGger:INTerval:SLOPe?",
    ":TRIGger:INTerval:LIMit?", ":TRIGger:INTerval:TLOWer?", ":TRIGger:INTerval:TUPPer?",
    ":TRIGger:INTerval:LEVel?", ":TRIGger:INTerval:COUPling?", ":TRIGger:NEDGe:SOURce?",
    ":TRIGger:NEDGe:SLOPe?", ":TRIGger:NEDGe:IDLE?", ":TRIGger:NEDGe:EDGE?", ":TRIGger:NEDGe:LEVel?",
)


@dataclass(frozen=True)
class ScopeSettings:
    resource: str | None            # VISA resource; None -> USB auto-detect from the ids below
    usb_vendor_id: str | None
    usb_product_id: str | None
    usb_serial: str | None
    timeout_ms: int
    visa_backend: str | None = None  # pyvisa backend, e.g. "@py"; None = pyvisa's default


@dataclass(frozen=True)
class Waveform:
    codes: np.ndarray       # raw signed 16-bit codes
    preamble: bytes         # raw :WAVeform:PREamble? answer
    volts: np.ndarray
    t0_s: float             # time of the first sample relative to the trigger
    dt_s: float


# ---- resource discovery -----------------------------------------------------------------------------------

def _norm(value: str) -> str:
    return value.lower().replace("0x", "")


def usb_candidates(s: ScopeSettings) -> list[str]:
    if not (s.usb_serial and s.usb_vendor_id and s.usb_product_id):
        return []
    v, p = f"0x{_norm(s.usb_vendor_id).upper()}", f"0x{_norm(s.usb_product_id).upper()}"
    return [f"USB0::{v}::{p}::{s.usb_serial}::INSTR", f"USB0::{v}::{p}::{s.usb_serial}::0::INSTR"]


def find_usb(resources, s: ScopeSettings) -> str | None:
    if not (s.usb_vendor_id and s.usb_product_id):
        return None
    v, p = _norm(s.usb_vendor_id), _norm(s.usb_product_id)
    return next((r for r in resources if v in _norm(r) and p in _norm(r)), None)


@contextlib.contextmanager
def open_resource(s: ScopeSettings):
    """Open the scope (explicit resource, else the USB device found by vendor/product id, else the serial-
    number resource strings) and yield the pyvisa instrument."""
    import pyvisa
    rm = pyvisa.ResourceManager(s.visa_backend) if s.visa_backend else pyvisa.ResourceManager()
    scope = None
    try:
        if s.resource:
            tried = [s.resource]
        else:
            resources = rm.list_resources()
            print(f"Resources: {resources}")
            found = find_usb(resources, s)
            tried = ([found] if found else []) + [c for c in usb_candidates(s) if c != found]
            if not tried:
                raise RuntimeError("no scope resource configured and no USB ids/serial to find one")
        last = None
        for candidate in tried:
            try:
                print(f"Opening scope resource: {candidate}")
                scope = rm.open_resource(candidate)
                break
            except Exception as exc:   # noqa: BLE001 - try the next candidate
                last = exc
        if scope is None:
            raise RuntimeError(f"could not open scope resource; tried {tried}") from last
        scope.timeout = s.timeout_ms
        scope.write_termination = "\n"
        scope.read_termination = "\n"
        with contextlib.suppress(Exception):
            print(f"Scope ID: {scope.query('*IDN?').strip()}")
        yield scope
    finally:
        if scope is not None:
            with contextlib.suppress(Exception):
                scope.close()
        with contextlib.suppress(Exception):
            rm.close()


# ---- per-trial use ----------------------------------------------------------------------------------------

def arm_single(scope) -> None:
    scope.write(ARM_SINGLE)
    print(f"Scope armed with: {ARM_SINGLE}")


def wait_stopped(scope, wait_s: float) -> str:
    """Poll the trigger status until the single acquisition is done ("Stop") or wait_s passes."""
    status = "?"
    for _ in range(max(1, int(round(wait_s / STATUS_POLL_S)))):
        with contextlib.suppress(Exception):
            status = scope.query(TRIGGER_STATUS).strip()
        if status.lower().startswith("stop"):
            break
        time.sleep(STATUS_POLL_S)
    return status


def screenshot_png(scope) -> bytes:
    scope.write(SCREENSHOT_PNG)
    img = scope.read_raw()
    k = img.find(PNG_MAGIC)
    return img[k:] if k >= 0 else img


def parse_wavedesc(preamble: bytes) -> dict:
    p = preamble[preamble.find(WAVEDESC_MAGIC):]
    return {name: struct.unpack_from("<" + fmt, p, off)[0] for name, (off, fmt) in WAVEDESC_FIELDS.items()}


def parse_block(data: bytes) -> np.ndarray:
    """IEEE 488.2 definite-length block '#<n><len><data>' -> codes."""
    n = int(data[1:2])
    size = int(data[2:2 + n])
    return np.frombuffer(data[2 + n:2 + n + size], WAVE_DTYPE)


def to_volts(codes: np.ndarray, desc: dict) -> tuple[np.ndarray, float, float]:
    """(volts, t0, dt). The horizontal delay is the time of the screen centre after the trigger."""
    v = codes.astype(np.float32) / desc["codes_per_div"] * desc["v_per_div"] - desc["offset_v"]
    dt = desc["dt_s"]
    return v, desc["delay_s"] - len(v) * dt / 2, dt


def read_waveform(scope, channel: str) -> Waveform:
    scope.write(f":WAVeform:SOURce {channel}")
    scope.write(":WAVeform:STARt 0")
    scope.write(":WAVeform:POINt 0")
    scope.write(":WAVeform:WIDTh WORD")
    scope.write(":WAVeform:PREamble?")
    pre = scope.read_raw()
    scope.write(":WAVeform:DATA?")
    codes = parse_block(scope.read_raw())
    volts, t0, dt = to_volts(codes, parse_wavedesc(pre))
    return Waveform(codes, pre, volts, t0, dt)


def save_capture(scope, out_dir: Path, name: str, channel: str, meta: dict, wait_s: float) -> None:
    """Save the last acquisition as <name>.png (screenshot) and <name>.npz (raw codes under the channel's
    lower-case name, e.g. 'c2', the WAVEDESC preamble and `meta` as JSON). Errors are printed, not raised:
    a failed capture must not stop the run."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    status = wait_stopped(scope, wait_s)   # the single acquisition is normally complete long before
    try:
        (out_dir / f"{name}.png").write_bytes(screenshot_png(scope))
        w = read_waveform(scope, channel)
        meta = {**meta, "saved_at": time.time(), "scope_status": status}
        np.savez_compressed(out_dir / f"{name}.npz", **{channel.lower(): w.codes},
                            preamble=np.frombuffer(w.preamble, np.uint8),
                            meta=json.dumps(meta, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
        print(f"Scope capture saved ({status}, {time.time() - t0:.1f} s)", flush=True)
    except Exception as exc:   # noqa: BLE001
        print(f"Scope capture failed: {exc}", flush=True)


# ---- setup ------------------------------------------------------------------------------------------------

def trigger_state(scope) -> dict:
    return {q: scope.query(q).strip() for q in TRIGGER_STATE_QUERIES}


def restore_trigger_state(scope, saved: dict) -> None:
    for q in TRIGGER_STATE_QUERIES:
        if q.startswith((":TRIGger:INTerval", ":TRIGger:NEDGe")):
            scope.write(f"{q[:-1]} {saved[q]}")
    scope.write(f":TRIGger:TYPE {saved[':TRIGger:TYPE?']}")
    scope.write(f":TRIGger:MODE {saved[':TRIGger:MODE?']}")


def settings_commands(cfg) -> list[str]:
    """SCPI for the acquisition and interval-trigger settings in [scope] (used when scope.apply_settings)."""
    ch = cfg.get("scope.channel")
    n = ch.lstrip("Cc")
    return [
        f":TIMebase:SCALe {cfg.get('scope.timebase_s_per_div', float):g}",
        f":TIMebase:DELay {cfg.get('scope.horizontal_delay_s', float):g}",
        f":ACQuire:MDEPth {cfg.get('scope.memory_depth')}",
        f":CHANnel{n}:SCALe {cfg.get('scope.channel_v_per_div', float):g}",
        f":CHANnel{n}:OFFSet {cfg.get('scope.channel_offset_v', float):g}",
        ":TRIGger:TYPE INTerval",
        f":TRIGger:INTerval:SOURce {ch}",
        f":TRIGger:INTerval:SLOPe {cfg.get('scope.trigger_slope')}",
        f":TRIGger:INTerval:LIMit {cfg.get('scope.trigger_limit')}",
        f":TRIGger:INTerval:TLOWer {cfg.get('scope.trigger_interval_s', float):g}",
        f":TRIGger:INTerval:LEVel {cfg.get('scope.trigger_level_v', float):g}",
        f":TRIGger:INTerval:COUPling {cfg.get('scope.trigger_coupling')}",
        f":TRIGger:MODE {cfg.get('scope.trigger_mode')}",
    ]


def settings_from_config(cfg) -> ScopeSettings:
    resource = cfg.get("scope.resource", default=None)
    return ScopeSettings(
        resource=resource,
        usb_vendor_id=None if resource else cfg.get("scope.usb_vendor_id"),
        usb_product_id=None if resource else cfg.get("scope.usb_product_id"),
        usb_serial=cfg.get("scope.usb_serial", default=None),
        timeout_ms=cfg.get("scope.timeout_ms", int),
        visa_backend=cfg.get("scope.visa_backend", default=None),
    )


@contextlib.contextmanager
def open_from_config(cfg):
    """The controller's scope: None when scope.enabled is false; applies [scope] settings if asked."""
    if not cfg.get("scope.enabled", bool):
        yield None
        return
    with open_resource(settings_from_config(cfg)) as scope:
        if cfg.get("scope.apply_settings", bool):
            for cmd in settings_commands(cfg):
                scope.write(cmd)
            print("Scope settings applied", flush=True)
        yield scope
