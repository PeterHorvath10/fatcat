"""Weight-reload worker (Thor): every trial reloads the model weights from disk while the host glitches.

Same protocol and prompt choice as vllm_query_trigger.py. The vLLM server must run with
vllm.weight_check = true (dev-mode /collective_rpc + jetson.thor.weight_check_ext in the worker).

Per trial, after the handshake:
  * reload_trigger.reload_only = true (the paper's fp8 runs): no prompt; the trial IS the reload
    (POST /collective_rpc reload_weights), which the host glitches. vLLM copies the checkpoint back into
    the existing tensor storage, so the weights keep their DRAM addresses.
  * otherwise: query first, then reload (every reload_every trials, or at once when the corruption watchdog
    trips), so the reload never shifts the glitch timing and the next trial starts on clean weights.
Then, if the host asked for it (check-weights flag CHECK_WEIGHTS_REPORT, its --weight-report option), the
loaded weights are compared with a clean baseline and a weight report goes back before the next COLLECT:
MATCH, CHANGED (tensors differ in memory), TRANSIENT (hash differed but every element re-reads equal: a
fault during the check itself) or ERROR.

Baseline (once at start): from the sha256-verified checkpoint file (bf16), or with baseline_from_reload
(needed with fp8: vLLM quantizes on load, so the checkpoint cannot be compared) from a clean reload; either
way further clean reloads must MATCH it before trials start.

Options from config: drop_cache (evict the checkpoint from the page cache before each reload so it comes
from NVMe, not from possibly glitched DRAM), zero_before_reload (zero the weights first: a write the reload
loses then shows as zero; the report splits lost writes from garbage), stop_on_change (exit with
EXIT_WEIGHTS_CHANGED after a CHANGED report; the unit does not restart on it).

    python3 -m jetson.thor.vllm_reload_trigger [--free-run [--weight-check]]

--free-run: no controller; trials run back to back over the prompts (reload_trigger.free_run_repeats per
prompt, free_run_delay_s apart), --weight-check checks the weights after every trial (log only).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import sys
import time
from collections.abc import Iterator
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from common import config, http, protocol
from jetson.common import results
from jetson.common import prompts as prompt_files
from jetson.thor.vllm_client import VllmClient, served_names
from jetson.thor.vllm_query_trigger import corrupted, run_query_trial, settings_from_config

EXIT_WEIGHTS_CHANGED = 3          # the unit has RestartPreventExitStatus=3
WEIGHT_FILE_PATTERNS = ("*.safetensors", "*.bin", "*.pt")
SHA256_SUFFIX = ".sha256"         # known-good checkpoint hash kept next to the file
REFERENCE_READS = 4               # drive reads to get two equal sha256 values
HASH_CHUNK_BYTES = 1 << 24
RPC_ERRORS = (http.HTTPError, http.URLError, TimeoutError, KeyError, ValueError)


@dataclass
class ReloadSettings:
    reload_every: int
    reload_timeout_s: float
    reload_only: bool
    drop_cache: bool
    baseline_from_reload: bool
    zero_before_reload: bool
    stop_on_change: bool
    baseline_verify_tries: int
    weight_report_log: Path
    free_run: bool = False
    free_run_repeats: int = 1
    free_run_delay_s: float = 0.0
    weight_check: bool = False


def reload_settings(cfg: config.Config, args: argparse.Namespace) -> ReloadSettings:
    s = "reload_trigger"
    settings = ReloadSettings(
        reload_every=cfg.get(f"{s}.reload_every", int),
        reload_timeout_s=cfg.get(f"{s}.reload_timeout_s", float),
        reload_only=cfg.get(f"{s}.reload_only", bool),
        drop_cache=cfg.get(f"{s}.drop_cache", bool),
        baseline_from_reload=cfg.get(f"{s}.baseline_from_reload", bool),
        zero_before_reload=cfg.get(f"{s}.zero_before_reload", bool),
        stop_on_change=cfg.get(f"{s}.stop_on_change", bool),
        baseline_verify_tries=cfg.get(f"{s}.baseline_verify_tries", int),
        weight_report_log=Path(cfg.get(f"{s}.weight_report_log")).expanduser(),
        free_run=args.free_run,
        weight_check=args.weight_check,
    )
    if args.free_run:
        settings.free_run_repeats = cfg.get(f"{s}.free_run_repeats", int)
        settings.free_run_delay_s = cfg.get(f"{s}.free_run_delay_s", float)
    return settings


# ---- checkpoint files --------------------------------------------------------------------------------

def weight_files(model_path: Path) -> list[Path]:
    files = sorted({f for pattern in WEIGHT_FILE_PATTERNS for f in model_path.glob(pattern)})
    if not files:
        raise ValueError(f"no weight files found in {model_path}")
    return files


def drop_page_cache(files: Sequence[Path]) -> int:
    """Evict files from the page cache (POSIX_FADV_DONTNEED drops clean, unmapped pages without root),
    so the next read comes from the drive; returns the bytes covered."""
    total = 0
    for path in files:
        fd = os.open(path, os.O_RDONLY)
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            total += os.fstat(fd).st_size
        finally:
            os.close(fd)
    return total


def file_sha256(path: Path) -> str:
    drop_page_cache([path])
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(HASH_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def reference_sha256(path: Path) -> str:
    """Known-good sha256 of a checkpoint file, kept in <file>.sha256; created on first use from two
    drive reads that agree, so a glitched read cannot become the reference."""
    ref = path.with_name(path.name + SHA256_SUFFIX)
    if ref.exists():
        return ref.read_text().split()[0]
    previous = None
    for _ in range(REFERENCE_READS):
        digest = file_sha256(path)
        if digest == previous:
            ref.write_text(f"{digest}  {path.name}\n")
            print(f"weight baseline: recorded reference sha256 {digest} in {ref}", flush=True)
            return digest
        previous = digest
    raise ValueError(f"{path}: {REFERENCE_READS} reads gave no two equal sha256 values")


# ---- reload and weight check -------------------------------------------------------------------------

class Reloader:
    def __init__(self, client: VllmClient, settings: ReloadSettings, files: list[Path]):
        self.client, self.s, self.files = client, settings, files

    def rpc(self, method: str, kwargs: dict | None = None) -> Any:
        return self.client.collective_rpc(method, kwargs or {}, self.s.reload_timeout_s)

    def zero_weights(self) -> bool:
        started = time.monotonic()
        try:
            out = self.rpc("wc_zero")
        except RPC_ERRORS as error:
            print(f"zero: wc_zero failed ({error}); is weight_check_ext loaded and up to date "
                  "(restart vLLM after updating it)?", flush=True)
            return False
        if out.get("nonzero_after"):
            print(f"zero: WARNING {out['nonzero_after']} elements still non-zero after zeroing", flush=True)
        print(f"zero: {out.get('zeroed_elems')} weight elements zeroed in {time.monotonic() - started:.2f} s.",
              flush=True)
        return True

    def reload(self, reason: str, *, zero_first: bool = True) -> bool:
        """Reload the served weights from disk into their existing storage."""
        if zero_first and self.s.zero_before_reload and not self.zero_weights():
            return False
        if self.s.drop_cache:
            reason += f", page cache dropped ({drop_page_cache(self.files) / 2**30:.2f} GiB)"
        started = time.monotonic()
        try:
            self.client.collective_rpc("reload_weights", None, self.s.reload_timeout_s)
        except http.HTTPError as error:
            hint = " (is VLLM_SERVER_DEV_MODE=1 set?)" if error.code == 404 else ""
            print(f"reload ({reason}): HTTP {error.code} from {self.client.root}/collective_rpc{hint}", flush=True)
            return False
        except (http.URLError, TimeoutError) as error:
            print(f"reload ({reason}): {self.client.root}/collective_rpc failed: {error}", flush=True)
            return False
        print(f"reload ({reason}): weights reloaded from disk in {time.monotonic() - started:.2f} s.", flush=True)
        return True

    def take_baseline(self) -> bool:
        """Reference from the checkpoint (or a clean reload), then require a clean reload to MATCH it."""
        try:
            started = time.monotonic()
            if self.s.baseline_from_reload:
                if not self.reload("baseline", zero_first=False):
                    return False
                out = self.rpc("wc_baseline")
            else:
                files = [{"path": str(f), "sha256": reference_sha256(f)} for f in self.files]
                out = self.rpc("wc_baseline_from_file", {"files": files})
        except RPC_ERRORS as error:
            print(f"weight baseline: unavailable ({error}); is vLLM started with vllm.weight_check = true?",
                  flush=True)
            return False
        source = "a clean reload" if self.s.baseline_from_reload else "the sha256-verified checkpoint"
        print(f"weight baseline: {out.get('tensors')} tensors, {out.get('bytes', 0) / 2**30:.2f} GiB from "
              f"{source} in {time.monotonic() - started:.1f} s.", flush=True)
        for attempt in range(1, self.s.baseline_verify_tries + 1):
            if not self.reload(f"baseline verify {attempt}", zero_first=False):
                return False
            try:
                check = self.rpc("wc_check", {"attempt": -attempt})
            except RPC_ERRORS as error:
                print(f"weight baseline: verify check failed ({error})", flush=True)
                return False
            changed = [t["name"] for t in check.get("changed", []) if t.get("changed_elems") != 0]
            if not changed:
                print("weight baseline: loaded weights match the checkpoint.", flush=True)
                return True
            print(f"weight baseline: reload {attempt} differs from the checkpoint in {len(changed)} tensor(s) "
                  f"{changed}; reloading again.", flush=True)
        print("weight baseline: loaded weights never matched the checkpoint.", flush=True)
        return False

    def check(self, attempt_index: int) -> tuple[str, list[dict], dict]:
        """(status name, tensors that really changed, summary for the host)."""
        try:
            out = self.rpc("wc_check", {"attempt": attempt_index})
        except RPC_ERRORS as error:
            return "ERROR", [], {"error": str(error)}
        changed = out.get("changed", [])
        # changed_elems == 0: the hash differed but the re-read equals the baseline element for element, so
        # the tensor was only misread during the check. None means no element diff was possible: count it.
        persistent = [t for t in changed if t.get("changed_elems") != 0]
        if out.get("baselined"):
            # The worker had no baseline (vLLM restarted) and took one now: nothing to compare against.
            return "ERROR", persistent, {"error": "no baseline on the worker; re-baselined now", "_changed": changed}
        status = "CHANGED" if persistent else "TRANSIENT" if changed else "MATCH"
        summary: dict[str, Any] = {"tensors": [{k: t.get(k) for k in ("name", "changed_elems", "zero_elems")}
                                               for t in changed]}
        if self.s.zero_before_reload and persistent:
            # Weights were zeroed before the reload: a changed element that is zero was never written (lost
            # write); anything else is a corrupted write (AES-XTS garbage).
            lost = sum(t.get("zero_elems") or 0 for t in persistent)
            summary["lost_write_elems"] = lost
            summary["garbage_elems"] = sum(t.get("changed_elems") or 0 for t in persistent) - lost
        summary["_changed"] = changed
        return status, persistent, summary


def check_and_report(reloader: Reloader, conn: socket.socket | None, handshake: protocol.Handshake) -> str:
    """Compare the weights with the baseline, log, and send the weight report (if there is a host)."""
    started = time.monotonic()
    status, persistent, summary = reloader.check(handshake.attempt_index)
    changed = summary.pop("_changed", [])
    print(f"weight check (attempt {handshake.attempt_index}): {status}, {len(persistent)} tensor(s) changed in "
          f"memory, {len(changed) - len(persistent)} misread only, {time.monotonic() - started:.2f} s."
          + (f" [{summary['lost_write_elems']} elems still zero = lost writes, {summary['garbage_elems']} garbage]"
             if "lost_write_elems" in summary else "")
          + (f" {json.dumps(summary)}" if status != "MATCH" else ""), flush=True)
    if status != "MATCH":
        results.append_jsonl(reloader.s.weight_report_log, {
            "collection_id": handshake.collection_id, "attempt_index": handshake.attempt_index,
            "status": status, "changed": changed,
            **({"error": summary["error"]} if "error" in summary else {})})
    if conn is not None:
        protocol.send_weight_report(conn, handshake.attempt_index, status, len(persistent),
                                    json.dumps(summary).encode("utf-8"))
    return status


def end_of_trial_check(reloader: Reloader, conn: socket.socket | None, handshake: protocol.Handshake) -> bool:
    """Run the end-of-trial weight check if requested; True means stop the run."""
    wanted = reloader.s.weight_check if conn is None else handshake.check_weights == protocol.CHECK_WEIGHTS_REPORT
    if not wanted:
        return False
    if check_and_report(reloader, conn, handshake) == "CHANGED" and reloader.s.stop_on_change:
        print(f"STOPPING: loaded weights CHANGED on attempt {handshake.attempt_index} (stop_on_change); details "
              f"in {reloader.s.weight_report_log}.", flush=True)
        return True
    return False


# ---- trial loop --------------------------------------------------------------------------------------

def free_run_handshakes(s: ReloadSettings, run_id: str, prompt_count: int, queries: int
                        ) -> Iterator[protocol.Handshake]:
    """Stand-in for the controller: slot/repeat cycle through the prompts."""
    for trial in range(queries):
        if trial and s.free_run_delay_s > 0:
            time.sleep(s.free_run_delay_s)
        yield protocol.Handshake(attempt_index=trial + 1, collection_id=f"free-run-{run_id}",
                                 prompt_slot=(trial // s.free_run_repeats) % prompt_count,
                                 prompt_repeat=trial % s.free_run_repeats + 1, check_weights=0)


def controller_handshakes(conn: socket.socket, queries: int) -> Iterator[protocol.Handshake]:
    for _ in range(queries):
        yield protocol.request_prompt_trial(conn)


def run_trials(reloader: Reloader, client: VllmClient, query, prompts: list[str], queries: int,
               conn: socket.socket | None) -> int:
    s = reloader.s
    handshakes = (free_run_handshakes(s, query.run_id, len(prompts), queries) if conn is None
                  else controller_handshakes(conn, queries))
    for trial, handshake in enumerate(handshakes, start=1):
        if s.reload_only:
            print(f"Trial {trial}/{queries} (attempt {handshake.attempt_index}): reload only.", flush=True)
            if not reloader.reload(f"trial {trial}"):
                return 1
        else:
            count = run_query_trial(client, query, prompts, trial, handshake, queries)
            bad = corrupted(query, count)
            if bad:
                print(f"CORRUPTION DETECTED on trial {trial}: '{query.reload_token}' x{count} >= threshold "
                      f"{query.reload_threshold}; reloading now.", flush=True)
            if bad or (s.reload_every > 0 and trial % s.reload_every == 0):
                if not reloader.reload("corruption" if bad else f"after trial {trial}"):
                    return 1
        if end_of_trial_check(reloader, conn, handshake):
            return EXIT_WEIGHTS_CHANGED
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="vLLM weight-reload worker (see module docstring).")
    parser.add_argument("--free-run", action="store_true", help="no controller: trials back to back")
    parser.add_argument("--weight-check", action="store_true", help="free run: check the weights after each trial")
    args = parser.parse_args(argv)

    cfg = config.load("thor.toml")
    client = VllmClient.from_config(cfg)
    model = client.wait_for_model(served_names(cfg))
    s = reload_settings(cfg, args)
    query = settings_from_config(cfg, "reload_trigger", model)
    files = weight_files(client.model_root(model))
    if s.drop_cache:
        print("drop-cache: evicting before every reload: " + ", ".join(map(str, files)), flush=True)
    prompts = prompt_files.load_lines(Path(cfg.get("prompts.file")))
    print(f"Starting run {query.run_id}; results file: {query.output}; {len(prompts)} prompts; reload every "
          f"{s.reload_every} trial(s)" + ("; FREE RUN (no controller)" if s.free_run else "")
          + ("; RELOAD ONLY (no prompts)" if s.reload_only else ""), flush=True)

    if not client.dev_mode_enabled():
        return 1
    reloader = Reloader(client, s, files)
    # Best effort: without a baseline the checks report ERROR; if reloads never matched, CHANGED.
    reloader.take_baseline()

    queries = cfg.get("reload_trigger.queries", int)
    session = nullcontext(None) if s.free_run else protocol.serve_one(
        cfg.get("network.listen_host"), cfg.get("network.listen_port", int))
    with session as conn:
        return run_trials(reloader, client, query, prompts, queries, conn)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, TimeoutError, ValueError, config.ConfigError) as error:
        print(f"error: {error}", file=sys.stderr, flush=True)
        sys.exit(1)
