"""Inference-time worker (Thor): one greedy vLLM query per trial, prompt chosen by the host.

The host controller connects to network.listen_host:listen_port. For every trial it acknowledges our
COLLECT with the prompt-slot form of the handshake (common/protocol.py); we send line `prompt_slot` of
prompts.file to the served model and append one JSON record to query_trigger.output. The scope sees the
GPU burst of the request and fires the glitcher; the host never waits for our answer.

Corruption watchdog: a hit that corrupts the weights in DRAM makes the model loop on one word (we saw "our"
repeated). If a response contains query_trigger.reload_token at least reload_threshold times (whole word,
any case), the active vLLM unit is restarted (sudoers NOPASSWD rule, see jetson/install.py) so the next
trial runs on weights freshly loaded from disk. reload_threshold <= 0 disables it.

    python3 -m jetson.thor.vllm_query_trigger
"""

from __future__ import annotations

import re
import socket
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from common import config, protocol
from jetson.common import prompts as prompt_files
from jetson.common import results, systemd
from jetson.thor.vllm_client import VllmClient, model_units, served_names

RECORD_SCHEMA_VERSION = 2
UNIT_RESTART_TIMEOUT_S = 180


@dataclass
class QuerySettings:
    model: str
    max_tokens: int
    temperature: float
    reload_token: str
    reload_threshold: int
    output: Path
    run_id: str


def count_sentinel(text: str, token: str) -> int:
    """Whole-word, case-insensitive occurrences of token in text."""
    if not text or not token:
        return 0
    return len(re.findall(rf"\b{re.escape(token)}\b", text, flags=re.IGNORECASE))


def query_record(result: dict[str, Any], *, settings: QuerySettings, trial: int,
                 handshake: protocol.Handshake, prompt: str) -> dict[str, Any]:
    """The JSONL record of one trial (schema 2, as analysed for the paper)."""
    try:
        choice = result["choices"][0]
        text = choice["message"]["content"]
        token_ids = choice["token_ids"]
    except (KeyError, IndexError, TypeError) as error:
        raise ValueError("vLLM response did not contain generated text and token IDs") from error
    if not isinstance(token_ids, list) or not all(isinstance(t, int) for t in token_ids):
        raise ValueError("vLLM response contained invalid generated token IDs")
    return {
        "schema_version": RECORD_SCHEMA_VERSION,
        "collection_id": handshake.collection_id,
        "run_id": settings.run_id,
        "trial": trial,
        "attempt_index": handshake.attempt_index,
        "prompt_slot": handshake.prompt_slot,
        "prompt_repeat": handshake.prompt_repeat,
        "saved_at": results.utc_now(),
        "response_id": result.get("id"),
        "model": result.get("model"),
        "prompt": prompt,
        "generated_text": text,
        "generated_token_ids": token_ids,
        "finish_reason": choice.get("finish_reason"),
        "usage": result.get("usage"),
        "sentinel_token": settings.reload_token,
        "sentinel_count": count_sentinel(text, settings.reload_token),
    }


def run_query_trial(client: VllmClient, settings: QuerySettings, prompts: list[str], trial: int,
                    handshake: protocol.Handshake, total: int) -> int:
    """Query the prompt of handshake's slot, save the record; return the sentinel count."""
    slot = handshake.prompt_slot % len(prompts)
    prompt = prompts[slot]
    if handshake.prompt_repeat == 1:
        print(f"Prompt slot {slot}: {prompt}", flush=True)
    print(f"Starting LLM query {trial}/{total} (slot {slot}, repeat {handshake.prompt_repeat}, "
          f"attempt {handshake.attempt_index}).", flush=True)
    result = client.chat(settings.model, prompt, settings.max_tokens, settings.temperature)
    record = query_record(result, settings=settings, trial=trial, handshake=handshake, prompt=prompt)
    results.append_jsonl(settings.output, record)
    print(f"Saved run {settings.run_id}, trial {trial} to {settings.output}.", flush=True)
    print(f"LLM query {trial} done.", flush=True)
    print(record["generated_text"] or "", flush=True)
    return record["sentinel_count"]


def corrupted(settings: QuerySettings, sentinel_count: int) -> bool:
    return settings.reload_threshold > 0 and sentinel_count >= settings.reload_threshold


def reload_by_restart(client: VllmClient, units: list[str], accepted: list[str], trial: int, count: int,
                      settings: QuerySettings) -> None:
    """Reload the weights from disk by restarting the active vLLM unit, then wait until it serves again."""
    unit = systemd.active_unit(units)
    print(f"CORRUPTION DETECTED on trial {trial}: '{settings.reload_token}' x{count} >= threshold "
          f"{settings.reload_threshold}. "
          + (f"Reloading weights by restarting {unit}." if unit else "No active vLLM unit found; cannot reload."),
          flush=True)
    if unit and systemd.systemctl("restart", [unit], timeout_s=UNIT_RESTART_TIMEOUT_S):
        client.wait_for_model(accepted)
        print(f"reload: {unit} restarted; fresh weights loaded.", flush=True)


def serve_trials(conn: socket.socket, client: VllmClient, settings: QuerySettings, prompts: list[str],
                 queries: int, units: list[str], accepted: list[str]) -> None:
    for trial in range(1, queries + 1):
        handshake = protocol.request_prompt_trial(conn)
        count = run_query_trial(client, settings, prompts, trial, handshake, queries)
        if corrupted(settings, count):
            reload_by_restart(client, units, accepted, trial, count, settings)


def settings_from_config(cfg: config.Config, section: str, model: str) -> QuerySettings:
    return QuerySettings(
        model=model,
        max_tokens=cfg.get(f"{section}.max_tokens", int),
        temperature=cfg.get("vllm.temperature", float),
        reload_token=cfg.get("query_trigger.reload_token"),
        reload_threshold=cfg.get("query_trigger.reload_threshold", int),
        output=Path(cfg.get(f"{section}.output")).expanduser(),
        run_id=results.new_run_id(),
    )


def main() -> int:
    cfg = config.load("thor.toml")
    client = VllmClient.from_config(cfg)
    accepted = served_names(cfg)
    model = client.wait_for_model(accepted)
    settings = settings_from_config(cfg, "query_trigger", model)
    prompts = prompt_files.load_lines(Path(cfg.get("prompts.file")))
    print(f"Starting run {settings.run_id}; results file: {settings.output}; {len(prompts)} prompts",
          flush=True)
    conn = protocol.serve_one(cfg.get("network.listen_host"), cfg.get("network.listen_port", int))
    with conn:
        serve_trials(conn, client, settings, prompts, cfg.get("query_trigger.queries", int),
                     model_units(cfg), accepted)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, TimeoutError, ValueError, config.ConfigError) as error:
        print(f"error: {error}", file=sys.stderr, flush=True)
        sys.exit(1)
