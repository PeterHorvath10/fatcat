"""Inference-time worker on the Orin (Ollama): one greedy query per trial, GPIO high during the request.

Waits for Ollama and the model, loads it (keep-alive), finds its llama-server runner, then accepts the host
controller on network.listen_host:listen_port. Per trial: handshake, query, one JSON record appended to
ollama.output.

Prompt: with ollama.follow_prompt_slot the host's prompt slot picks the line of ollama.prompt_file (the host
must then send the prompt-slot handshake); otherwise every trial uses line ollama.prompt_index.

Reload: a model whose weights or logits are corrupted emits a special token (ollama.reload_token) or only
question marks; after such an answer (and every ollama.reload_every trials, 0 = never) the model is
unloaded and loaded again from disk before the next trial.

    python3 -m jetson.orin.ollama_query_trigger
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from common import config, protocol
from jetson.common import prompts as prompt_files
from jetson.common import results
from jetson.common.gpio import GpioOutput
from jetson.orin.ollama_client import OllamaClient

RECORD_SCHEMA_VERSION = 3


def reload_reason(result: dict[str, Any], trial: int, reload_every: int, corruption_token: str) -> str | None:
    """Why the model must be reloaded after this trial, or None."""
    if reload_every and trial % reload_every == 0:
        return f"configured after every {reload_every} queries"
    reply = result.get("response")
    if not isinstance(reply, str):
        return None
    if corruption_token in reply:
        return f"generated {corruption_token}"
    stripped = reply.strip()
    if stripped and set(stripped) == {"?"}:
        return f"generated only question marks: {reply!r}"
    return None


def query_record(result: dict[str, Any], *, run_id: str, trial: int, handshake: protocol.Handshake,
                 prompt: str) -> dict[str, Any]:
    """The JSONL record of one trial (schema 3; the runner gives no token ids, hence the null fields)."""
    return {
        "schema_version": RECORD_SCHEMA_VERSION,
        "collection_id": handshake.collection_id,
        "run_id": run_id,
        "trial": trial,
        "attempt_index": handshake.attempt_index,
        "saved_at": results.utc_now(),
        "model": result.get("model"),
        "prompt": prompt,
        "generated_text": result.get("response"),
        "generated_token_ids": None,
        "context_token_ids": result.get("context"),
        "done": result.get("done"),
        "done_reason": result.get("done_reason"),
        "created_at": result.get("created_at"),
        "total_duration_ns": result.get("total_duration"),
        "load_duration_ns": result.get("load_duration"),
        "prompt_eval_count": result.get("prompt_eval_count"),
        "prompt_eval_duration_ns": result.get("prompt_eval_duration"),
        "eval_count": result.get("eval_count"),
        "eval_duration_ns": result.get("eval_duration"),
    }


def main() -> int:
    cfg = config.load("orin.toml")
    client = OllamaClient(cfg)
    client.wait_for_model()
    client.warm_up()
    client.wait_for_runner()
    prompts = prompt_files.load_lines(Path(cfg.get("ollama.prompt_file")))
    follow_slot = cfg.get("ollama.follow_prompt_slot", bool)
    fixed = None if follow_slot else prompts[cfg.get("ollama.prompt_index", int)]
    reload_every = cfg.get("ollama.reload_every", int)
    corruption_token = cfg.get("ollama.reload_token")
    queries = cfg.get("ollama.queries", int)
    output = Path(cfg.get("ollama.output")).expanduser()
    run_id = results.new_run_id()
    print(f"Starting run {run_id}; results file: {output}; "
          + ("prompt chosen by the host's prompt slot" if follow_slot else f"fixed prompt: {fixed}"), flush=True)

    with GpioOutput(cfg.get("gpio.chip"), cfg.get("ollama.gpio_line", int), cfg.get("gpio.consumer")) as gpio:
        with protocol.serve_one(cfg.get("network.listen_host"), cfg.get("network.listen_port", int)) as conn:
            for trial in range(1, queries + 1):
                handshake = protocol.request_trial_any(conn)
                if fixed is not None:
                    prompt = fixed
                else:
                    if handshake.prompt_slot is None:
                        raise ValueError("ollama.follow_prompt_slot is set but the host sent no prompt slot")
                    prompt = prompts[handshake.prompt_slot % len(prompts)]
                    if handshake.prompt_repeat == 1:
                        print(f"Prompt {handshake.prompt_slot % len(prompts) + 1}/{len(prompts)}: {prompt}",
                              flush=True)
                gpio.set_value(1)
                try:
                    result = client.complete(prompt)
                finally:
                    gpio.set_value(0)
                results.append_jsonl(output, query_record(result, run_id=run_id, trial=trial, handshake=handshake,
                                                          prompt=prompt))
                print(f"Saved run {run_id}, trial {trial} to {output}.", flush=True)
                print(f"Ollama query {trial} done.", flush=True)
                print(result["response"], flush=True)
                reason = reload_reason(result, trial, reload_every, corruption_token)
                if reason is not None:
                    print(f"Reloading Ollama model after query {trial}: {reason}.", flush=True)
                    client.reload()
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, TimeoutError, ValueError, config.ConfigError) as error:
        print(f"error: {error}", file=sys.stderr, flush=True)
        sys.exit(1)
