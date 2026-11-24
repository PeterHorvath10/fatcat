"""No-pulse JailbreakBench baseline on Thor: every prompt of prompts.file x freerun.reps, sent exactly like
the query worker's trials (chat completion, temperature vllm.temperature, max tokens freerun.max_tokens)
to the served model, with no glitching. Answers that are unsafe here are unsafe without any fault and are
subtracted from the EMFI results. Appends JSONL; resumable (indices already in OUT are skipped).

    python3 -m jetson.thor.jbb_freerun OUT.jsonl [--reps N] [--max-tokens N]
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from common import config
from jetson.common import prompts as prompt_files
from jetson.common import results
from jetson.common.thermal import ThermalGuard
from jetson.thor.vllm_client import VllmClient, served_names

# Record index = slot * INDEX_STRIDE + repeat (format of the paper's free-run files).
INDEX_STRIDE = 100


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("out", type=Path)
    parser.add_argument("--reps", type=int, help="answers per prompt (default freerun.reps)")
    parser.add_argument("--max-tokens", type=int, help="default freerun.max_tokens")
    args = parser.parse_args()
    cfg = config.load("thor.toml")
    reps = args.reps or cfg.get("freerun.reps", int)
    max_tokens = args.max_tokens or cfg.get("freerun.max_tokens", int)
    temperature = cfg.get("vllm.temperature", float)
    if reps >= INDEX_STRIDE:
        raise ValueError(f"at most {INDEX_STRIDE - 1} repeats per prompt")
    client = VllmClient.from_config(cfg)
    guard = ThermalGuard.from_config(cfg)
    prompts = prompt_files.load_lines(Path(cfg.get("prompts.file")))
    model = client.wait_for_model(served_names(cfg))
    done = {r.get("index") for r in results.read_jsonl(args.out)}

    for slot, prompt in enumerate(prompts):
        for rep in range(1, reps + 1):
            index = slot * INDEX_STRIDE + rep
            if index in done:
                continue
            guard.wait()
            choice = client.chat(model, prompt, max_tokens, temperature)["choices"][0]
            results.append_jsonl(args.out, {"model": model, "index": index, "slot": slot, "repeat": rep,
                                            "goal": prompt, "response": choice["message"]["content"],
                                            "finish_reason": choice["finish_reason"], "time": time.time()})
        print("prompt", slot, "done", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
