"""Sanity check of the served model: greedy answers to a few fixed questions (prefix cache reset first).

    python3 -m jetson.thor.eval.eval_model [--model SERVED_NAME]
"""

from __future__ import annotations

import argparse

from common import config
from jetson.thor.eval import served_model
from jetson.thor.vllm_client import VllmClient

QUESTIONS = ("What is the capital of France?", "Write one sentence about the ocean.", "What is 12 + 7?")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", help="served model name (default: the one vLLM serves)")
    args = parser.parse_args()
    cfg = config.load("thor.toml")
    client = VllmClient.from_config(cfg)
    model = served_model(cfg, client, args.model)
    client.reset_prefix_cache()
    for question in QUESTIONS:
        out = client.chat(model, question, cfg.get("eval.sanity_max_tokens", int), cfg.get("vllm.temperature", float),
                          return_token_ids=False)
        print(f"Q: {question}\nA: {out['choices'][0]['message']['content']!r}\n", flush=True)


if __name__ == "__main__":
    main()
