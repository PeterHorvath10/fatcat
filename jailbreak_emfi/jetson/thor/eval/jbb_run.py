"""Answer the 100 harmful JailbreakBench goals with the served model (run on Thor after a weight-reload hit;
the host judges the answers). One JSON line per goal.

    python3 -m jetson.thor.eval.jbb_run OUT.jsonl [--model SERVED_NAME]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from common import config
from jetson.common import prompts, results
from jetson.common.thermal import ThermalGuard
from jetson.thor.eval import served_model
from jetson.thor.vllm_client import VllmClient


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("out", type=Path)
    parser.add_argument("--model", help="served model name (default: the one vLLM serves)")
    args = parser.parse_args()
    cfg = config.load("thor.toml")
    client = VllmClient.from_config(cfg)
    model = served_model(cfg, client, args.model)
    guard = ThermalGuard.from_config(cfg)
    max_tokens = cfg.get("eval.jbb_max_tokens", int)
    temperature = cfg.get("vllm.temperature", float)
    rows = prompts.load_jbb_csv(Path(cfg.get("prompts.jbb_csv")))

    client.reset_prefix_cache()           # answers must come from the current weights, not cached KV
    args.out.unlink(missing_ok=True)
    for row in rows:
        guard.wait()
        choice = client.chat(model, row["Goal"], max_tokens, temperature, return_token_ids=False)["choices"][0]
        results.append_jsonl(args.out, {"index": int(row["Index"]), "category": row["Category"],
                                        "behavior": row["Behavior"], "goal": row["Goal"],
                                        "response": choice["message"]["content"],
                                        "finish": choice["finish_reason"]})
    print("done", len(rows))


if __name__ == "__main__":
    main()
