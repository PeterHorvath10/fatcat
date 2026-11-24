"""Zero-shot multiple-choice accuracy (MMLU subset and ARC-Challenge, mc_items.jsonl) of the served model.

The assistant turn is prefilled with "The correct answer is", so the next token is the answer; the
prediction is the most likely of A-D among the top logprobs. A corrupted model can return non-finite
logprobs, which vLLM rejects: those items count as "no letter".

    python3 -m jetson.thor.eval.mc_run OUT.jsonl [--items mc_items.jsonl] [--model SERVED_NAME]
"""

from __future__ import annotations

import argparse
import json
import math
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from common import config
from common import http
from jetson.thor.eval import served_model
from jetson.thor.vllm_client import VllmClient, model_key_for

ITEMS = Path(__file__).with_name("mc_items.jsonl")
LETTERS = "ABCD"
TOP_LOGPROBS = 20
ANSWER_PREFIX = "The correct answer is"
# Raw chat formats (vLLM adds the BOS for Gemma); selected by models.<key>.family.
TEMPLATES = {
    "gemma": "<start_of_turn>user\n{q}<end_of_turn>\n<start_of_turn>model\n" + ANSWER_PREFIX,
    "llama": ("<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{q}<|eot_id|>"
              "<|start_header_id|>assistant<|end_header_id|>\n\n" + ANSWER_PREFIX),
}
ERROR_DETAIL_BYTES = 200


def question(item: dict) -> str:
    about = f" about {item['subject'].replace('_', ' ')}" if item["subject"] else ""
    return (f"The following is a multiple choice question{about}.\n\n{item['question']}\n"
            + "".join(f"{letter}. {choice}\n" for letter, choice in zip(LETTERS, item["choices"])))


def ask(client: VllmClient, model: str, template: str, temperature: float, item: dict) -> dict:
    row = dict(bench=item["bench"], subject=item["subject"], answer=item["answer"])
    try:
        res = client.completion(model, template.format(q=question(item)), 1, temperature, logprobs=TOP_LOGPROBS)
    except http.HTTPError as error:
        return {**row, "pred": None, "letter_mass": 0.0,
                "error": f"{error.code}: {error.read()[:ERROR_DETAIL_BYTES].decode(errors='replace')}"}
    logprob: dict[str, float] = {}
    for token, value in res["choices"][0]["logprobs"]["top_logprobs"][0].items():
        key = token.strip()
        if key and key in LETTERS and key not in logprob:
            logprob[key] = value
    pred = max(logprob, key=logprob.get) if logprob else None
    return {**row, "pred": pred, "letter_mass": sum(math.exp(v) for v in logprob.values())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("out", type=Path)
    parser.add_argument("--items", type=Path, default=ITEMS)
    parser.add_argument("--model", help="served model name (default: the one vLLM serves)")
    args = parser.parse_args()
    cfg = config.load("thor.toml")
    client = VllmClient.from_config(cfg)
    model = served_model(cfg, client, args.model)
    template = TEMPLATES[cfg.get(f"models.{model_key_for(cfg, model)}.family")]
    temperature = cfg.get("vllm.temperature", float)
    items = [json.loads(line) for line in args.items.read_text().splitlines() if line.strip()]

    with ThreadPoolExecutor(cfg.get("eval.mc_parallel_requests", int)) as pool:
        rows = list(pool.map(lambda item: ask(client, model, template, temperature, item), items))
    args.out.write_text("".join(json.dumps(r) + "\n" for r in rows))
    for bench in ("mmlu", "arc_c"):
        rs = [r for r in rows if r["bench"] == bench]
        if rs:
            print(f"{bench}: {sum(r['pred'] == r['answer'] for r in rs) / len(rs) * 100:.1f}% of {len(rs)}, "
                  f"no letter {sum(r['pred'] is None for r in rs)}, errors {sum('error' in r for r in rs)}, "
                  f"mean letter mass {sum(r['letter_mass'] for r in rs) / len(rs):.2f}")


if __name__ == "__main__":
    main()
