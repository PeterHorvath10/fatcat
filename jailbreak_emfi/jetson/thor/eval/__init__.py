"""Evaluation of the (corrupted) model on Thor: JailbreakBench answers, MMLU/ARC-C accuracy, sanity prompts."""

from __future__ import annotations

from common import config
from jetson.thor.vllm_client import VllmClient, served_names


def served_model(cfg: config.Config, client: VllmClient, requested: str | None) -> str:
    """The model to evaluate: the requested served name, else whichever configured model vLLM serves."""
    return client.wait_for_model([requested] if requested else served_names(cfg))
