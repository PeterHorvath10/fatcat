"""Ollama on the Orin: model lifecycle through the Ollama API, generation straight from the resident
llama-server runner.

The trials bypass Ollama's own generate API and post to the runner's /completion endpoint (as in the paper's
runs; unlike Ollama's OpenAI-compatible API it honours cache_prompt). The prompt therefore has to be
wrapped in the model's chat turn format here (ollama.prompt_template).
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from common import config
from common import http

RUNNER_PROCESS = "llama-server"
LIST_TIMEOUT_S = 10.0
NS_PER_MS = 1_000_000
TEMPLATE_FIELD = "{prompt}"


def keep_alive_value(value: str) -> str | int:
    """Ollama takes an int (seconds, -1 = forever) or a duration string."""
    try:
        return int(value)
    except ValueError:
        return value


class OllamaClient:
    def __init__(self, cfg: config.Config):
        self.base_url = f"http://{cfg.get('ollama.host')}:{cfg.get('ollama.port', int)}"
        self.model = cfg.get("ollama.model")
        self.keep_alive = keep_alive_value(str(cfg.get("ollama.keep_alive")))
        self.timeout_s = cfg.get("ollama.request_timeout_s", float)
        self.startup_timeout_s = cfg.get("ollama.startup_timeout_s", float)
        self.retry_s = cfg.get("ollama.startup_retry_s", float)
        self.runner_host = cfg.get("ollama.runner_host")
        self.runner_port_setting = cfg.get("ollama.runner_port", int)     # 0 = find the running llama-server
        self.template = cfg.get("ollama.prompt_template")
        if TEMPLATE_FIELD not in self.template:
            raise config.ConfigError(f"ollama.prompt_template must contain {TEMPLATE_FIELD}")
        self.max_tokens = cfg.get("ollama.max_tokens", int)
        self.temperature = cfg.get("ollama.temperature", float)
        self.seed = cfg.get("ollama.seed", int)
        self.cache_prompt = cfg.get("ollama.cache_prompt", bool)
        self.runner_port: int | None = None

    # ---- Ollama API: model lifecycle -----------------------------------------------------------------

    def available_models(self) -> set[str]:
        payload = http.get_json(f"{self.base_url}/api/tags", min(self.timeout_s, LIST_TIMEOUT_S))
        models = payload.get("models")
        if not isinstance(models, list):
            raise ValueError(f"{self.base_url}/api/tags returned an invalid model list")
        return {m[k] for m in models if isinstance(m, dict) for k in ("name", "model") if isinstance(m.get(k), str)}

    def wait_for_model(self) -> None:
        def probe():
            models = self.available_models()
            if self.model in models:
                return True
            raise ValueError(f"available models: {sorted(models)}")

        http.poll("Ollama", probe, self.retry_s, self.startup_timeout_s)
        print(f"Ollama model ready: {self.model}", flush=True)

    def _lifecycle(self, keep_alive: str | int, timeout_s: float) -> dict:
        """An empty generate request: loads the model (keep_alive) or unloads it (keep_alive 0)."""
        result = http.post_json(f"{self.base_url}/api/generate",
                                {"model": self.model, "prompt": "", "stream": False, "keep_alive": keep_alive},
                                timeout_s)
        if result.get("model") != self.model or result.get("done") is not True:
            raise ValueError("Ollama returned an invalid load/unload response")
        return result

    def warm_up(self) -> None:
        """Load the model before the trial socket opens."""
        result = http.poll(f"Ollama warm-up of {self.model}", lambda: self._lifecycle(self.keep_alive, self.timeout_s),
                           self.retry_s, self.startup_timeout_s)
        print(f"Ollama model warm-up complete: {self.model} (load_duration_ns={result.get('load_duration')})",
              flush=True)

    def unload(self) -> None:
        self._lifecycle(0, self.timeout_s)
        print(f"Ollama model unload complete: {self.model}", flush=True)

    def reload(self) -> None:
        """Fresh weights from disk: unload, load, and find the new runner."""
        self.unload()
        self.warm_up()
        self.wait_for_runner()

    # ---- llama-server runner: generation -------------------------------------------------------------

    @staticmethod
    def find_runner_port() -> int | None:
        """--port of the local llama-server process Ollama started for the model."""
        for process in Path("/proc").glob("[0-9]*"):
            try:
                command = (process / "cmdline").read_bytes().split(b"\0")
            except (OSError, ValueError):
                continue
            if not command or not command[0] or Path(os.fsdecode(command[0])).name != RUNNER_PROCESS:
                continue
            for i, argument in enumerate(command[:-1]):
                if argument == b"--port":
                    try:
                        return int(command[i + 1])
                    except ValueError:
                        return None
        return None

    def wait_for_runner(self) -> int:
        def probe():
            port = self.runner_port_setting or self.find_runner_port()
            if port is None:
                raise ValueError(f"{RUNNER_PROCESS} was not found")
            http.get(f"http://{self.runner_host}:{port}/health", min(self.timeout_s, LIST_TIMEOUT_S))
            return port

        self.runner_port = http.poll(RUNNER_PROCESS, probe, self.retry_s, self.startup_timeout_s)
        print(f"{RUNNER_PROCESS} ready: {self.runner_host}:{self.runner_port}", flush=True)
        return self.runner_port

    def complete(self, prompt: str) -> dict[str, Any]:
        """One greedy completion; the result in the fields of Ollama's /api/generate response."""
        result = http.post_json(f"http://{self.runner_host}:{self.runner_port}/completion",
                                {"prompt": self.template.replace(TEMPLATE_FIELD, prompt), "n_predict": self.max_tokens,
                                 "temperature": self.temperature, "seed": self.seed, "cache_prompt": self.cache_prompt},
                                self.timeout_s)
        normalized = normalize_runner_result(self.model, result)
        if normalized is None:
            raise ValueError(f"{RUNNER_PROCESS} returned no content: {result!r}")
        return normalized


def normalize_runner_result(model: str, result: Any) -> dict[str, Any] | None:
    """Map llama-server /completion fields to Ollama's generate response fields (record schema)."""
    if not isinstance(result, dict) or not isinstance(result.get("content"), str):
        return None
    timings = result.get("timings") if isinstance(result.get("timings"), dict) else {}
    prompt_ms = timings.get("prompt_ms", 0)
    predicted_ms = timings.get("predicted_ms", 0)
    prompt_ms = prompt_ms if isinstance(prompt_ms, (int, float)) else 0
    predicted_ms = predicted_ms if isinstance(predicted_ms, (int, float)) else 0
    return {
        "model": model,
        "response": result["content"],
        "context": None,
        "done": True,
        "done_reason": "stop" if result.get("stop") else "length",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "total_duration": int((prompt_ms + predicted_ms) * NS_PER_MS),
        "load_duration": 0,
        "prompt_eval_count": result.get("tokens_evaluated", timings.get("prompt_n")),
        "prompt_eval_duration": int(prompt_ms * NS_PER_MS),
        "eval_count": result.get("tokens_predicted", timings.get("predicted_n")),
        "eval_duration": int(predicted_ms * NS_PER_MS),
        "llama_server_timings": timings,
    }
