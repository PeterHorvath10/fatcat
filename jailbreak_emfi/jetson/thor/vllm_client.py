"""Client for the local vLLM OpenAI-compatible server (one implementation for the workers, the
evaluation and the free run).

Requests are greedy chat completions: {"model", "messages": [user prompt], "max_tokens", "temperature"},
plus vLLM's "return_token_ids" so the generated token ids are recorded. The dev-mode routes
(/collective_rpc, /reset_prefix_cache, /is_sleeping) exist only when the server runs with
VLLM_SERVER_DEV_MODE=1 (vllm.weight_check = true in config).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from common import config
from common import http

# The model listing is a quick request even while vLLM is busy; never wait the full request timeout on it.
LIST_TIMEOUT_S = 10.0
# Extra time over the RPC's own timeout before the HTTP request gives up.
RPC_HTTP_MARGIN_S = 10.0


class VllmClient:
    def __init__(self, root_url: str, timeout_s: float, retry_s: float):
        self.root = root_url.rstrip("/")
        self.api = self.root + "/v1"
        self.timeout_s, self.retry_s = timeout_s, retry_s

    @classmethod
    def from_config(cls, cfg: config.Config) -> "VllmClient":
        return cls(f"http://{cfg.get('vllm.client_host')}:{cfg.get('vllm.port', int)}",
                   cfg.get("vllm.request_timeout_s", float), cfg.get("vllm.startup_retry_s", float))

    # ---- models --------------------------------------------------------------------------------------

    def served_models(self) -> dict[str, str | None]:
        """{served model name: checkpoint root} from /v1/models."""
        payload = http.get_json(self.api + "/models", min(self.timeout_s, LIST_TIMEOUT_S))
        models = payload.get("data")
        if not isinstance(models, list):
            raise ValueError(f"{self.api}/models returned an invalid model list")
        return {m["id"]: m.get("root") for m in models if isinstance(m, dict) and isinstance(m.get("id"), str)}

    def wait_for_model(self, accepted: Sequence[str], timeout_s: float = 0.0) -> str:
        """Block until vLLM serves one of accepted (0 = wait forever); return its name."""
        print(f"Waiting for one of: {', '.join(accepted)}", flush=True)

        def probe() -> str | None:
            served = self.served_models()
            for name in accepted:
                if name in served:
                    return name
            raise ValueError(f"available models: {sorted(served)}")

        model = http.poll("vLLM", probe, self.retry_s, timeout_s)
        print(f"vLLM model ready: {model}", flush=True)
        return model

    def model_root(self, model: str) -> Path:
        root = self.served_models().get(model)
        if not root:
            raise ValueError(f"{self.api}/models reports no root for {model}")
        return Path(root)

    # ---- generation ----------------------------------------------------------------------------------

    def chat(self, model: str, prompt: str, max_tokens: int, temperature: float,
             return_token_ids: bool = True) -> dict[str, Any]:
        body: dict[str, Any] = {"model": model, "messages": [{"role": "user", "content": prompt}],
                                "max_tokens": max_tokens, "temperature": temperature}
        if return_token_ids:
            body["return_token_ids"] = True
        result = http.post_json(self.api + "/chat/completions", body, self.timeout_s)
        try:
            result["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as error:
            raise ValueError(f"vLLM returned no chat message: {result!r}") from error
        return result

    def completion(self, model: str, prompt: str, max_tokens: int, temperature: float,
                   logprobs: int | None = None) -> dict[str, Any]:
        body: dict[str, Any] = {"model": model, "prompt": prompt, "max_tokens": max_tokens,
                                "temperature": temperature}
        if logprobs is not None:
            body["logprobs"] = logprobs
        return http.post_json(self.api + "/completions", body, self.timeout_s)

    # ---- dev-mode routes -----------------------------------------------------------------------------

    def dev_mode_enabled(self) -> bool:
        try:
            http.get_json(self.root + "/is_sleeping", min(self.timeout_s, LIST_TIMEOUT_S))
        except http.HTTPError as error:
            print(f"dev-mode check: HTTP {error.code} from {self.root}/is_sleeping; start vLLM with "
                  "VLLM_SERVER_DEV_MODE=1 (vllm.weight_check = true) to allow in-place reloads.", flush=True)
            return False
        except (http.URLError, TimeoutError) as error:
            print(f"dev-mode check: {self.root}/is_sleeping failed: {error}", flush=True)
            return False
        return True

    def reset_prefix_cache(self) -> None:
        """Drop vLLM's prefix cache (dev mode only; without dev mode nothing is cached across restarts)."""
        try:
            http.post(self.root + "/reset_prefix_cache", {}, self.timeout_s)
        except http.HTTPError:
            pass

    def collective_rpc(self, method: str, kwargs: dict | None = None, timeout_s: float | None = None) -> Any:
        """Run method on the vLLM worker (e.g. reload_weights or a weight_check_ext method); its result."""
        timeout_s = timeout_s or self.timeout_s
        body: dict[str, Any] = {"method": method, "timeout": timeout_s}
        if kwargs is not None:
            body["kwargs"] = kwargs
        payload = http.post_json(self.root + "/collective_rpc", body, timeout_s + RPC_HTTP_MARGIN_S)
        return payload["results"][0] if payload else None

def model_names(cfg: config.Config) -> list[str]:
    """Keys of all [models.*] sections."""
    return list(cfg.section("models"))


def served_names(cfg: config.Config) -> list[str]:
    return [cfg.get(f"models.{key}.served_name") for key in model_names(cfg)]


def model_units(cfg: config.Config) -> list[str]:
    return [cfg.get(f"models.{key}.unit") for key in model_names(cfg)]


def model_key_for(cfg: config.Config, served_name: str) -> str:
    for key in model_names(cfg):
        if cfg.get(f"models.{key}.served_name") == served_name:
            return key
    raise ValueError(f"no [models.*] section serves {served_name!r}")
