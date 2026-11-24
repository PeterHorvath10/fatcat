"""Start the vLLM OpenAI-compatible server for one model (ExecStart of the vllm-<model> units).

One script for every model and serving mode, which are set in [vllm] and [models.*] of config/thor.toml:

    python3 -m jetson.thor.serve_vllm --model gemma3_1b [extra vllm serve arguments]

The paper's JailbreakBench runs: bf16 checkpoint, prefix caching off, --enforce-eager, temperature 0.
The fp8 weight-reload runs: --quantization fp8 (vLLM quantizes the bf16 checkpoint on load) and
weight_check = true, which loads jetson.thor.weight_check_ext into the worker and enables vLLM's dev-mode
routes (POST /collective_rpc for reload_weights and the weight checks).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from common import config

PAPER_CODE = Path(__file__).resolve().parents[2]
WORKER_EXTENSION = "jetson.thor.weight_check_ext.WeightCheckWorkerExtension"
# vLLM's quantization value for "serve the checkpoint's own dtype".
NO_QUANTIZATION = "none"


def vllm_command(cfg: config.Config, model_key: str, extra: list[str]) -> tuple[list[str], dict[str, str]]:
    venv = Path(cfg.get("vllm.venv"))
    cuda = Path(cfg.get("vllm.cuda_dir"))
    temperature = cfg.get("vllm.temperature", float)
    command = [
        str(venv / "bin" / "vllm"), "serve", cfg.get(f"models.{model_key}.dir"),
        "--served-model-name", cfg.get(f"models.{model_key}.served_name"),
        "--host", cfg.get("vllm.host"),
        "--port", str(cfg.get("vllm.port", int)),
        "--tensor-parallel-size", "1",
        "--gpu-memory-utilization", str(cfg.get("vllm.gpu_memory_utilization", float)),
        "--enable-prefix-caching" if cfg.get("vllm.prefix_caching", bool) else "--no-enable-prefix-caching",
        "--override-generation-config", json.dumps({"temperature": temperature}),
        "--trust-remote-code",
    ]
    if cfg.get("vllm.enforce_eager", bool):
        command.append("--enforce-eager")
    quantization = cfg.get("vllm.quantization")
    if quantization != NO_QUANTIZATION:
        command += ["--quantization", quantization]

    env = dict(os.environ)
    env["PATH"] = f"{cuda / 'bin'}:{venv / 'bin'}:{env.get('PATH', '')}"
    env["LD_LIBRARY_PATH"] = str(cuda / "lib64") + (f":{env['LD_LIBRARY_PATH']}" if env.get("LD_LIBRARY_PATH") else "")
    env["TRITON_PTXAS_PATH"] = str(cuda / "bin" / "ptxas")
    if cfg.get("vllm.weight_check", bool):
        command += ["--worker-extension-cls", WORKER_EXTENSION]
        env["VLLM_SERVER_DEV_MODE"] = "1"
        # The worker process imports the extension (and common.config) from this folder (jailbreak_emfi/).
        env["PYTHONPATH"] = str(PAPER_CODE) + (f":{env['PYTHONPATH']}" if env.get("PYTHONPATH") else "")
    return command + extra, env


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", required=True, help="key of a [models.*] section in config/thor.toml")
    args, extra = parser.parse_known_args()
    command, env = vllm_command(config.load("thor.toml"), args.model, extra)
    print("exec:", " ".join(command), flush=True)
    os.execve(command[0], command, env)


if __name__ == "__main__":
    main()
