# Jetson side: install and run

Everything under `jetson/` runs on the boards; nothing there talks to the instruments. The host connects
to the board's worker over TCP (`common/protocol.py`): the worker is the server, sends COLLECT when it is
ready, the host answers with the trial's attempt index, run UUID and (prompt-slot form) prompt slot,
repeat and check-weights flag, then the worker runs the GPU work that the scope sees and the glitcher hits.
Only one worker may own the trial port at a time.

```
jetson/
  common/        prompts.py  results.py  thermal.py  systemd.py  gpio.py   (shared by both boards; HTTP is ../common/http.py, shared with host/analysis)
  install.py     renders the systemd units from config/<board>.toml
  selftest.py    tests without hardware: PYTHONPATH=<jailbreak_emfi> python3 -m jetson.selftest
  thor/          vllm_client.py  serve_vllm.py  vllm_query_trigger.py  vllm_reload_trigger.py
                 weight_check_ext.py  jbb_freerun.py
                 eval/        jbb_run.py  mc_run.py (+ mc_items.jsonl)  eval_model.py
                 gpu_memfault/ gpu_memfault.cu  Makefile  launch.py  README.md
                 systemd/     *.in templates, jetson-clocks.service      bin/lock_gpu_clock.sh
  orin/          memfault.py  memcheck.py  memcheck_kernels.cu  cuda_driver.py  pa_capture.py
                 ollama_query_trigger.py  ollama_client.py
                 orin_gpu_pa/ (kernel module + query tool, see its README/HOW_IT_WORKS)
                 systemd/     *.in templates      bin/load-orin-gpu-pa
```

All scripts are run as modules with `PYTHONPATH=<jailbreak_emfi>` (the units set it), and read their settings
from `config/thor.toml` or `config/orin.toml`. Every value there is empty until you fill it in; a missing
value stops the script with the key's name.

## Install

### Thor
1. JetPack 7 (CUDA 13). vLLM 0.26.0 in a virtualenv (`vllm.venv`); the workers use the system Python 3.12
   (`system.python`, standard library only). Models (Hugging Face checkpoints) in `models.<key>.dir`.
   JailbreakBench prompt files (`prompts.file`, `prompts.jbb_csv`).
2. Copy `jailbreak_emfi/` to Thor (`system.paper_code`), fill in `config/thor.toml`.
3. Render and install the units:
   ```sh
   PYTHONPATH=$PWD python3 -m jetson.install thor /tmp/units
   sudo cp /tmp/units/*.service /etc/systemd/system/
   sudo visudo -cf /tmp/units/vllm-restart.sudoers && sudo install -m 0440 /tmp/units/vllm-restart.sudoers /etc/sudoers.d/vllm-restart
   sudo systemctl daemon-reload
   sudo systemctl enable --now jetson-clocks.service gpu-clock-lock.service
   ```
   The sudoers rule lets the query worker restart a model unit (corruption watchdog) without a password.
4. GPU memory check: `cd jetson/thor/gpu_memfault && make ARCH=<memfault.cuda_arch> NVCC=<memfault.nvcc>`
   (`memfault.binary` points at the result).

The serving mode is configuration, not a separate unit: after changing `[vllm]` (prefix caching,
quantization, weight_check) restart the model unit.

Thor can overheat under sustained GPU load: `[thermal]` makes the free run, the
evaluation pause; the host controller paces the EMFI trials.

### Orin Nano
1. JetPack with CUDA, Python 3.10 with numpy and cuda-python, libgpiod Python bindings, the user in the
   `video` and `gpio` groups. Ollama with the models (`ollama.model`).
2. Build the physical-address probe against the running kernel: `cd jetson/orin/orin_gpu_pa && make`;
   set `memfault.pa_module` / `memfault.pa_query_tool` to the built files.
3. Copy `jailbreak_emfi/` (`system.paper_code`), fill in `config/orin.toml`, then
   ```sh
   PYTHONPATH=$PWD python3 -m jetson.install orin /tmp/units
   sudo install -m 0755 jetson/orin/bin/load-orin-gpu-pa /usr/local/sbin/
   sudo cp /tmp/units/*.service /etc/systemd/system/ && sudo systemctl daemon-reload
   sudo systemctl enable --now jetson-clocks.service
   ```

## Experiments

### 1. Locating sensitive spots (memory check)
- **Thor:** `sudo systemctl disable --now vllm-query-trigger vllm-reload-trigger <model units>`, then
  `sudo systemctl start gpu-memfault` (ready at "waiting for controller..."). Settings:
  `[memfault]`. Output: `memfault.log` (+ `_ecc.tsv`, `_win.tsv`). Stopping a large buffer takes over a
  minute; let systemd wait (TimeoutStopSec=300 s). See `jetson/thor/gpu_memfault/README.md`.
- **Orin:** `sudo systemctl enable --now orin-gpu-pa memfault` (stops Ollama: `Conflicts=`). At start the
  worker allocates the buffers, records their physical pages under `memfault.map_dir`, warms up, then
  accepts the host. Logs in `memfault.log_dir`.

### 2. Weight-reload faults (Thor, fp8)
1. `[vllm]`: `quantization = "fp8"`, `weight_check = true`, then
   `sudo systemctl restart <model unit>`.
2. Fill in `[reload_trigger]` (`reload_only`, `baseline_from_reload`, `stop_on_change`, ...), then
   `sudo systemctl start vllm-reload-trigger`. At start it takes the weight baseline and checks that clean
   reloads match it; then every trial is one in-place reload, followed by a weight report when the host
   asks for it. Non-MATCH reports go to `reload_trigger.weight_report_log`.
3. With `stop_on_change` the worker exits (status 3, unit stays down) after a CHANGED report, leaving the
   corrupted weights in place for the evaluation:
   ```sh
   python3 -m jetson.thor.eval.jbb_run OUT.jsonl          # 100 harmful goals, eval.jbb_max_tokens
   python3 -m jetson.thor.eval.mc_run OUT_mc.jsonl        # MMLU subset + ARC-Challenge accuracy
   python3 -m jetson.thor.eval.eval_model                 # three sanity questions
   ```
   (`cd jailbreak_emfi; export PYTHONPATH=$PWD` first; the host's hunt script runs these over ssh.) The next
   reload (restart the worker) restores the weights.

`--free-run [--weight-check]` runs reload trials back to back without a host (scope setup, timing).

### 3. Inference-time faults
- **Thor (vLLM):** `[vllm]`: `quantization = "none"`, `weight_check = false`, `prefix_caching = false`;
  start the model unit and `vllm-query-trigger`. Each trial answers line `prompt_slot` of `prompts.file`
  (max `query_trigger.max_tokens` tokens, greedy) and appends a record to `query_trigger.output`. If an
  answer repeats `reload_token` at least `reload_threshold` times, the model unit is restarted (fresh
  weights) before the next trial.
- **No-pulse baseline:** with the same server and no glitching,
  `python3 -m jetson.thor.jbb_freerun OUT.jsonl` (resumable; `--max-tokens` for the short variant).
- **Orin (Ollama):** `sudo systemctl enable --now ollama ollama-query-trigger`. The worker waits for the
  model, loads it, finds Ollama's llama-server runner and queries it directly; GPIO `ollama.gpio_line` is
  high during each request. Answers with `ollama.reload_token` or only question marks trigger a model reload.

## Checks
```sh
cd jailbreak_emfi && PYTHONPATH=$PWD python3 -m jetson.selftest
```
Runs without GPU, board or network: protocol round trips with a fake host over a socketpair, the query and
reload workers against a fake vLLM server, the Ollama runner client, log formats, unit rendering, and the
config error for empty values.
