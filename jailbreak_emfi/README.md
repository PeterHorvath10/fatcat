# EM fault injection on Jetson GPUs to jailbreak LLMs: code

Code for the experiments and analyses of the paper, for a Jetson Orin Nano and a Jetson AGX Thor.

## Layout

| Folder | Runs on | Does | Docs |
|---|---|---|---|
| `common/` | everywhere | wire protocol between host and Jetson (`protocol.py`), settings loader (`config.py`), HTTP (`http.py`), Jetson temperature (`thermal.py`), settings for shell scripts (`cfgget.py`) | — |
| `host/` | the PC with the instruments | VC Glitcher, oscilloscope, XYZ stage, board resets, the trial loop; one entry script per experiment; the weight-reload hunt | `docs/host.md` |
| `jetson/` | Orin Nano / AGX Thor | the glitched workloads: memory check, vLLM/Ollama query workers, weight-reload worker with weight check, evaluation, systemd units | `docs/jetson.md` |
| `analysis/` | any PC | LLM judging and unsafe counts, scope-trace analysis (layer bursts, where the pulse lands), DRAM channel maps | `docs/analysis.md` |
| `config/` | — | all settings, as TOML: `host.toml`, `experiments/*.toml`, `thor.toml`, `orin.toml`, `analysis.toml` | — |
| `datasets/` | — | JailbreakBench harmful and benign behaviours (`jailbreakbench_harmful_prompts.txt` = the 100 harmful goals) | — |

## Settings

There are no setup values in the code. Every address, path, user, model, delay, pulse width, amplitude, scope
setting, threshold, token count and limit is read from `config/*.toml`, where **all values are shipped empty**
with a comment giving the unit and meaning. A script stops at start with the name of the first empty setting it
needs. Any setting can also be given as an environment variable `EMFI_<SECTION>__<KEY>` (e.g.
`EMFI_GLITCH__WINDOW_START_NS=<ns>`). Passwords are only read from environment variables whose names are in
the TOML.

Constants that are not settings (the wire protocol, 2 ns per VC Glitcher delay count, file formats) are named
once in the code.

## Running

Run everything from `jailbreak_emfi/` with `PYTHONPATH=$PWD` (on the Jetsons the systemd units set it):

```sh
PYTHONPATH=$PWD python3 host/run_thor_fixed_time.py --trials N --trials-per-prompt K   # an experiment
PYTHONPATH=$PWD python3 -m analysis.judge.count RUN_DIR                                    # an analysis
```

How one trial works:

```
 host PC                                        Jetson
 host/run_*.py  ── TCP connect ───────────────▶ worker (memfault / *_query_trigger / vllm_reload_trigger)
                ◀── COLLECT ─────────────────── worker is ready
 arm scope + VC Glitcher (delay, width, amplitude)
                ── ack: attempt, run id, [prompt slot, repeat, check-weights] ─▶
                                                worker runs the GPU work (memory check / prompt / weight reload)
 near-field probe → scope trigger → VC Glitcher fires the EM pulse
                ◀── [weight report] ─────────── (weight-reload experiments)
 log the trial, save the scope trace; power-cycle the board if it stopped answering
```

## Self-tests (no hardware, no models, no judges)

```sh
PYTHONPATH=$PWD python3 host/tests/selftest.py
PYTHONPATH=$PWD python3 -m jetson.selftest
PYTHONPATH=$PWD python3 -m analysis.selftest
```
