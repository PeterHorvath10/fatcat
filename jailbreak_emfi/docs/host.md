# Host PC: running the EMFI experiments

Everything here runs on the host PC that drives the instruments (VC Glitcher, oscilloscope, XYZ stage) and
talks to the Jetson worker over TCP. The Jetson side (workers, vLLM/Ollama, evaluation) is in `jetson/`.

## Setup

- Python 3.10, `pip install -r host/requirements.txt`, plus the vendor packages: Keysight VC Glitcher SDK
  (`vcglitchersdk`) and Riscure Inspector's Python API (`riscure.inspector`). Put both on `PYTHONPATH`
  (or install them in the venv). They are imported only when an instrument is used, so the self-test runs
  without them.
- Fill in `config/host.toml` (instruments, network, boards) and the experiment TOML you run. Every value is
  empty in the shipped files; a missing value stops the script at start with the key to set.
  Any value can be overridden per run by an environment variable
  `EMFI_<SECTION>__<KEY>` (e.g. `EMFI_SCOPE__CAPTURE_DIR=run1/scope`).
- Passwords only come from environment variables, whose *names* are in the TOML:
  `network.sudo_password_env` (this PC's sudo, for `boardctl` and `ip` after a reset) and, for the hunt,
  `hunt.sudo_password_env` (the Thor's sudo).
- Instruments: the near-field probe over the GPU feeds the scope channel `scope.channel`; the scope's
  interval trigger on it (settings in `[scope]`, or `scope.apply_settings = true` to write them) goes to the
  VC Glitcher's trigger input via the scope's Aux output. Thor needs an ssh alias (`boards.thor.ssh_host`)
  for thermal pacing.
- Run everything from `jailbreak_emfi/` with `PYTHONPATH=$PWD` (add the vendor SDK folders as needed).

## Experiments

| Experiment | Jetson worker | TOML | Command |
|---|---|---|---|
| Orin XYZ scan (memory check or JailbreakBench) | Orin `memfault` / `ollama_query_trigger` | `experiments/orin_xyz.toml` | `python host/run_orin_xyz.py` |
| Orin JailbreakBench at one position | Orin `ollama_query_trigger` | `experiments/orin_fixed_time.toml` | `python host/run_orin_fixed_time.py --position-index N` |
| Thor memory-check XYZ scan | Thor `gpu-memfault` | `experiments/thor_memfault_xyz.toml` | `python host/run_thor_memfault_xyz.py --stops XxY --trials N` |
| Thor JailbreakBench, inference time | Thor `vllm-query-trigger` + vLLM | `experiments/thor_fixed_time.toml` | `python host/run_thor_fixed_time.py --trials N --trials-per-prompt K --trial-log RUN/trials.csv` |
| Thor weight-reload hunt (wipe one layer's `down_proj` during the reload, then evaluate) | Thor `vllm-reload-trigger` + vLLM | `experiments/thor_weight_reload_hunt.toml` | `host/weight_reload/hunt_layers.sh` |

Only one Jetson worker may own the worker port at a time.

Common options of the `run_*.py` scripts (everything else is in the TOMLs):
`--experiment TOML`, `--trials N` (per position), `--trials-per-prompt N`, `--start-prompt SLOT`,
`--trial-log CSV`, `--collection-id UUID`, `--stay` (never move the stage), `--position-index I`,
`--indices LIST|FILE`, `--start-index I` (resume a scan), `--stops XxY` (grid instead of the dialog's).

Dialog-mode stages (`stage.mode = "dialog"`) open Inspector's parameter dialog at start to choose the stage
and grid. `python host/goto_xyz.py X Y [Z]` moves the probe to one position (e.g. before a `--stay` run).
`python -m host.emfi.reset experiments/<exp>.toml` power-cycles that experiment's board by hand.

## What one trial does

1. Wait for the worker's COLLECT (`common/protocol.py`).
2. Draw the trial's pulse: amplitude and width at random from the lists, delay by `glitch.delay_mode`
   (`window` or `random`); one pulse per trial.
3. Load the pattern, arm the scope (single) and the glitcher, wait `trigger.arm_to_ack_s`, send the
   acknowledgement (`target.ack_form`: `short`, `slot` = + prompt slot/repeat, `slot_weights` = + check-weights).
4. `trigger.source = "scope"`: the scope edge starts the pattern (software start after `trigger.timeout_s`
   without an edge, so the trial still ends); `"software"`: start `trigger.software_delay_s` after the ack.
5. With `target.weight_report`, read and print the worker's weight report.
6. Log the trial (`trial_parameters` CSV). With `scope.capture`,
   save `t<trial>.png` and `t<trial>.npz` (raw channel codes, WAVEDESC preamble, trial metadata).

If the connection dies (hang/crash), the trial is logged with `reset_hit = True`, the board is power-cycled
(`boards.<board>.reset_method`), the Ethernet link is set up again and the run continues (a scan with
`run.next_position_on_reset = true` moves to the next position). Before each trial: `run.pause_s`, an extra
`run.cool_s` every `run.cool_every` trials, and thermal pacing if `thermal.enabled`.

## Weight-reload hunt

`hunt_layers.sh [EXPERIMENT_TOML]` reads all settings through `python -m common.cfgget`. Per layer in
`hunt.layers`: rounds of `hunt.round_trials` controller trials with the window from `window.py` (passed as
`EMFI_GLITCH__WINDOW_START_NS/END_NS`); `until_hit.py` stops a round at the first CHANGED weight report;
`hit_delay.py` appends the hit to `hunt.hits_csv`; `is_full_wipe.py` accepts only a full wipe of the
target `down_proj`; then the Thor evaluation (`eval_model.py`, `jbb_run.py`, `mc_run.py` in
`hunt.remote_paper_code`, modules `jetson.thor.eval.*`) runs over ssh and the results are copied to `hunt.out_dir/L<layer>/`.

## Files

```
host/
  run_orin_xyz.py, run_orin_fixed_time.py, run_thor_memfault_xyz.py, run_thor_fixed_time.py   entry points
  goto_xyz.py                    move the stage
  emfi/
    runner.py      the trial loop + building an experiment from the config
    cli.py         shared command line
    glitcher.py    VC Glitcher (NS_PER_COUNT = 2), GlitchParams
    scope.py       oscilloscope: open/arm/read waveform/save capture/trigger state (also used by analysis/)
    target.py      worker connection, acknowledgement forms, weight report
    plan.py        per-trial pulse parameters and prompt slots
    delays.py      delay strategies (window, random)
    reset.py       Orin Arduino / Thor boardctl reset, Ethernet re-up
    thermal.py     thermal pacing over ssh
    triallog.py    trial CSV
    stage.py       Inspector XYZ stage
  weight_reload/   hunt_layers.sh, window.py, until_hit.py, hit_delay.py, is_full_wipe.py, sensible.py, hits/
  board/arduino_ctr/arduino_ctr.ino   Orin reset Arduino sketch (J14 header)
  tests/selftest.py
```

## Self-test (no hardware)

`PYTHONPATH=$PWD python3 host/tests/selftest.py`: config errors on the empty TOMLs, every key present
(dummy-filled TOMLs build), the three acknowledgement forms against the worker-side protocol functions,
delay strategies, WAVEDESC parsing, and the whole trial loop against a fake worker over
a socketpair (weight reports, a dropped connection -> reset path, CSV).
