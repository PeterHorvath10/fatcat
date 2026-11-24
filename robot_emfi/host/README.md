# Host PC

Everything in this folder runs on the host PC. It has three kinds of scripts:
- **Campaigns:** drive the ChipWhisperer-Husky and the ChipSHOUTER, and start the Jetson side over ssh.
- **Simulation:** runs the SmolVLA policy in PyTorch on a local CUDA GPU.
- **Analysis:** reads the logs; no hardware needed.

Run every script from this folder (`cd host`). The scripts import each other and `site_config.py`.

## Setup

1. **Two Python environments** (the versions are the ones we used):

   | For | Packages |
   |---|---|
   | Campaigns (hardware) | Python 3.12, `chipwhisperer` 6.0.0, `chipshouter` 1.1.0, `numpy` 1.26, `pyserial` 3.5, `matplotlib` 3.11 |
   | Simulation | Python 3.12, PyTorch 2.7.1 (CUDA 12.8), LeRobot 0.5.1 with SmolVLA (`transformers` 5.3.0), `numpy` 2.2 |

   The analysis scripts only need `numpy`, so either environment runs them. A FaultyCat instead of the ChipSHOUTER (`fi_v3_run.py --emfi faultycat`) also needs its `faultycat` and `faultycmd` packages.
2. **ssh:** set up key-based ssh to the Jetson (`ssh-copy-id user@jetson`). The scripts call ssh in batch mode and never answer a password prompt.
3. **Settings:** edit `site_config.py`, or put your values in a `site_local.py` next to it, which is not part of the repository:
   - `JETSON`: the ssh target.
   - `JETSON_DIR`: an absolute folder on the Jetson that receives `../jetson` and `../data`.
   - `AOT_PYTHON`: the TensorRT runtime's Python on the Jetson.
   - `CONDA_ACTIVATE`: the command that activates the LeRobot environment on the Jetson.
   - `SHOUTER_PORT`: found automatically under `/dev/serial/by-id`.
   - `ARDUINO_PORT`: optional, the reset line.
   - `SMOLVLA_CKPT`: the fine-tuned checkpoint, for the simulation.
4. **USB reset (optional):** `fi_v3_run.py` can rebind the Jetson's USB controller when a pulse kills it (`sudo -n`, never a password). This needs a sudoers rule on the Jetson for the `tegra-xusb` bind/unbind writes. Without the rule, such a crash stops the campaign.

## Scripts

| Script | Kind | What it does | Needs |
|---|---|---|---|
| `shouter.py` | campaign | ChipSHOUTER driver; run alone, it prints the injector's state and never arms | ChipSHOUTER |
| `probe_monitor.py` | campaign | Live probe trace, to place the sense probe and choose a trigger level (no high voltage) | Husky; `jetson/probe_load.py` running |
| `gpumem_run.py` | campaign | GPU memory read/write test under EM pulses (`jetson/gpumem.cu`) | Husky, ChipSHOUTER, Jetson |
| `fi_v3_run.py` | campaign | EM pulses into the live rollout's inferences; `--move` lets the arm execute the plans | Husky, ChipSHOUTER, Jetson, arm |
| `arm_load_run.py` | campaign | EM pulses into the live rollout's weight load; the arm runs only an erratic policy | Husky, ChipSHOUTER, Jetson, arm |
| `live_view.py` | campaign | Live window with each shot's probe trace and outcome (`--live` file of the campaigns) | none |
| `husky.py` | helper | Husky set-up of the probe input and the glitch output, and the delay lists; the campaigns import it | none |
| `swfi2.py` | simulation | One garbled 16-byte block per inference in the PyTorch policy (Section 6.3) | CUDA GPU, checkpoint |
| `swfi2_analyze.py` | analysis | Outcome per fault source (Table 6) | none |
| `trt_blocksim_compare.py` | analysis | The block simulation on the Jetson's TensorRT engines (`jetson/trt_blocksim.py`) against `swfi2.py`: Table 6 and the numbers of Section 6.3 | none |
| `swfi2_em.py` | analysis | EM pulses against the simulation, per stage (Table 8) | none |
| `pos_analyze.py` | analysis | Outcome rates of an inference campaign, by where each pulse landed (Table 7) | none |
| `move_analyze.py` | analysis | Each pulse of a moving-arm episode and what the arm executed, against the fault-free replay (Section 6.4) | ssh to fetch session logs |
| `gpumem_analyze.py` | analysis | What each corrupted memory word looked like (garbled 16-byte blocks, bit flips, ...) | none |
| `arm_load_summary.py` | analysis | Rollout load campaigns: pulses, crashes, corrupted loads, plan classes (Section 6.6) | none |
| `arm_motion.py` | analysis | What the arm executed in a faulted episode (Section 6.6) | none |
| `extract_observations.py` | data | Cuts the 8 reference observations out of logged rollouts into `../data/traces` | none |

## Running the campaigns

**Before every campaign:**
- Check that the Jetson runs with fixed clocks (`jetson/README.md`).
- Run the campaign once with `--char`. That fires nothing and checks the trigger and the timing.

**Outputs:** each campaign writes `OUT.jsonl` (one row per shot), `OUT_rms.npz` or `OUT_env.npz` (probe traces) and `OUT.console`. Create the output folder first.

**Stopping:** stop with one SIGINT to the python process (see the [safety notes](../README.md#safety)), then check the injector with `python shouter.py`.

### Probe placement and the memory test (Section 4.3)

1. On the Jetson, run `probe_load.py`. It repeatedly reloads the expert's engine, which gives the probe a steady burst to look at.
2. On the host, run `python probe_monitor.py` and move the sense probe for the largest burst.
3. Place the injection tip, then check it with the memory read test. An `--live` file feeds the optional live view:

```
mkdir -p out/gpumem
python gpumem_run.py out/gpumem/char --char --shots 20
python gpumem_run.py out/gpumem/check300 --mode read --voltage 300 --width-ns 80 --delays 10:90:10 --shots 3 \
    --live out/gpumem/check300_live.npz
python live_view.py out/gpumem/check300_live.npz out/gpumem/check300.jsonl     # optional, in a second terminal
python gpumem_analyze.py out/gpumem/check300.jsonl
```

**What to look for:** a good spot garbles at least one 16-byte block in most read pulses. The coupling depends more on the exact tip spot than on the voltage, and it can drift within hours, so re-check before trusting new robot data.

### EM pulses during inference (Section 6.4)

The Husky triggers on the probe's signal at the start of an inference, then fires after `--delays` ms. With our probe the pulse lands about 1.8 ms later in the inference than the delay. Approximate delays:
- side-camera encoder: 30–59 ms;
- prefill: 59–68 ms;
- last denoising step: 106–111 ms.

`pos_analyze.py` reports where each pulse actually landed.

```
mkdir -p out/v3fi
# trigger level: no high voltage, the arm holds still
python fi_v3_run.py out/v3fi/char --char --levels=-0.18,-0.20,-0.22 --shots 25
# one 30 s episode with the arm moving; reset the scene between episodes
python fi_v3_run.py out/v3fi/prefill_e1 --move --nan-continue --sessions 1 --session-s 30 \
    --delays 59:68:0.25 --shots 1 --level=-0.20 --voltage 300 --width-ns 80 --seed 1
```

**Options:**
- `--nan-continue`: a NaN plan does not end the episode; the arm keeps the previous plan. This is the paper's convention; the stock rollout stops instead.
- `--no-limits`: runs without the rollout's slew limit and first-action check.
- Without `--move`, every motor command is dropped and logged, and the arm holds its pose.

### EM pulses during the live rollout's weight load (Section 6.6)

Each attempt starts the rollout and fires one pulse into the expert's weight copy. The arm moves only if the faulted policy is erratic on the 8 reference observations. Attempts with any other outcome are restarted, without motion.

```
mkdir -p out/arm
# once, no high voltage: fault-free reference plans and decode weights (saved on the Jetson)
python arm_load_run.py out/arm/golden --char --attempts 1 --capture-decode-golden
# check the pipeline, no high voltage and no motion
python arm_load_run.py out/arm/check --char --attempts 3
# campaign: 120 s episodes on an erratic load
python arm_load_run.py out/arm/run1 --voltage 500 --width-ns 80 --delays 40:56:2 --attempts 40 --duration 120 \
    --nan-drop --live out/arm/run1_live.npz
```

**Replaying a kept fault:** `--char --test-plant out/arm/RUN_fault.npz` writes the bytes of an erratic load into a clean load, with no pulse. It then runs the same episode again.

**Paper conventions:** `--nan-drop` keeps the previous plan on a NaN plan. Without it, the stock rollout stops at the first NaN plan.

## Simulation (Section 6.3)

```
mkdir -p out/swfi2
python swfi2.py out/swfi2/swfi2.npz --faults 250          # 11 sources x 8 observations x 250 = 22,000 faults, about 1 h
python swfi2_analyze.py out/swfi2/swfi2.npz              # Table 6
```

`--plan` picks the fault sources, and `--obs` and `--faults` give a shorter run. The default plan also has two KV bit-flip models that are not in the paper. Leave them in: every source draws from the same random stream, so `--seed 0` reproduces the paper's faults only with the full plan.

The TRT column of Table 6 comes from the same simulation on the Jetson's engines (`jetson/trt_blocksim.py`, see the [Jetson README](../jetson/README.md#files)). Copy its output to the host and compare:

```
mkdir -p out/blocksim && scp user@jetson:fi_logs/blocksim.npz out/blocksim/
python trt_blocksim_compare.py out/blocksim/blocksim.npz out/swfi2/swfi2.npz
```

## Analysis

| Paper | Command |
|---|---|
| Section 4.3 | `python gpumem_analyze.py out/gpumem/RUN.jsonl` |
| Table 6 | `python swfi2_analyze.py out/swfi2/swfi2.npz` (PT), `python trt_blocksim_compare.py out/blocksim/blocksim.npz out/swfi2/swfi2.npz` (TRT) |
| Table 7 | `python pos_analyze.py out/v3fi/RUN [RUN ...]`, one call per target; the table pools the runs of a target |
| Section 6.4, what the arm executed | `python move_analyze.py out/v3fi/RUN [RUN ...]` (fetches the episodes' session logs from the Jetson) |
| Table 8 | `python swfi2_em.py out/swfi2/swfi2.npz $(ls out/v3fi/*.jsonl \| sed 's/.jsonl$//') --moving --vmin 0 --vmax 1000` |
| Section 6.6, the load results | `python arm_load_summary.py out/arm/*.jsonl` |
| Section 6.6, the arm episodes | `python arm_motion.py out/arm/RUN_motion.npz --golden arm_golden.npz` (`arm_golden.npz` is in `JETSON_DIR`) |
