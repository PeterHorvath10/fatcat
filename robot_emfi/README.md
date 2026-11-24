# Faulting a robot policy with EMFI

This is the artifact for Section 6, "Faulting a real robot", of the paper *FATCAT: Fault Injection Attacks Targeting Chip-level Alignment* by Peter Horvath, Ilia Shumailov, Zhuoran Liu, Łukasz Chmielewski and Lejla Batina (section, table and figure numbers as in the arXiv version). A Jetson Orin Nano runs a fine-tuned SmolVLA vision-language-action policy (TensorRT, fp16) that drives an SO-101 robot arm. The code does three things:

- **Simulates** one garbled 16-byte DRAM block per inference, both in a PyTorch copy of the policy and in the deployed TensorRT engines (Section 6.3).
- **Injects** electromagnetic pulses while the policy infers (Section 6.4) and while it loads its weights (Section 6.6), and lets the arm execute the faulted plans. A ChipSHOUTER makes the pulses and a ChipWhisperer-Husky times them.
- **Analyzes** the campaigns into Tables 6–8 and the load results of Section 6.6.

## Contents

| Folder | Runs on | Contents |
|---|---|---|
| [`host/`](host/README.md) | Host PC | Campaign control (Husky, ChipSHOUTER, ssh to the Jetson), the PyTorch simulation, all analyses |
| [`jetson/`](jetson/README.md) | Jetson Orin Nano | The policy rollout, the workers the host starts over ssh, the TensorRT simulation |
| [`data/`](data/README.md) | Both | The 8 reference observations and the canary frames |

Each folder has its own README with setup steps and a table of its files.

You never copy `jetson/` to the Jetson by hand. Every host campaign script first copies `jetson/` and `data/` to the Jetson (`site_config.deploy_jetson()`), then starts the Jetson side over ssh.

## Paper results

| Paper | Result | Run | Analyze (in `host/`) | Needs |
|---|---|---|---|---|
| 6.3, Table 6 (PT) | Block-fault simulation in PyTorch | `host/swfi2.py --faults 250` | `swfi2_analyze.py` | CUDA GPU, the checkpoint; about 1 h |
| 6.3, Table 6 (TRT) | The same faults in the deployed TensorRT engines | `jetson/trt_blocksim.py --faults 500 --seed 1` | `trt_blocksim_compare.py` | Jetson; about 2 h |
| 6.4, Table 7 | EM pulses during inference, with the arm moving | `host/fi_v3_run.py --move` | `pos_analyze.py`, `move_analyze.py` | Husky, ChipSHOUTER, Jetson, arm |
| 6.5, Table 8 | EM pulses against the simulation, per stage | (nothing to run) | `swfi2_em.py` | The outputs of 6.3 and 6.4 |
| 6.6 | EM pulses during the live rollout's weight load | `host/arm_load_run.py` | `arm_load_summary.py`, `arm_motion.py` | Husky, ChipSHOUTER, Jetson, arm |

Fig. 10 shows frames from a camera recording of the 6.4 episodes; no script here makes it.

The exact commands are in the host README, under [campaigns](host/README.md#running-the-campaigns), [simulation](host/README.md#simulation-section-63) and [analysis](host/README.md#analysis). The TensorRT simulation's command is in the [Jetson README](jetson/README.md#files).

## Hardware

| Part | Role |
|---|---|
| NVIDIA Jetson Orin Nano Developer Kit (Super) | The target: runs the policy |
| SO-101 follower arm (LeRobot), two USB cameras | The robot, with a top and a side view |
| NewAE ChipWhisperer-Husky | Captures the sense probe, triggers, and fires the injector through its glitch output |
| NewAE ChipSHOUTER with an injection tip | The EM pulses. `fi_v3_run.py` can also drive a FaultyCat instead |
| Near-field langer probe | Over a capacitor near the Jetson module, into the Husky's MEASURE input; it times the pulses |
| Host PC (Linux) | Drives everything; it needs a CUDA GPU for the PyTorch simulation |
| Arduino (optional) | Pulls the Jetson's reset line when a pulse hangs it |

```
                        USB                    MCX glitch out -> SMB trigger in (50 ohm)
   +---------+ ------------------ Husky ------------------------------------ ChipSHOUTER
   | Host PC |                     ^ MEASURE                                    | injection tip
   |         |                     |                                            v
   |         |                near-field sense probe ------> over the Jetson module (GPU / LPDDR5)
   |         | ---- Ethernet, ssh ---------------------------> Jetson Orin Nano --USB--> SO-101 arm,
   |         | ---- USB (optional) -- Arduino --> reset line /                          2 cameras
   +---------+
```

- **Trigger:** in the inference campaigns, the Husky triggers on the Langer probe's signal. In the memory test and the load campaigns, the host drives the Husky's TIO4 line instead.
- **Pulse width:** the ChipSHOUTER fires on the Husky's glitch output, so the glitch length is the pulse width: 80 ns (4 cycles at 50 MHz) in the paper.
- **Clocks:** the Jetson must run with fixed clocks (`jetson/jetson-max-perf.service`). With dynamic clocks, the pulse timing drifts.

## Software

- **Host:** Python 3.12 in two environments:
  - one for the campaigns, with `chipwhisperer` and `chipshouter`;
  - one for the simulation, with PyTorch and LeRobot.

  The analyses need only `numpy`. See [host setup](host/README.md#setup) for the versions.
- **Jetson:** JetPack 7.2.1 (CUDA 13.2, TensorRT 10.16) and LeRobot 0.6.2. The policy runs on [jetson-orin-nano-vla](https://github.com/eetmie/jetson-orin-nano-vla), an open-source (MIT) TensorRT runtime for SmolVLA, with a small patch from `jetson/patches/`. See [Jetson setup](jetson/README.md#setup).

## Getting started

1. **Set up both machines.** Edit the two `site_config.py` files, or put your values in a `site_local.py` next to each ([host](host/README.md#setup), [Jetson](jetson/README.md#setup)).
2. **Capture the canary's golden chunk** once on the Jetson (`capture_golden.py`).
3. **Place the probes.**
   - Find the spot for the sense probe with `host/probe_monitor.py` while `jetson/probe_load.py` runs.
   - Place the injection tip, then run the memory read test (`gpumem_run.py --mode read`) until the pulses garble blocks reliably.
4. **Run the simulation** on the host GPU, and its TensorRT version on the Jetson. Neither needs the EM setup.
5. **Run the campaigns.** Run each one first with `--char`, which fires nothing and checks the trigger timing.
6. **Analyze** on the host.

