# Jetson Orin Nano

Everything in this folder runs on the Jetson. The host copies it, together with `../data`, to `JETSON_DIR` before every campaign (`host/site_config.py`). Most of it is started by the host over ssh; the rest you run by hand on the Jetson, as the table says.

There are two Python environments, as in the deployed robot:
- **AOT venv:** the TensorRT runtime's virtual environment (`site_config.AOT_PYTHON`). It runs the policy and everything that touches the GPU.
- **LeRobot env:** a conda environment with LeRobot. It runs the rollout's main process, which talks to the arm and the cameras and starts the policy worker in the AOT venv.

## Setup

1. **System:** JetPack 7.2.1 (L4T R39.2.1, CUDA 13.2, TensorRT 10.16) on a Jetson Orin Nano developer kit (Super).
2. **Fixed clocks:** install `jetson-max-perf.service` (copy it to `/etc/systemd/system/`, then `systemctl enable --now jetson-max-perf`). It sets MAXN_SUPER and runs `jetson_clocks`. Check before every campaign that the GPU's minimum frequency equals its maximum (1020 MHz). The trigger timing assumes fixed clocks.
3. **TensorRT runtime:** the policy runs on [jetson-orin-nano-vla](https://github.com/eetmie/jetson-orin-nano-vla), an open-source (MIT) TensorRT runtime for SmolVLA on the Jetson Orin Nano. We use commit `6e2d857`.
   ```
   git clone https://github.com/eetmie/jetson-orin-nano-vla ~/vla-runtime && cd ~/vla-runtime
   git checkout 6e2d857 && git apply /path/to/jetson/patches/vla-runtime.patch
   ```
   The patch passes only the active camera slots to the fixture check, and takes the action dimension from the bundle. Then follow the runtime's README:
   - create its `.venv`;
   - export the fine-tuned SmolVLA checkpoint to an ONNX bundle;
   - build the TensorRT engines (`site_config.BUNDLE`, `site_config.AOT_CACHE`).
4. **LeRobot:** create a conda environment with LeRobot 0.6.2. Calibrate the SO-101 follower with LeRobot so the rollout finds its calibration.
5. **Settings:** edit `site_config.py`, or write a `site_local.py` next to it with your values; the host's deploy copies it along:
   - `VLA_REPO`, `AOT_PYTHON`: the runtime and its Python;
   - `BUNDLE`, `AOT_CACHE`: the model;
   - `FOLLOWER_PORT`: the arm's serial port, by USB serial number;
   - `CAMERA_TOP`, `CAMERA_SIDE`: the camera paths, by USB port.

   Check the camera views: the rollout picks the cameras by USB port, and swapped plugs feed the policy swapped views.
6. **Golden chunk:** deploy once from the host (any campaign script does it), then capture the canary's fault-free chunk. It depends on the engines built on this Jetson:
   ```
   cd $JETSON_DIR && $AOT_PYTHON capture_golden.py
   ```

## Files

| File | Runs in | Started by | What |
|---|---|---|---|
| `smolvla_212_v3_rollout_async.py` | LeRobot env, plus its worker in the AOT venv | `fi_v3.py`, `arm_load_fi.py` | The deployed rollout: 30 Hz control, a new 50-action chunk every 19 ticks, blended 30/70 with the previous one. On its own, `python smolvla_212_v3_rollout_async.py --duration 30` runs a fault-free episode |
| `fi_v3.py` | LeRobot env | `host/fi_v3_run.py` | Runs the rollout for an inference campaign. It logs every command, and drops the commands unless `FI_V3_MOVE=1`. It swaps the policy worker for `fi_worker.py` |
| `fi_worker.py` | AOT venv | `fi_v3.py` | Wraps the policy worker. After each reply, it recomputes the same observation (the fault-free reference) and runs the canary. `FI_V3_NAN=drop` keeps the previous plan on a NaN plan |
| `arm_load_fi.py` | LeRobot env | `host/arm_load_run.py` | Runs the rollout for a load campaign. The arm moves only once the load is labelled erratic |
| `arm_load_worker.py` | AOT venv | `arm_load_fi.py`, `host/arm_load_run.py` | Marks the start of the expert's weight copy, so the host can time the pulse. Reads the weights back, classifies the plans of the 8 observations, and bypasses the runtime's known-answer test, so faulted loads are kept |
| `gpumem.cu` | (built with `nvcc`) | `host/gpumem_run.py` | Memory test: streams a known pattern from DRAM, or writes it, and checks every word |
| `swfi.py`, `wmem.py`, `onnx_inits.py` | AOT venv | the workers above | Loading the policy chain, a recording allocator that finds the weights in GPU memory, and the weight map |
| `capture_golden.py` | AOT venv | by hand, once | The canary's fault-free chunk, which `fi_worker.py` compares against |
| `trt_blocksim.py` | AOT venv | by hand | The block simulation of Section 6.3 on the deployed TensorRT engines: one garbled 16-byte block per inference in the weights or the KV cache, in TensorRT's own weight layout (the activation sources are inside the engines and not reachable). Checks Table 6 |
| `probe_load.py` | AOT venv | by hand | Reloads the expert's engine in a loop, for placing the sense probe with `host/probe_monitor.py` |
| `thermguard.sh` | shell | by hand | `thermguard.sh PID` pauses a long GPU job while the chip is hot. The dev kit's fan may not spin |
| `site_config.py` | | | This Jetson's paths and devices |

The workers write their logs to `~/fi_logs`. The host fetches what it needs.

**Running the hand-started tools:** run them in the AOT venv from `JETSON_DIR`:

```
cd $JETSON_DIR
$AOT_PYTHON trt_blocksim.py ~/fi_logs/blocksim.npz --faults 500 --seed 1 & ./thermguard.sh $!   # Table 6 (TRT): 4,000 faults per source, about 2 h
```
