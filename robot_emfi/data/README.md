# Data

Small inputs that both machines need. The host uses them for the simulation, and copies them to the Jetson with the scripts.

**`traces/RUN/trace.npz`:** the 8 reference observations.
- **What:** camera images (`req_top`, `req_side`, 480x640 RGB), joint state (`req_state`, degrees) and control step (`req_step`) of 8 inference requests from five fault-free rollouts of 27 Oct 2026.
- **Who uses them:**
  - the simulation (`host/swfi2.py`);
  - the classification of faulted plans on the Jetson (`trt_blocksim.py`, `arm_load_worker.py`, through `wmem.py`);
  - the analyses of both.

  The observations are listed in `swfi2.OBS` and `wmem.OBS`.
- **How they were made:** `host/extract_observations.py` cut them from the full rollout logs.

**`canary/first_obs_{top,side}.png`:** the first camera frames of a fault-free rollout, with the arm at rest.
- `fi_worker.py` runs them, with `CANARY_STATE` from `jetson/site_config.py`, as a fixed canary observation after every inference of a live campaign. This catches corruption that stays in GPU memory.
- The canary's fault-free chunk depends on the engines, so it is captured on each Jetson (`jetson/capture_golden.py`).
