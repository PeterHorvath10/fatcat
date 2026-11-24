#!/bin/bash
# Pin the Thor GPU compute clock (devfreq gpu-gpc-0) to one frequency: lock_gpu_clock.sh HZ
# (HZ = gpu_clock.gpc_hz in config/thor.toml). A fixed clock keeps the glitch timing stable.
set -u
TARGET=${1:?usage: lock_gpu_clock.sh HZ}
D=/sys/class/devfreq/gpu-gpc-0
for i in $(seq 1 60); do [ -e "$D/min_freq" ] && break; sleep 1; done
[ -e "$D/min_freq" ] || { echo "gpu-gpc-0 devfreq not present" >&2; exit 1; }
echo userspace > "$D/governor" 2>/dev/null || true
# raise max before min when going up
echo "$TARGET" > "$D/max_freq"
echo "$TARGET" > "$D/min_freq"
echo "$TARGET" > "$D/userspace/set_freq" 2>/dev/null || true
echo "gpu-gpc-0 locked: gov=$(cat $D/governor) cur=$(cat $D/cur_freq) min=$(cat $D/min_freq) max=$(cat $D/max_freq)"
