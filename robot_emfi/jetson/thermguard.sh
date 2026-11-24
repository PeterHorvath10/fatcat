#!/bin/bash
# usage: thermguard.sh PID [HOT_C] [COOL_C] -- pause PID (SIGSTOP) while the Jetson's tj-thermal reads HOT_C or
# more and resume it (SIGCONT) at COOL_C or less, until PID exits. The board's fan does not spin (tach 0 rpm at
# PWM 255): a long swfi.py run reached 99 C, the GPU's passive trip, on 2026-11-05. Starts as if PID were paused.
pid=$1; hot=${2:-90}; cool=${3:-75}
zone=$(dirname "$(grep -l '^tj-thermal$' /sys/devices/virtual/thermal/thermal_zone*/type | head -1)")
trap 'kill -CONT $pid 2>/dev/null' EXIT
paused=1
while kill -0 "$pid" 2>/dev/null; do
  t=$(( $(cat "$zone/temp") / 1000 ))
  if [ $paused = 0 ] && [ $t -ge $hot ]; then
    kill -STOP "$pid"; paused=1; echo "$(date +%T) tj $t C: paused"
  elif [ $paused = 1 ] && [ $t -le $cool ]; then
    kill -CONT "$pid"; paused=0; echo "$(date +%T) tj $t C: resumed"
  fi
  sleep 1
done
