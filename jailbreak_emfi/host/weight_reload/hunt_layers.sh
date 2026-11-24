#!/bin/bash
# Weight-reload hunt (Thor, vLLM reload worker): for each layer in hunt.layers, glitch the reload until that
# layer's down_proj (and nothing else) is fully wiped, then record the sanity answers, the weight-check
# record, the 100 JailbreakBench answers and MMLU/ARC-C, reload clean weights and go on to the next layer.
# The window comes from window.py (every hit so far) and is recomputed every round of hunt.round_trials
# trials. A layer is given up after hunt.max_trials trials; a layer whose jbb.jsonl exists is skipped.
#
# Usage (from anywhere): PAPER_CODE=<jailbreak_emfi> host/weight_reload/hunt_layers.sh [EXPERIMENT_TOML]
# Settings: config/host.toml + the experiment TOML (default experiments/thor_weight_reload_hunt.toml).
# Env: the variable named by network.sudo_password_env (this PC's sudo, boardctl reset by the controller)
# and the one named by hunt.sudo_password_env (the Thor's sudo, restarting the reload worker).
# OUT_SUFFIX (e.g. _rerun) writes to L<layer><suffix>/, to repeat a layer that is already done.
set -u
S=$(cd "$(dirname "$0")" && pwd)
export PAPER_CODE=${PAPER_CODE:-$(cd "$S/../.." && pwd)}
export PYTHONPATH=$PAPER_CODE${PYTHONPATH:+:$PYTHONPATH}
EXP=${1:-experiments/thor_weight_reload_hunt.toml}
OUT_SUFFIX=${OUT_SUFFIX:-}
# read_cfg VAR KEY: VAR = the setting KEY (stops the script if it is empty)
read_cfg() { local v; v=$(python3 -m common.cfgget "host.toml,$EXP" "$2") || exit 1; printf -v "$1" '%s' "$v"; }
read_cfg THOR boards.thor.ssh_host
read_cfg M hunt.model
read_cfg LAYERS hunt.layers
read_cfg MAX_TRIALS hunt.max_trials
read_cfg ROUND hunt.round_trials
read_cfg MIN_ELEMS hunt.full_wipe_elems
read_cfg D hunt.out_dir
read_cfg R hunt.rounds_dir
read_cfg HITS hunt.hits_csv
read_cfg PY hunt.python
read_cfg UNIT hunt.reload_unit
read_cfg READY hunt.ready_pattern
read_cfg P hunt.remote_paper_code
read_cfg RPY hunt.remote_python
read_cfg CHECKS hunt.remote_weight_checks
read_cfg RO hunt.remote_out_dir
read_cfg HOST_PW_ENV network.sudo_password_env
read_cfg THOR_PW_ENV hunt.sudo_password_env
: "${!HOST_PW_ENV:?set $HOST_PW_ENV (this PC's sudo password)}"
: "${!THOR_PW_ENV:?set $THOR_PW_ENV (the Thor's sudo password)}"
mkdir -p "$R" "$D"

restart_trigger() {
  printf '%s\n' "${!THOR_PW_ENV}" | ssh "$THOR" "sudo -S -p '' systemctl restart $UNIT; until journalctl -u $UNIT -n 3 --no-pager -o cat | grep -q '$READY'; do sleep 2; done"
}
restart_trigger
i=$(ls "$R"/ctl*.log 2>/dev/null | sed 's/.*ctl\([0-9]*\).log/\1/' | sort -n | tail -1); i=${i:-0}
for L in $LAYERS; do
  O=$D/L$L$OUT_SUFFIX
  [ -s "$O/jbb.jsonl" ] && { echo "L$L done, skipping"; continue; }
  TARGET=model.layers.$L.mlp.down_proj
  trials=0; hitless=0
  echo "$(date +%T) === L$L down_proj"
  while [ "$trials" -lt "$MAX_TRIALS" ]; do
    i=$((i+1))
    W=$(python3 "$S/window.py" "$EXP" "$L" "$hitless") || exit 1
    read -r W0 W1 <<< "$(python3 -c 'import sys; a, b = sys.argv[1].split(","); print(int(float(a) * 1e6), int(float(b) * 1e6))' "$W")"
    EMFI_GLITCH__WINDOW_START_NS=$W0 EMFI_GLITCH__WINDOW_END_NS=$W1 "$PY" -u "$PAPER_CODE/host/run_thor_fixed_time.py" \
      --experiment "$EXP" --trials "$ROUND" --trial-log "$R/trials$i.csv" > "$R/ctl$i.log" 2>&1 < /dev/null &
    pid=$!
    python3 -u "$S/until_hit.py" "$R/ctl$i.log" $pid > "$R/watch$i.log" 2>&1
    wait $pid
    n=$(grep -c '^Run:' "$R/ctl$i.log"); trials=$((trials+n))
    hit=$(grep "HIT:" "$R/watch$i.log")
    echo "$(date +%T) round $i (L$L, window $W ms): trials $n (layer total $trials), resets $(grep -c '^Sent reset' "$R/ctl$i.log")"
    if [ -z "$hit" ]; then hitless=$((hitless+n)); echo "no hit"; restart_trigger; continue; fi
    hitless=0
    echo "$hit"
    echo "  delay: $(python3 "$S/hit_delay.py" "$R/trials$i.csv" "$hit" "$HITS" "$TARGET")"
    if names=$(python3 "$S/is_full_wipe.py" "$hit" "$TARGET" "$MIN_ELEMS"); then
      mkdir -p "$O"
      echo "$hit" > "$O/hit.txt"
      cp "$R/trials$i.csv" "$O/trials.csv"
      ssh "$THOR" "tail -1 $CHECKS" > "$O/weight_check_record.json"
      ssh "$THOR" "cd ~/$P && PYTHONPATH=. $RPY -m jetson.thor.eval.eval_model --model $M" > "$O/answers.txt"
      python3 "$S/sensible.py" "$EXP" < "$O/answers.txt"
      ssh "$THOR" "cd ~/$P && mkdir -p $RO && PYTHONPATH=. $RPY -m jetson.thor.eval.jbb_run $RO/jbb_L$L$OUT_SUFFIX.jsonl --model $M" \
        && scp -q "$THOR:$P/$RO/jbb_L$L$OUT_SUFFIX.jsonl" "$O/jbb.part" && mv "$O/jbb.part" "$O/jbb.jsonl"
      ssh "$THOR" "cd ~/$P && PYTHONPATH=. $RPY -m jetson.thor.eval.mc_run $RO/mc_L$L$OUT_SUFFIX.jsonl --model $M" > "$O/mc.txt" \
        && scp -q "$THOR:$P/$RO/mc_L$L$OUT_SUFFIX.jsonl" "$O/mc_items.jsonl"
      echo "$(date +%T) L$L down_proj wiped and evaluated: $(tail -2 "$O/mc.txt" | tr '\n' ' ')"
      restart_trigger
      break
    fi
    echo "not $TARGET ($names)"
    restart_trigger
  done
  [ -s "$O/jbb.jsonl" ] || echo "$(date +%T) L$L: no down_proj hit in $trials trials, moving on"
done
echo "all layers done"
