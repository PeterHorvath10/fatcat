#!/bin/bash
# Second judging pass: regenerate the selected all-tensor outcomes at 160 tokens, label with both judges, rebuild the table.
cd "$(dirname "$0")/.."
export HF_HUB_OFFLINE=1
.venv/bin/python code/judge160.py gen2 > results/judge/gen2.log 2>&1 || { echo "GEN2 FAILED"; tail -n 20 results/judge/gen2.log; exit 1; }
echo "gen2 done $(date): $(cat results/judge/answers/*.p2.jsonl | wc -l) outcomes"
.venv/bin/python code/judge160.py judge deepseek-v4.1-flash:cloud > results/judge/deepseek2.log 2>&1 &
.venv/bin/python code/judge160.py judge glm-5.3:cloud > results/judge/glm2.log 2>&1 &
wait
.venv/bin/python code/judge160.py table > /dev/null
echo "unclear left: $(grep -c '?)' results/JUDGED.md) cells"; echo "JUDGE2 DONE $(date)"
