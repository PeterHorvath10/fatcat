#!/bin/bash
# Llama BF16, all 20 random blocks of four targets: generate the 160-token answers, then label them with both judges.
cd "$(dirname "$0")/.."; export HF_HUB_OFFLINE=1
mkdir -p results/judge
.venv/bin/python code/judge_expect.py > results/judge/gen4.log 2>&1 || { echo GEN4 FAILED; tail -n 15 results/judge/gen4.log; exit 1; }
tail -n 1 results/judge/gen4.log
.venv/bin/python code/judge160.py judge deepseek-v4.1-flash:cloud > results/judge/deepseek4.log 2>&1 &
.venv/bin/python code/judge160.py judge glm-5.3:cloud > results/judge/glm4.log 2>&1 &
wait; echo "JUDGE4 DONE $(date)"
