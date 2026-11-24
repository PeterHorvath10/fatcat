#!/bin/bash
# Regenerate the selected outcomes of all finished sweeps at 160 tokens, judge with both models, build the table. Resumable.
cd "$(dirname "$0")/.."; export HF_HUB_OFFLINE=1
.venv/bin/python code/judge160.py gen > results/judge/gen.log 2>&1 || { tail -3 results/judge/gen.log; exit 1; }
.venv/bin/python code/judge160.py judge deepseek-v4.1-flash:cloud > results/judge/deepseek.log 2>&1 &
.venv/bin/python code/judge160.py judge glm-5.3:cloud > results/judge/glm.log 2>&1 &
wait
.venv/bin/python code/judge160.py table > /dev/null
echo JUDGE DONE
