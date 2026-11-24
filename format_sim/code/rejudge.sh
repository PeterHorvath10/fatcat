#!/bin/bash
# Label what is still missing or unclear, with both judges, then rebuild the table.
cd "$(dirname "$0")/.."
.venv/bin/python code/judge160.py judge deepseek-v4.1-flash:cloud > results/judge/deepseek.log 2>&1 &
.venv/bin/python code/judge160.py judge glm-5.3:cloud > results/judge/glm.log 2>&1 &
wait
.venv/bin/python code/judge160.py table > /dev/null
echo "unclear left: $(grep -o '([0-9]*?)' results/JUDGED.md | wc -l) cells"
