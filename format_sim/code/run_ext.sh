#!/bin/bash
# After the main sweep: (1) judged outcomes of the main sweep, (2) extended sweep over the other six tensors. Resumable.
cd "$(dirname "$0")/.."; export HF_HUB_OFFLINE=1
while pgrep -f "code/run_all.sh" > /dev/null; do sleep 30; done
echo "main sweep finished $(date)"
.venv/bin/python code/judge160.py gen > results/judge/gen.log 2>&1 || { tail -3 results/judge/gen.log; echo GEN FAILED; }
( .venv/bin/python code/judge160.py judge deepseek-v4.1-flash:cloud > results/judge/deepseek.log 2>&1 &
  .venv/bin/python code/judge160.py judge glm-5.3:cloud > results/judge/glm.log 2>&1 &
  wait; .venv/bin/python code/judge160.py table > /dev/null; echo "JUDGE DONE $(date)" ) &
DRAWS=${DRAWS:-20}
for f in bf16 fp8 int8 int4 nvfp4 mxfp4 gguf-Q4_K_M gguf-Q4_0 gguf-Q8_0; do for m in llama gemma; do echo "$m $f"; done; done |
  xargs -P 4 -L 1 bash -c '.venv/bin/python code/sim.py $0 $1 results/${0}_$1.jsonl --draws '$DRAWS' --plan ext > results/${0}_$1.ext.log 2>&1'
wait
echo "EXT DONE $(date)"
