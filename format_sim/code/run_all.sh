#!/bin/bash
# Full sweep: 2 models x 9 formats, 4 jobs at a time on one GPU. Resumable (finished configurations are skipped).
cd "$(dirname "$0")/.."; export HF_HUB_OFFLINE=1
DRAWS=${DRAWS:-20}
for f in bf16 fp8 int8 int4 nvfp4 mxfp4 gguf-Q4_K_M gguf-Q4_0 gguf-Q8_0; do for m in llama gemma; do echo "$m $f"; done; done |
  xargs -P 4 -L 1 bash -c '.venv/bin/python code/sim.py $0 $1 results/${0}_$1.jsonl --draws '$DRAWS' > results/${0}_$1.log 2>&1'
echo ALL DONE
