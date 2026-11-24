#!/bin/bash
# Random-sample judging: generate the 160-token answers while both judges label the finished answer files.
cd "$(dirname "$0")/.."; export HF_HUB_OFFLINE=1
S=results/judge/sample; mkdir -p $S; rm -f $S/gen.done $S/answers/*.tmp
( .venv/bin/python code/judge_sample.py gen > $S/gen.log 2>&1 && touch $S/gen.done || echo GEN FAILED ) &
GEN=$!
jloop() {
  while :; do
    fin=0; [ -e $S/gen.done ] && fin=1
    .venv/bin/python code/judge_sample.py judge "$1" >> "$2" 2>&1
    [ $fin = 1 ] && break
    kill -0 $GEN 2>/dev/null || [ -e $S/gen.done ] || break
    /bin/sleep 60
  done
}
jloop deepseek-v4.1-flash:cloud $S/deepseek.log &
jloop glm-5.3:cloud $S/glm.log &
wait; echo "SAMPLE JUDGING DONE $(date)"
