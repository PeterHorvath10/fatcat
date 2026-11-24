# Analysis

Run everything from `jailbreak_emfi/` as `PYTHONPATH=$PWD python -m analysis.<package>.<module>`. All settings come
from `config/analysis.toml` (shipped empty).
`python -m analysis.selftest` checks the code without hardware, judges or run data.

## Layout

| Package | Modules | Does |
|---|---|---|
| `analysis/judge/` | `build_input`, `judge`, `labels`, `count`, `agreement` | JailbreakBench answers → judge input → LLM labels → unsafe/bypass counts, judge agreement |
| `analysis/scope/` | `trace`, `pulse_layer_report`, `layer_timing_table`, `trigger_test` | near-field traces: layer bursts, where the pulse lands, per-prompt layer timing, live trigger test |
| `analysis/` | `keyword` | refusal-keyword and coherence check |
| `analysis/dram/` | `dram_map` | address → DRAM channel for Thor and Orin |
| `analysis/` | `paper_numbers`, `selftest` | all paper numbers in one go; self-test |

Shared code: the trace decoding (`WAVEDESC` preamble) is `host/emfi/scope.py`, the same code that saved the
traces; HTTP calls (judge, trigger test) use `common/http.py`.

## Typical use

1. `analysis.judge.build_input RUN_DIR` builds `judge/all.jsonl` from the run's `results.jsonl`
   (index = prompt_slot·100 + prompt_repeat; stops on a duplicate slot/repeat).
2. `analysis.judge.judge` labels it with each judge model (**this calls paid cloud models**). It repeats the
   file until no `UNCLEAR` label is left (HTTP 429s), up to `judge.max_rounds`.
3. `analysis.judge.count RUN_DIR [--baseline LABELS] [--first-k K ...]`: unsafe by every judge / by at least
   one, answers and prompts, per prompt. **Counts are raw**: `--baseline` only adds a column and a
   "without baseline-unsafe prompts" line, it never drops prompts silently.
4. `analysis.scope.pulse_layer_report RUN_DIR --cache DIR` and `analysis.scope.layer_timing_table RUN_DIR --cache DIR`
   (the first call summarises the raw 10 MB traces into the cache, later calls read only the cache).
5. `analysis.paper_numbers [--scope]`: all of the above for the runs in `[paper]`.

`analysis.scope.trigger_test` is the only analysis script that talks to instruments (the scope and the vLLM
server); run it on purpose only. It saves and restores the scope's trigger settings.

## Which script makes which paper number

- **tab:emfi-judged-outcomes**, judge columns and judge agreement: `judge.agreement`.
- **tab:results_faultinj_jbb**: `judge.count --first-k K [--first-k K2 ...]`.
- **tab:channel-masks**: `dram.dram_map`.
- Thor JailbreakBench runs with scope traces (unsafe counts, pulse layer, per-prompt timing): `judge.count`,
  `scope.pulse_layer_report`, `scope.layer_timing_table`.

## DRAM map caveats

- The Thor channel masks apply to PA − 0x80000000, not the raw PA.
- A `cudaMalloc` pointer has no CPU PTE, so the memory-check workloads use pinned host memory plus
  `/proc/self/pagemap` to get physical addresses.
- Logs of physical addresses can be labelled with channels with `dram_map thor annotate`.
