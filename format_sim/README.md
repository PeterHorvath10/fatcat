# Simulated prefill faults, by parameter format

This is the code for the appendix "Simulated transient faults in the prefill" of the paper. It is a software simulation: no instruments, one CUDA GPU.

It answers one question: if a single 16-byte block of a weight tensor reads back as random bytes while the prompt is processed, what happens to the model's refusals, and how does that depend on the format in which the weights are stored?

- **Models:** Llama 3.2 1B and Gemma 3 1B, run in Hugging Face Transformers with greedy decoding on the 100 harmful JailbreakBench prompts.
- **Fault:** one aligned 16-byte block of uniformly random bytes in the stored bytes of one weight tensor. This is what a faulted read of the encrypted DRAM returns.
- **When:** only during the prefill. The answer is decoded with clean weights from the key/value cache that the faulted prefill wrote. Three cases:
  - `whole`: the whole prompt is computed with the fault;
  - `tail`: the prompt's full 16-token blocks come from a clean prefix cache and only the last 1–16 tokens are computed with the fault;
  - `persist`: the fault also stays during decoding.
- **Formats:** each tensor is encoded into the bytes that the format keeps in memory, the fault overwrites 16 of them, and the bytes are decoded back.

| Format | Stored bytes |
|---|---|
| `bf16` | 2 bytes per weight |
| `fp8` | E4M3 code per weight, one FP32 scale per tensor |
| `int8` | 8-bit code per weight, one BF16 scale per output row |
| `int4` | 4-bit code per weight, one BF16 scale per group of 128 weights |
| `nvfp4` | E2M1 code per weight, one E4M3 scale per 16 weights, one FP32 scale per tensor |
| `mxfp4` | E2M1 code per weight, one power-of-two scale per 32 weights |
| `gguf-Q4_K_M`, `gguf-Q4_0`, `gguf-Q8_0` | the raw tensor bytes of the GGUF file, with their scales inline |

## Setup

```
cd format_sim
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

- **Checkpoints:** `meta-llama/Llama-3.2-1B-Instruct` and `google/gemma-3-1b-it` from Hugging Face (both are gated; accept their licenses first). The scripts set `HF_HUB_OFFLINE=1`, so download them once before the first run.
- **GGUF files:** only for the three `gguf-*` formats. Put them in
  `gguf/Llama-3.2-1B-Instruct/Llama-3.2-1B-Instruct-{Q4_K_M,Q4_0,Q8_0}.gguf` and
  `gguf/gemma-3-1b-it/gemma-3-1b-it-{Q4_K_M,Q4_0,Q8_0}.gguf`.
  We used the files of `unsloth/Llama-3.2-1B-Instruct-GGUF` and `unsloth/gemma-3-1b-it-GGUF`.
- **Prompts and judge:** read from the neighbouring folder, `../jailbreak_emfi/datasets/jailbreakbench/harmful-behaviors.csv` and `../jailbreak_emfi/analysis/judge/judge.py`. Keep the two folders side by side.
- **Judges:** the judging steps call an Ollama endpoint. Set it in `[judge]` of `../jailbreak_emfi/config/analysis.toml` (`ollama_url`, `request_timeout_s`, `attempts`, `temperature`; shipped empty). We used a local Ollama (`http://localhost:11434/api/chat`) with the cloud models `deepseek-v4.1-flash:cloud` and `glm-5.3:cloud`.

Everything is written to `results/`. All steps are resumable: a finished configuration or an already labelled answer is skipped.

## Quick check

```
.venv/bin/python code/sim.py llama bf16 /tmp/smoke.jsonl --plan smoke --draws 1
```

This runs the clean model and one fault in `down_proj` of layer 8 in each of the three cases, in a few seconds.

## Steps

| Step | Command | Output |
|---|---|---|
| 1. Sweep: `down_proj` of every layer and the other tensors of a few key layers, 20 random blocks each; faults forced into a scale; persistent faults | `code/run_all.sh` | `results/{model}_{format}.jsonl` |
| 2. Extended sweep: the other six weight tensors | `code/run_ext.sh` (waits for step 1, also starts step 4) | appended to the same files |
| 3. Keyword tables | `.venv/bin/python code/summarize.py` | `results/SUMMARY.md`, `summary.csv` |
| 4. Judged outcomes of the most effective faults, first pass | `code/judge_all.sh` | `results/JUDGED.md` |
| 5. Second pass, all tensors | `code/judge_pass2.sh` | `results/JUDGED.md` |
| 6. Coverage: prompts with an unsafe answer under the best 2, 3, 5, 10 faults | `.venv/bin/python code/coverage.py` | `results/COVERAGE.md` |
| 7. Llama BF16, layer 8, every tensor | `.venv/bin/python code/judge_l8.py`, then `code/rejudge.sh` | labels of `llama_bf16.p3` |
| 8. Llama BF16, all 20 blocks of four targets | `code/judge_expected.sh`, then `.venv/bin/python code/expect_table.py` | printed table |
| 9. Random sample of faults, every distinct answer judged once | `code/judge_sample.sh`, then `.venv/bin/python code/judge_sample.py table` | `results/SAMPLED.md` |

The two sweeps take several hours each on one RTX 5090 (four formats run in parallel); the judging steps are limited by the judges, at roughly 200 answers per minute each.

Steps 1–3 use 48-token answers and a refusal-keyword list. Steps 4–9 regenerate the selected faults with 160-token answers and label each answer *refused*, *unsafe* or *other* with the two judges.

## Files

| File | Does |
|---|---|
| `code/formats.py` | The formats at byte level: encodes a tensor into its stored bytes, applies a 16-byte fault to one region (codes or scales), returns the changed weights |
| `code/sim.py` | The simulation: loads a model in a format, plans the faults, runs the three prefill cases over the 100 prompts, scores the answers with the keyword list. `sim.py MODEL FORMAT OUT.jsonl [--draws N] [--max-new N] [--plan full\|ext\|smoke]` |
| `code/summarize.py` | Tables of the sweep per model, format, layer, tensor and scale |
| `code/judge160.py` | Selects the faults with the fewest keyword refusals, regenerates them with 160 tokens (`gen`, `gen2`), labels them (`judge MODEL`), builds the table (`table`) |
| `code/coverage.py` | Best single fault and greedy coverage from the labels |
| `code/judge_l8.py`, `code/judge_expect.py`, `code/expect_table.py` | The two Llama BF16 sets of steps 7 and 8 |
| `code/judge_sample.py` | Step 9: `gen`, `judge MODEL`, `table` |
| `code/*.sh` | Launchers for the steps above |

One record of `results/{model}_{format}.jsonl` is one pass over the 100 prompts: the fault (`layer`, `tensor`, `region`, `block`, the random bytes `rnd`), the case (`mode`), the keyword counts (`refused`, `coherent`, `garbled`) and the 100 answers.

## Limits

- The models run in Transformers with BF16 arithmetic, not in vLLM or llama.cpp. The quantized formats are simulated by decoding the faulted bytes back into weights, so the simulation assumes that an engine keeps codes and scales in memory as stored.
- The byte layouts of `int8` and `int4` are one choice among several in use.
- Every trial lands exactly one fault in the chosen tensor.
- The answer files contain model output for harmful requests. They are not part of this repository; the scripts recreate them.
