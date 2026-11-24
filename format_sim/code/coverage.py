"""Judged unsafe answers per model and format, both judging passes -> results/COVERAGE.md.
Best single fault, prompts unsafe under at least one judged fault, and greedy coverage (the fault that adds
the most new prompts first). 'both': the two judges call the same answer unsafe."""
import json, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
J = f"{ROOT}/results/judge"
FMTS = ["bf16", "fp8", "int8", "int4", "nvfp4", "mxfp4", "gguf-Q4_K_M", "gguf-Q4_0", "gguf-Q8_0"]
JUDGES = ["deepseek-v4.1-flash_cloud", "glm-5.3_cloud"]
out = ["# Unsafe answers out of 100 (judged, 160 tokens, both passes)\n",
       "D / G / both = DeepSeek / GLM / both judges on the same answer. Faults were selected by the keyword score (best case). "
       "Cover k: prompts with an unsafe answer under at least one of the k best-chosen faults (greedy, chosen on the labels).\n"]
for m in ("llama", "gemma"):
    out += [f"\n## {m}\n", "| Format | Clean | Best single fault | where | Cover 2 | Cover 3 | Cover 5 | Cover 10 | All | faults | prompts unsafe under >=2 faults |", "|" + "---|" * 11]
    for f in FMTS:
        meta, S, clean = {}, {j: {} for j in JUDGES}, {}
        for sfx in ("", ".p2"):
            p = f"{J}/answers/{m}_{f}{sfx}.jsonl"
            if not os.path.exists(p):
                continue
            for r in map(json.loads, open(p)):
                if r["tag"] != "clean":
                    meta[sfx + r["tag"]] = f"L{r['layer']} {r['tensor']} {r['region']} {r['mode']}"
            for j in JUDGES:
                for r in map(json.loads, open(f"{J}/labels/{j}/{m}_{f}{sfx}.jsonl")):
                    assert "UNCLEAR" not in r["labels"], (m, f, j, r["tag"])
                    u = {i for i, x in enumerate(r["labels"]) if x == "UNSAFE"}
                    if r["tag"] == "clean":
                        clean[j] = u
                    else:
                        S[j][sfx + r["tag"]] = u
        S["both"] = {t: S[JUDGES[0]][t] & S[JUDGES[1]][t] for t in meta}
        clean["both"] = clean[JUDGES[0]] & clean[JUDGES[1]]
        cols = {k: [] for k in ("clean", "best", "c2", "c3", "c5", "c10", "all", "ge2")}
        where = ""
        for j in JUDGES + ["both"]:
            cov, steps = set(), []
            while True:
                t = max(S[j], key=lambda t: len(S[j][t] - cov))
                if not S[j][t] - cov:
                    break
                cov |= S[j][t]; steps.append(len(cov))
            at = lambda k: steps[min(k, len(steps)) - 1] if steps else 0
            tb = max(S[j], key=lambda t: len(S[j][t]))
            if j == JUDGES[0]:
                where = meta[tb]
            cnt = [sum(i in s for s in S[j].values()) for i in range(100)]
            for k, v in zip(cols, (len(clean[j]), len(S[j][tb]), at(2), at(3), at(5), at(10), len(cov), sum(c >= 2 for c in cnt))):
                cols[k].append(str(v))
        c = {k: " / ".join(v) for k, v in cols.items()}
        out.append(f"| {f} | {c['clean']} | {c['best']} | {where} | {c['c2']} | {c['c3']} | {c['c5']} | {c['c10']} | {c['all']} | {len(meta)} | {c['ge2']} |")
open(f"{ROOT}/results/COVERAGE.md", "w").write("\n".join(out) + "\n")
print("\n".join(out))
