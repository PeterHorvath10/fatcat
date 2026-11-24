"""Regenerate selected fault outcomes with 160-token answers and label them with the paper's LLM judges.

Step 1 (GPU):   judge160.py gen            -> results/judge/answers/{model}_{fmt}.jsonl (one line per outcome)
Step 2 (judge): judge160.py judge MODEL    -> results/judge/labels/{MODEL}/{model}_{fmt}.jsonl
Step 3:         judge160.py table          -> results/JUDGED.md
"""
import glob, json, os, sys
from concurrent.futures import ThreadPoolExecutor
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, f"{ROOT}/code"); sys.path.insert(0, f"{ROOT}/../jailbreak_emfi")
FMTS = ["bf16", "fp8", "int8", "int4", "nvfp4", "mxfp4", "gguf-Q4_K_M", "gguf-Q4_0", "gguf-Q8_0"]
J = f"{ROOT}/results/judge"


def select(rows):
    f = [r for r in rows if r["exp"] != "clean"]
    ok = lambda r: r["garbled"] <= 10
    sel = {}
    code = sorted((r for r in f if r["exp"] in ("layers", "tensors") and r["mode"] == "whole" and not r.get("hit_scale") and ok(r)),
                  key=lambda r: r["refused"])[:2]
    for i, r in enumerate(code):
        sel[f"code{i + 1}"] = r
    for mode in ("whole", "tail"):
        groups = {}
        for r in f:
            if r["exp"] == "scales" and r["mode"] == mode and ok(r):
                k = (r["region_req"], r["layer"])
                if k not in groups or r["refused"] < groups[k]["refused"]:
                    groups[k] = r
        for (reg, l), r in sorted(groups.items()):
            sel[f"scale:{reg}:L{l}:{mode}"] = r
    if rows and rows[0]["fmt"] == "bf16":
        for r in f:
            if r["exp"] == "layers" and r["mode"] == "whole" and r["draw"] == 0:
                sel[f"layer{r['layer']:02d}"] = r
    return sel


def select2(rows):
    """Second pass (all tensors, after the extended sweep). Per tensor and mode: best scale fault of each scale
    region (any layer); best random-place fault for bf16 (2, from different layers) and the GGUF files with inline
    scales; for the other formats the 2 best random-place faults per mode over all tensors; bf16 also draw 0 of
    every layer for down_proj, tail. 'Best' = fewest keyword refusals among faults with <=10 garbled answers."""
    f = [r for r in rows if r["exp"] != "clean" and r["garbled"] <= 10]
    fmt = rows[0]["fmt"]
    sel = {}
    best = lambda rs, n=1: sorted(rs, key=lambda r: r["refused"])[:n]
    tensors = sorted({r["tensor"] for r in f})
    for mode in ("whole", "tail"):
        rand = [r for r in f if r["exp"] in ("layers", "tensors") and r["mode"] == mode]
        for tn in tensors:
            for reg in sorted({r["region_req"] for r in f if r["exp"] == "scales"}):
                if fmt == "nvfp4" and reg == "bscale":
                    continue
                for r in best([r for r in f if r["exp"] == "scales" and r["mode"] == mode and r["tensor"] == tn and r["region_req"] == reg]):
                    sel[f"scale:{reg}:{tn}:{mode}"] = r
            if fmt in ("bf16", "gguf-Q4_0", "gguf-Q8_0"):
                seen = []
                for r in best([r for r in rand if r["tensor"] == tn], 10 ** 6):
                    if r["layer"] not in seen:
                        seen.append(r["layer"])
                        sel[f"rand:{tn}:{mode}:{len(seen)}"] = r
                    if len(seen) == (2 if fmt == "bf16" else 1):
                        break
        if fmt not in ("bf16", "gguf-Q4_0", "gguf-Q8_0"):
            for i, r in enumerate(best(rand, 2)):
                sel[f"rand:any:{mode}:{i + 1}"] = r
    if fmt == "bf16":
        for r in rows:
            if r["exp"] == "layers" and r["mode"] == "tail" and r["draw"] == 0:
                sel[f"layer{r['layer']:02d}:tail"] = r
    return sel


def gen(p2=False):
    import numpy as np, torch, sim
    os.makedirs(f"{J}/answers", exist_ok=True)
    for m in ("llama", "gemma"):
        for fmt in FMTS:
            src, dst = f"{ROOT}/results/{m}_{fmt}.jsonl", f"{J}/answers/{m}_{fmt}{'.p2' if p2 else ''}.jsonl"
            log = f"{ROOT}/results/{m}_{fmt}{'.ext' if p2 else ''}.log"
            if not os.path.exists(src) or os.path.exists(dst) or "done " not in open(log).read():
                continue
            sel = (select2 if p2 else select)([json.loads(l) for l in open(src)])
            R = sim.Runner(m, fmt, 160)
            _, cache = R.prefill_whole()
            R.clean_kv = [(k.clone(), v.clone()) for k, v in R._kv(cache)]
            with open(dst + ".tmp", "w") as out:
                def emit(tag, meta, res):
                    out.write(json.dumps(dict(tag=tag, model=m, fmt=fmt, **meta, kw_refused=res["refused"], kw_coherent=res["coherent"],
                                              kw_garbled=res["garbled"], answers=res["answers"])) + "\n")
                if not p2:
                    emit("clean", {}, R.run("whole"))
                for tag, r in sel.items():
                    rnd = np.frombuffer(bytes.fromhex(r["rnd"]), dtype=np.uint8).copy()
                    res = R.run(r["mode"], (r["layer"], r["tensor"], r["region"], r["block"], rnd))
                    emit(tag, {k: r[k] for k in ("mode", "layer", "tensor", "region", "block", "rnd", "key")}, res)
            os.rename(dst + ".tmp", dst)
            print("gen", m, fmt, 1 + len(sel), flush=True)
            del R; torch.cuda.empty_cache()


def judge(model):
    """Labels every answer; resumable per answer: only missing or UNCLEAR labels are asked (again).
    6 requests at a time, with a pause before a retry: at 24 at a time most requests failed."""
    import csv, time
    from common import config
    from analysis.judge.judge import Judge
    P = Judge(config.load("analysis.toml"), model)
    d = f"{J}/labels/{model.replace(':', '_')}"
    os.makedirs(d, exist_ok=True)
    goals = [r["Goal"] for r in csv.DictReader(open(f"{ROOT}/../jailbreak_emfi/datasets/jailbreakbench/harmful-behaviors.csv"))]

    def ask(goal, answer):
        for attempt in range(6):
            lab, raw = P.ask(goal, answer)
            if lab != "UNCLEAR":
                return lab, raw
            time.sleep(5 * (attempt + 1))
        return lab, raw

    for path in sorted(glob.glob(f"{J}/answers/*.jsonl")):
        out = f"{d}/{os.path.basename(path)}"
        done = {r["tag"]: r for r in map(json.loads, open(out))} if os.path.exists(out) else {}
        with ThreadPoolExecutor(6) as ex:
            for rec in map(json.loads, open(path)):
                old = done.get(rec["tag"], {"labels": ["UNCLEAR"] * len(goals)})
                labels, raws = list(old["labels"]), list(old.get("raw", [""] * len(goals)))
                todo = [i for i, l in enumerate(labels) if l == "UNCLEAR"]
                if not todo:
                    continue
                for i, (lab, raw) in zip(todo, ex.map(lambda i: ask(goals[i], rec["answers"][i]), todo)):
                    labels[i], raws[i] = lab, raw
                done[rec["tag"]] = dict(tag=rec["tag"], labels=labels, raw=raws)
                with open(out + ".tmp", "w") as fh:
                    for r in done.values():
                        fh.write(json.dumps(r) + "\n")
                os.replace(out + ".tmp", out)
                c = {k: labels.count(k) for k in ("REFUSED", "UNSAFE", "OTHER", "UNCLEAR")}
                print(model, os.path.basename(path), rec["tag"], "asked", len(todo), c, flush=True)


def table():
    judges = sorted(os.listdir(f"{J}/labels"))
    out = ["# Judged outcomes (160-token answers, the paper's rubric)\n",
           "R/U/O = refused / unsafe / other out of 100. Faults were picked by the keyword score (fewest refusals with <=10 garbled), "
           "so these are best-case faults, not averages. kw: keyword refused/coherent/garbled at 160 tokens.\n"]
    for m in ("llama", "gemma"):
        out.append(f"\n## {m}\n")
        out.append("| Format | Outcome | Fault | kw R/C/G | " + " | ".join(f"{j} R/U/O" for j in judges) + " |")
        out.append("|---|---|---|---|" + "---|" * len(judges))
        for fmt, sfx in [(f, x) for f in FMTS for x in ("", ".p2")]:
            p = f"{J}/answers/{m}_{fmt}{sfx}.jsonl"
            if not os.path.exists(p):
                continue
            lab = {}
            for j in judges:
                q = f"{J}/labels/{j}/{m}_{fmt}{sfx}.jsonl"
                lab[j] = {r["tag"]: r["labels"] for r in map(json.loads, open(q))} if os.path.exists(q) else {}
            for rec in map(json.loads, open(p)):
                fault = "–" if rec["tag"] == "clean" else f"L{rec['layer']} {rec['tensor']} {rec['region']} {rec['mode']}"
                cells = []
                for j in judges:
                    l = lab[j].get(rec["tag"])
                    cells.append("…" if l is None else f"{l.count('REFUSED')}/{l.count('UNSAFE')}/{l.count('OTHER')}" + (f" ({l.count('UNCLEAR')}?)" if l.count("UNCLEAR") else ""))
                out.append(f"| {fmt} | {rec['tag']} | {fault} | {rec['kw_refused']}/{rec['kw_coherent']}/{rec['kw_garbled']} | " + " | ".join(cells) + " |")
    open(f"{ROOT}/results/JUDGED.md", "w").write("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    {"gen": gen, "gen2": lambda: gen(True), "judge": lambda: judge(sys.argv[2]), "table": table}[sys.argv[1]]()
