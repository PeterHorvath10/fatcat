"""Fourth judging set: Llama BF16, all 20 random blocks of the most promising targets, to get the expected
(not best-case) judged outcome of one fault. Blocks with identical 48-token answers are judged once (weight = count).
-> results/judge/answers/llama_bf16.p4.jsonl"""
import json, os, sys
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, f"{ROOT}/code")
import sim
rows = [json.loads(l) for l in open(f"{ROOT}/results/llama_bf16.jsonl")]
sel = {}
for tn, l in (("k_proj", 12), ("k_proj", 11), ("down_proj", 10), ("down_proj", 8)):
    for mode in ("whole", "tail"):
        groups = {}
        for r in rows:
            if r.get("layer") == l and r.get("tensor") == tn and r["mode"] == mode and r["exp"] in ("layers", "tensors"):
                groups.setdefault(tuple(r["answers"]), []).append(r)
        for i, rs in enumerate(sorted(groups.values(), key=lambda rs: -len(rs))):
            sel[f"E:L{l}:{tn}:{mode}:{i:02d}"] = (rs[0], len(rs))
R = sim.Runner("llama", "bf16", 160)
_, cache = R.prefill_whole()
R.clean_kv = [(k.clone(), v.clone()) for k, v in R._kv(cache)]
dst = f"{ROOT}/results/judge/answers/llama_bf16.p4.jsonl"
dead = []
with open(dst + ".tmp", "w") as out:
    for tag, (r, w) in sel.items():
        if r["garbled"] >= 90:
            dead.append(dict(tag=tag, weight=w, **{k: r[k] for k in ("mode", "layer", "tensor", "key")}))
            continue
        rnd = np.frombuffer(bytes.fromhex(r["rnd"]), dtype=np.uint8).copy()
        res = R.run(r["mode"], (r["layer"], r["tensor"], r["region"], r["block"], rnd))
        out.write(json.dumps(dict(tag=tag, weight=w, model="llama", fmt="bf16", **{k: r[k] for k in ("mode", "layer", "tensor", "region", "block", "rnd", "key")},
                                  kw48_refused=r["refused"], kw_refused=res["refused"], kw_coherent=res["coherent"], kw_garbled=res["garbled"], answers=res["answers"])) + "\n")
os.rename(dst + ".tmp", dst)
json.dump(dead, open(f"{ROOT}/results/judge/p4_dead.json", "w"))
print("gen done", len(sel))
