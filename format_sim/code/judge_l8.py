"""Third judging set: Llama BF16, layer 8, every tensor except v_proj, whole and tail:
the median-refusal block and the block with the fewest keyword refusals among those with <=10 garbled answers.
-> results/judge/answers/llama_bf16.p3.jsonl (then: judge160.py judge MODEL)."""
import json, os, sys
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, f"{ROOT}/code")
import sim
rows = [json.loads(l) for l in open(f"{ROOT}/results/llama_bf16.jsonl")]
sel = {}
for mode in ("whole", "tail"):
    for tn in ("down_proj", "up_proj", "o_proj", "gate_proj", "q_proj", "k_proj"):
        rs = sorted((r for r in rows if r.get("layer") == 8 and r["exp"] in ("layers", "tensors") and r["mode"] == mode and r["tensor"] == tn),
                    key=lambda r: r["refused"])
        ok = [r for r in rs if r["garbled"] <= 10]
        sel[f"L8:{tn}:{mode}:median"] = rs[len(rs) // 2]
        sel[f"L8:{tn}:{mode}:min"] = ok[0]
        sel[f"L8:{tn}:{mode}:min2"] = ok[1]
R = sim.Runner("llama", "bf16", 160)
_, cache = R.prefill_whole()
R.clean_kv = [(k.clone(), v.clone()) for k, v in R._kv(cache)]
dst = f"{ROOT}/results/judge/answers/llama_bf16.p3.jsonl"
seen = {}
with open(dst + ".tmp", "w") as out:
    for tag, r in sel.items():
        if r["key"] in seen:
            print(tag, "same fault as", seen[r["key"]]); continue
        seen[r["key"]] = tag
        rnd = np.frombuffer(bytes.fromhex(r["rnd"]), dtype=np.uint8).copy()
        res = R.run(r["mode"], (r["layer"], r["tensor"], r["region"], r["block"], rnd))
        out.write(json.dumps(dict(tag=tag, model="llama", fmt="bf16", **{k: r[k] for k in ("mode", "layer", "tensor", "region", "block", "rnd", "key")},
                                  kw48_refused=r["refused"], kw_refused=res["refused"], kw_coherent=res["coherent"], kw_garbled=res["garbled"], answers=res["answers"])) + "\n")
os.rename(dst + ".tmp", dst)
print("gen done", len(seen))
