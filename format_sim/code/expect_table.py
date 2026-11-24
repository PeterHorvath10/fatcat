"""Expected judged outcome per pulse (Llama BF16, p4): weighted over the 20 random blocks of each target."""
import json, collections, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
A = [json.loads(l) for l in open(f"{ROOT}/results/judge/answers/llama_bf16.p4.jsonl")]
dead = json.load(open(f"{ROOT}/results/judge/p4_dead.json"))
J = ("deepseek-v4.1-flash_cloud", "glm-5.3_cloud")
lab = {j: {r["tag"]: r["labels"] for r in map(json.loads, open(f"{ROOT}/results/judge/labels/{j}/llama_bf16.p4.jsonl"))} for j in J}
groups = collections.defaultdict(list)
for a in A:
    groups[(a["layer"], a["tensor"], a["mode"])].append(a)
dw = collections.Counter()
for d in dead:
    dw[(d["layer"], d["tensor"], d["mode"])] += d["weight"]
print("| Target | Prefill | blocks judged | mean unsafe D / G | median D / G | max D / G | blocks with ≥20 unsafe (D / G) | mean refused D / G |")
print("|---|---|---|---|---|---|---|---|")
for k in sorted(groups, key=lambda k: (-k[0], k[2] == "tail")):
    cells = {}
    done = None
    for j in J:
        u, rf = [], []
        for a in groups[k]:
            if a["tag"] in lab[j]:
                c = collections.Counter(lab[j][a["tag"]])
                u += [c["UNSAFE"]] * a["weight"]; rf += [c["REFUSED"]] * a["weight"]
        u += [0] * dw[k]; rf += [0] * dw[k]
        cells[j] = (u, rf)
    n = [len(cells[j][0]) for j in J]
    f = lambda fn: " / ".join(fn(cells[j]) for j in J)
    print(f"| L{k[0]} {k[1]} | {k[2]} | {n[0]} / {n[1]} of 20 | " + f(lambda c: f"{sum(c[0])/max(1,len(c[0])):.1f}") + " | "
          + f(lambda c: str(sorted(c[0])[len(c[0])//2]) if c[0] else "–") + " | " + f(lambda c: str(max(c[0])) if c[0] else "–") + " | "
          + f(lambda c: str(sum(x >= 20 for x in c[0]))) + " | " + f(lambda c: f"{sum(c[1])/max(1,len(c[1])):.0f}") + " |")
