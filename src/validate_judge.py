import json
from collections import defaultdict

from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix

human = {}
for l in open("data/labels/judge_manual.jsonl"):
    r = json.loads(l)
    human[r["item_id"]] = r["human_correct"]

llm, cond, answer = {}, {}, {}
for l in open("data/processed/signals.jsonl"):
    r = json.loads(l)
    llm[r["item_id"]], cond[r["item_id"]], answer[r["item_id"]] = r["correct"], r["condition"], r["answer"]

ids = [i for i in human if llm.get(i) is not None]
y_h = [human[i] for i in ids]
y_l = [llm[i] for i in ids]

print(f"Items compared: {len(ids)}/50")
print(f"Accuracy (agreement): {accuracy_score(y_h, y_l):.3f}")
print(f"Cohen's kappa:        {cohen_kappa_score(y_h, y_l):.3f}")
print("Confusion matrix (rows = you, cols = LLM judge; order 0, 1):")
print(confusion_matrix(y_h, y_l, labels=[0, 1]))

by = defaultdict(list)
for i in ids:
    by[cond[i]].append(human[i] == llm[i])
print("\nAgreement by condition:")
for c, v in sorted(by.items()):
    print(f"  {c:13s} n={len(v):2d}  agree={sum(v)/len(v):.2f}")

print("\nDisagreements (item_id | you | judge | answer):")
for i in ids:
    if human[i] != llm[i]:
        print(f"  {i} | {human[i]} | {llm[i]} | {answer[i][:100]}")