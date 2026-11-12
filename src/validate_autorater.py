import json
from collections import defaultdict

from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix

manual = {}
for line in open("data/labels/sufficiency_manual.jsonl"):
    r = json.loads(line)
    manual[r["item_id"]] = r["label"]

auto, cond = {}, {}
for line in open("data/processed/signals.jsonl"):
    r = json.loads(line)
    auto[r["item_id"]] = r["sufficiency"]
    cond[r["item_id"]] = r["condition"]

ids = [i for i in manual if auto.get(i) is not None]
y_h = [manual[i] for i in ids]
y_a = [auto[i] for i in ids]

print(f"Items compared: {len(ids)}/50")
print(f"Accuracy (agreement): {accuracy_score(y_h, y_a):.3f}")
print(f"Cohen's kappa:        {cohen_kappa_score(y_h, y_a):.3f}")
print("Confusion matrix (rows = you, cols = autorater; order 0, 1):")
print(confusion_matrix(y_h, y_a, labels=[0, 1]))

by = defaultdict(list)
for i in ids:
    by[cond[i]].append(manual[i] == auto[i])
print("\nAgreement by condition:")
for c, v in sorted(by.items()):
    print(f"  {c:13s} n={len(v):2d}  agree={sum(v)/len(v):.2f}")

print("\nDisagreements (item_id | you | autorater):")
for i in ids:
    if manual[i] != auto[i]:
        print(f"  {i} | {manual[i]} | {auto[i]}")