import json
from collections import Counter, defaultdict

def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]

inc = load_jsonl("data/raw/inconsistent_test.jsonl")

def classify(item):
    docs = [d.strip() for d in item["context"].split("Document:") if d.strip()]
    d1, d2 = docs[0].lower(), docs[1].lower()
    a0, a1 = item["answers"][0].lower(), item["answers"][1].lower()

    a0_in1, a0_in2 = a0 in d1, a0 in d2
    a1_in1, a1_in2 = a1 in d1, a1 in d2

    if a0_in1 and a1_in2 and not a0_in2:
        return "clean"
    if a0_in2 and a1_in1 and not a0_in1:
        return "swapped_order"
    if not (a0_in1 or a0_in2):
        return "orig_answer_missing"
    if not (a1_in1 or a1_in2):
        return "new_answer_missing"
    if a0_in1 and a0_in2:
        return "orig_answer_in_both_docs"
    return "other"

counts = Counter()
by_subset = defaultdict(Counter)
examples = {}

for item in inc:
    label = classify(item)
    counts[label] += 1
    by_subset[item["subset"]][label] += 1
    examples.setdefault(label, item)

print("Overall:", dict(counts))
print("\nBy subset:")
for subset, c in by_subset.items():
    print(f"  {subset}: {dict(c)}")

print("\nOne example per failure type:")
for label, item in examples.items():
    if label == "clean":
        continue
    print(f"\n--- {label} ({item['subset']}) ---")
    print("Q:", item["question"])
    print("Answers:", item["answers"])
    print("Context start:", item["context"][:400])