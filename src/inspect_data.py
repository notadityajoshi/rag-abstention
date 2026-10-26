import json
from collections import Counter

def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]

inc = load_jsonl("data/raw/inconsistent_test.jsonl")
una = load_jsonl("data/raw/unanswerable_test.jsonl")

# 1. How many "Document:" markers per inconsistent context?
doc_counts = Counter(item["context"].count("Document:") for item in inc)
print("Inconsistent - number of 'Document:' markers per item:", dict(doc_counts))

# 2. How many answers per inconsistent item?
ans_counts = Counter(len(item["answers"]) for item in inc)
print("Inconsistent - answers per item:", dict(ans_counts))

# 3. Does each answer appear in exactly one of the two documents?
ok = 0
for item in inc:
    docs = [d.strip() for d in item["context"].split("Document:") if d.strip()]
    if len(docs) == 2:
        a0, a1 = item["answers"][0], item["answers"][1]
        if a0 in docs[0] and a1 in docs[1] and a0 not in docs[1]:
            ok += 1
print(f"Inconsistent - clean split (orig answer in doc1 only, new answer in doc2): {ok}/{len(inc)}")

# 4. Source datasets
print("\nInconsistent subsets:", dict(Counter(i["subset"] for i in inc)))
print("Unanswerable subsets:", dict(Counter(i["subset"] for i in una)))

# 5. Unanswerable: any 'Document:' markers?
print("\nUnanswerable - items containing 'Document:':",
      sum("Document:" in i["context"] for i in una))

# 6. Context length (words) - affects NLI cost
def avg_words(items):
    return sum(len(i["context"].split()) for i in items) / len(items)
print(f"Avg context words - inconsistent: {avg_words(inc):.0f}, unanswerable: {avg_words(una):.0f}")