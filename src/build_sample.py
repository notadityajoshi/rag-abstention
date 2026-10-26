import json
import random
import re
import string
from collections import Counter, defaultdict

random.seed(42)
PER_SUBSET = 50

def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]

def normalise(text):
    """Lowercase, strip punctuation and articles, collapse spaces (SQuAD-style)."""
    text = text.lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())

def contains_answer(passage, answer):
    """Whole-word match, so 'sky' does NOT match 'bskyb' and '2' does NOT match '2007'."""
    p, a = normalise(passage), normalise(answer)
    if not a:
        return False
    return re.search(r"\b" + re.escape(a) + r"\b", p) is not None

def split_docs(context):
    return [d.strip() for d in context.split("Document:") if d.strip()]

def format_context(passages):
    return "\n\n".join(f"Document: {p}" for p in passages)

inc = load_jsonl("data/raw/inconsistent_test.jsonl")
una = load_jsonl("data/raw/unanswerable_test.jsonl")

inc_by_subset = defaultdict(list)
for item in inc:
    inc_by_subset[item["subset"]].append(item)
una_by_subset = defaultdict(list)
for item in una:
    una_by_subset[item["subset"]].append(item)

sample = []
rejected_distractors = 0

for subset in sorted(inc_by_subset):
    items = inc_by_subset[subset][:]
    random.shuffle(items)
    group_a = items[:PER_SUBSET]                    # inconsistent condition
    group_b = items[PER_SUBSET:2 * PER_SUBSET]      # control questions
    group_c = items[2 * PER_SUBSET:]                # distractor pool

    # Inconsistent: original + edited passage
    for item in group_a:
        docs = split_docs(item["context"])
        sample.append({
            "item_id": f"inc_{item['qid']}",
            "condition": "inconsistent",
            "subset": subset,
            "question": item["question"],
            "passages": docs,
            "context": format_context(docs),
            "gold_answers": item["answers"],        # [original, edited]
            "source_qid": item["qid"],
        })

    # Answerable control: original passage + unrelated distractor
    for item in group_b:
        original = split_docs(item["context"])[0]
        gold = item["answers"][0]
        pool = group_c[:]
        random.shuffle(pool)
        distractor, distractor_qid = None, None
        for cand in pool:
            cand_doc = split_docs(cand["context"])[0]
            if not contains_answer(cand_doc, gold):
                distractor, distractor_qid = cand_doc, cand["qid"]
                break
            rejected_distractors += 1
        passages = [original, distractor]
        sample.append({
            "item_id": f"ctl_{item['qid']}",
            "condition": "control",
            "subset": subset,
            "question": item["question"],
            "passages": passages,
            "context": format_context(passages),
            "gold_answers": [gold],
            "source_qid": item["qid"],
            "distractor_qid": distractor_qid,
        })

    # Unanswerable: single passage
    u_items = una_by_subset[subset][:]
    random.shuffle(u_items)
    for item in u_items[:PER_SUBSET]:
        passages = [item["context"].strip()]
        sample.append({
            "item_id": f"una_{item['qid']}",
            "condition": "unanswerable",
            "subset": subset,
            "question": item["question"],
            "passages": passages,
            "context": format_context(passages),
            "gold_answers": item["answers"],        # abstention phrases
            "source_qid": item["qid"],
        })

random.shuffle(sample)

with open("data/processed/sample.jsonl", "w") as f:
    for row in sample:
        f.write(json.dumps(row) + "\n")

with open("data/processed/sample_ids.txt", "w") as f:
    for row in sample:
        f.write(row["item_id"] + "\n")

# Sanity checks
print("Total items:", len(sample))
print("By condition:", dict(Counter(r["condition"] for r in sample)))
print("Distractors rejected for containing gold answer:", rejected_distractors)
print("Duplicate source questions across conditions:",
      len(sample) - len({r["source_qid"] for r in sample}))
print("Controls missing a distractor:",
      sum(1 for r in sample if r["condition"] == "control" and r["passages"][1] is None))

print("\nExample control item:")
ex = next(r for r in sample if r["condition"] == "control")
print("Q:", ex["question"], "| Gold:", ex["gold_answers"])
print(ex["context"][:700])