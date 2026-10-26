import json
import random
import re
import string
from collections import Counter, defaultdict

SEED = 42
TARGET = 500           # items per condition
MIN_DISTRACTORS = 10   # inconsistent items per subset kept back as distractor pool
random.seed(SEED)

def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]

def normalise(text):
    text = text.lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())

def contains_answer(passage, answer):
    p, a = normalise(passage), normalise(answer)
    return bool(a) and re.search(r"\b" + re.escape(a) + r"\b", p) is not None

def split_docs(context):
    return [d.strip() for d in context.split("Document:") if d.strip()]

def format_context(passages):
    return "\n\n".join(f"Document: {p}" for p in passages)

def dedupe(items):
    seen, kept = set(), []
    for item in items:
        key = normalise(item["question"])
        if key not in seen:
            seen.add(key)
            kept.append(item)
    return kept

def by_subset(items):
    groups = defaultdict(list)
    for item in items:
        groups[item["subset"]].append(item)
    for s in groups:
        random.shuffle(groups[s])
    return groups

def allocate(caps, target):
    """Spread `target` evenly over subsets, never exceeding a subset's cap."""
    alloc = {s: 0 for s in caps}
    remaining = target
    while remaining > 0:
        open_subsets = [s for s in sorted(caps) if alloc[s] < caps[s]]
        if not open_subsets:
            raise ValueError(f"Not enough items: short by {remaining}")
        for s in open_subsets:
            if remaining == 0:
                break
            alloc[s] += 1
            remaining -= 1
    return alloc

# ---------- load + dedupe ----------
inc_raw = load_jsonl("data/raw/inconsistent_test.jsonl")
una_raw = load_jsonl("data/raw/unanswerable_test.jsonl")
inc, una = dedupe(inc_raw), dedupe(una_raw)
print(f"Dedup: inconsistent {len(inc_raw)} -> {len(inc)}, unanswerable {len(una_raw)} -> {len(una)}")

inc_by, una_by = by_subset(inc), by_subset(una)

# ---------- inconsistent + control (same pool, disjoint groups) ----------
ic_caps = {s: (len(v) - MIN_DISTRACTORS) // 2 for s, v in inc_by.items()}
ic_alloc = allocate(ic_caps, TARGET)

groups = {}
for s, items in inc_by.items():
    n = ic_alloc[s]
    groups[s] = {"A": items[:n], "B": items[n:2 * n], "C": items[2 * n:]}

used_qids = {i["qid"] for g in groups.values() for i in g["A"] + g["B"]}
used_qs = {normalise(i["question"]) for g in groups.values() for i in g["A"] + g["B"]}

# ---------- unanswerable (exclude any used question, globally) ----------
una_avail = {
    s: [i for i in v if i["qid"] not in used_qids and normalise(i["question"]) not in used_qs]
    for s, v in una_by.items()
}
una_alloc = allocate({s: len(v) for s, v in una_avail.items()}, TARGET)

# ---------- build ----------
sample = []
distractor_use = Counter()

for s in sorted(groups):
    for item in groups[s]["A"]:
        docs = split_docs(item["context"])
        sample.append({
            "item_id": f"inc_{item['qid']}", "condition": "inconsistent", "subset": s,
            "question": item["question"], "passages": docs, "context": format_context(docs),
            "gold_answers": item["answers"], "source_qid": item["qid"],
        })

    for item in groups[s]["B"]:
        original = split_docs(item["context"])[0]
        gold = item["answers"][0]
        pool = groups[s]["C"][:]
        random.shuffle(pool)
        cand = next((c for c in pool if not contains_answer(split_docs(c["context"])[0], gold)), None)
        assert cand is not None, f"No valid distractor for {item['qid']} in {s}"
        distractor_use[cand["qid"]] += 1
        passages = [original, split_docs(cand["context"])[0]]
        sample.append({
            "item_id": f"ctl_{item['qid']}", "condition": "control", "subset": s,
            "question": item["question"], "passages": passages, "context": format_context(passages),
            "gold_answers": [gold], "source_qid": item["qid"], "distractor_qid": cand["qid"],
        })

    for item in una_avail.get(s, [])[:una_alloc.get(s, 0)]:
        passages = [item["context"].strip()]
        sample.append({
            "item_id": f"una_{item['qid']}", "condition": "unanswerable", "subset": s,
            "question": item["question"], "passages": passages, "context": format_context(passages),
            "gold_answers": item["answers"], "source_qid": item["qid"],
        })

random.shuffle(sample)

with open("data/processed/sample.jsonl", "w") as f:
    for row in sample:
        f.write(json.dumps(row) + "\n")
with open("data/processed/sample_ids.txt", "w") as f:
    f.write("\n".join(r["item_id"] for r in sample) + "\n")

# ---------- report + hard checks ----------
print("\nAllocation per subset (inconsistent / control / unanswerable):")
for s in sorted(groups):
    print(f"  {s:22s} {ic_alloc[s]:3d} / {ic_alloc[s]:3d} / {una_alloc.get(s, 0):3d}")

counts = Counter(r["condition"] for r in sample)
print("\nBy condition:", dict(counts))
print("Max times one distractor reused:", max(distractor_use.values()))

assert all(counts[c] == TARGET for c in ("inconsistent", "control", "unanswerable")), counts
assert len({r["source_qid"] for r in sample}) == len(sample), "duplicate qids"
assert len({normalise(r["question"]) for r in sample}) == len(sample), "duplicate questions"
assert all(p for r in sample for p in r["passages"]), "empty passage"
print("\nALL CHECKS PASSED")