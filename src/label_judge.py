import json
import os
import random
import textwrap
from collections import defaultdict

SEED = 11
PER_CONDITION = {"control": 17, "inconsistent": 17, "unanswerable": 16}   # = 50
SIGNALS = "data/processed/signals.jsonl"
SAMPLE = "data/processed/sample.jsonl"
OUT_PATH = "data/labels/judge_manual.jsonl"

os.makedirs("data/labels", exist_ok=True)
random.seed(SEED)

sample = {}
for l in open(SAMPLE):
    r = json.loads(l)
    sample[r["item_id"]] = r

by_cond = defaultdict(list)
for l in open(SIGNALS):
    r = json.loads(l)
    by_cond[r["condition"]].append(r)

picked = []
for cond, n in PER_CONDITION.items():
    pool = sorted(by_cond[cond], key=lambda r: r["item_id"])
    picked += random.sample(pool, n)
random.shuffle(picked)

done = set()
if os.path.exists(OUT_PATH):
    done = {json.loads(l)["item_id"] for l in open(OUT_PATH)}

print("You judge the MODEL'S ANSWER. The LLM judge's decision is hidden.\n"
      "q = quit (progress saved), s = skip for now\n")
with open(OUT_PATH, "a") as f:
    for n, r in enumerate(picked, 1):
        if r["item_id"] in done:
            continue
        print("=" * 90)
        print(f"ITEM {n}/{len(picked)}\n")
        print(f"QUESTION: {sample[r['item_id']]['question']}")
        print("MODEL ANSWER:")
        print(textwrap.fill(r["answer"], width=90))
        if r["condition"] == "control":
            print(f"\nGROUND TRUTH: {sample[r['item_id']]['gold_answers']}")
            prompt = "Does the answer match the ground truth in meaning? [1 = correct / 0 = wrong / s / q]: "
            valid = {"1", "0"}
        else:
            prompt = ("Does the answer COMMIT to one specific answer? "
                      "[c = commits / a = abstains or flags conflict / s / q]: ")
            valid = {"c", "a"}
        while True:
            ans = input("\n" + prompt).strip().lower()
            if ans in valid | {"s", "q"}:
                break
        if ans == "q":
            break
        if ans == "s":
            continue
        if r["condition"] == "control":
            correct = int(ans)
        else:
            correct = int(ans == "a")      # same rule as the pipeline: committing = wrong
        f.write(json.dumps({"item_id": r["item_id"], "condition": r["condition"],
                            "human_correct": correct}) + "\n")
        f.flush()

print(f"\nLabelled {sum(1 for _ in open(OUT_PATH))}/{len(picked)}. Re-run to continue.")