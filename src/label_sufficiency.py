import json
import os
import random
import re
import textwrap
from collections import defaultdict

SEED = 7                      # different seed from sampling, fixed for reproducibility
PER_CONDITION = {"inconsistent": 17, "control": 17, "unanswerable": 16}   # = 50
IN_PATH = "data/processed/sample.jsonl"
OUT_PATH = "data/labels/sufficiency_manual.jsonl"

os.makedirs("data/labels", exist_ok=True)
random.seed(SEED)

items = [json.loads(l) for l in open(IN_PATH)]
by_cond = defaultdict(list)
for it in items:
    by_cond[it["condition"]].append(it)

picked = []
for cond, n in PER_CONDITION.items():
    pool = sorted(by_cond[cond], key=lambda r: r["item_id"])   # fixed order before sampling
    picked += random.sample(pool, n)
random.shuffle(picked)          # mix conditions so you label blind

done = {}
if os.path.exists(OUT_PATH):
    for l in open(OUT_PATH):
        r = json.loads(l)
        done[r["item_id"]] = r["label"]


def clean(text):
    text = re.sub(r"\[(PAR|TLE|SEP|DOC)\]", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n\s*\n+", "\n\n", text).strip()


print("Label: 1 = sufficient, 0 = insufficient, s = skip for now, q = quit (progress is saved)\n")
with open(OUT_PATH, "a") as f:
    for n, it in enumerate(picked, 1):
        if it["item_id"] in done:
            continue
        print("=" * 90)
        print(f"ITEM {n}/{len(picked)}")      # condition and item_id hidden on purpose
        print(f"\nQUESTION: {it['question']}\n")
        for para in clean(it["context"]).split("\n\n"):
            print(textwrap.fill(para, width=90))
            print()
        while True:
            ans = input("Sufficient? [1/0/s/q]: ").strip().lower()
            if ans in {"1", "0", "s", "q"}:
                break
        if ans == "q":
            break
        if ans == "s":
            continue
        f.write(json.dumps({"item_id": it["item_id"], "label": int(ans)}) + "\n")
        f.flush()

n_done = sum(1 for _ in open(OUT_PATH))
print(f"\nLabelled {n_done}/{len(picked)}. Re-run to continue.")