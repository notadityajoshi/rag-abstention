import argparse
import json
import os
import re
import time
from collections import defaultdict

import nltk
import torch
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL = "cross-encoder/nli-deberta-v3-base"
TOP_K = 10
THRESHOLDS = [0.2, 0.3, 0.4, 0.5]   # TF-IDF similarity needed to count as "same claim"
IN_PATH = "data/processed/sample.jsonl"
OUT_PATH = "data/processed/nli_scores.jsonl"

parser = argparse.ArgumentParser()
parser.add_argument("--pilot", action="store_true", help="10 items per condition only")
parser.add_argument("--device", default=None, help="force 'cpu' if mps errors")
args = parser.parse_args()

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

device = args.device or ("mps" if torch.backends.mps.is_available() else "cpu")
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL).to(device).eval()
CONTRA = next(i for i, l in model.config.id2label.items() if l.lower() == "contradiction")
print(f"Device: {device} | labels: {model.config.id2label} | contradiction index: {CONTRA}")

def clean(text):
    text = re.sub(r"\[(PAR|TLE|SEP|DOC)\]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def get_sentences(passages):
    sents = []
    for p in passages:
        for s in sent_tokenize(clean(p)):
            if len(s.split()) >= 4:
                sents.append(s.strip())
    return sents

def top_k_relevant(question, sents, k=TOP_K):
    if len(sents) <= k:
        return sents
    try:
        vec = TfidfVectorizer(stop_words="english").fit(sents + [question])
        sims = cosine_similarity(vec.transform([question]), vec.transform(sents))[0]
        idx = sims.argsort()[::-1][:k]
    except ValueError:  # empty vocabulary after stop-word removal
        idx = range(k)
    return [sents[i] for i in sorted(idx)]

def sentence_sims(sents):
    """Pairwise TF-IDF cosine similarity between sentences."""
    try:
        m = TfidfVectorizer(stop_words="english").fit_transform(sents)
        return cosine_similarity(m)
    except ValueError:
        return [[0.0] * len(sents) for _ in sents]
@torch.no_grad()
def contradiction_probs(premises, hypotheses, batch=16):
    out = []
    for i in range(0, len(premises), batch):
        enc = tok(premises[i:i + batch], hypotheses[i:i + batch], padding=True,
                  truncation=True, max_length=512, return_tensors="pt").to(device)
        probs = torch.softmax(model(**enc).logits, dim=-1)[:, CONTRA]
        out.extend(probs.cpu().tolist())
    return out

items = [json.loads(l) for l in open(IN_PATH)]
if args.pilot:
    picked, per = [], defaultdict(int)
    for it in items:
        if per[it["condition"]] < 10:
            picked.append(it)
            per[it["condition"]] += 1
    items = picked

done = set()
if os.path.exists(OUT_PATH):
    done = {json.loads(l)["item_id"] for l in open(OUT_PATH)}
todo = [it for it in items if it["item_id"] not in done]
print(f"{len(items)} items selected, {len(done)} already scored, {len(todo)} to run")

start = time.time()
with open(OUT_PATH, "a") as f:
    for n, it in enumerate(todo, 1):
        sents = top_k_relevant(it["question"], get_sentences(it["passages"]))
        ns = len(sents)
        sims = sentence_sims(sents) if ns > 1 else [[0.0]]

        # aligned pairs only: sentences that talk about the same claim
        cand = [(i, j) for i in range(ns) for j in range(i + 1, ns)
                if sims[i][j] >= min(THRESHOLDS)]

        sym = []
        if cand:
            k = len(cand)
            prem = [sents[i] for i, j in cand] + [sents[j] for i, j in cand]
            hyp = [sents[j] for i, j in cand] + [sents[i] for i, j in cand]
            probs = contradiction_probs(prem, hyp)
            sym = [min(probs[x], probs[x + k]) for x in range(k)]   # both directions must agree

        rec = {"item_id": it["item_id"], "condition": it["condition"], "subset": it["subset"],
               "n_sentences": ns, "n_aligned_pairs": len(cand)}
        for t in THRESHOLDS:
            vals = [s for s, (i, j) in zip(sym, cand) if sims[i][j] >= t]
            rec[f"nli_sym_t{int(t * 100)}"] = max(vals) if vals else 0.0
        if sym:
            b = max(range(len(sym)), key=sym.__getitem__)
            i, j = cand[b]
            rec["top_pair"] = [sents[i], sents[j]]
            rec["top_pair_sim"] = float(sims[i][j])
        else:
            rec["top_pair"], rec["top_pair_sim"] = [], 0.0
                # DIAGNOSTIC ONLY - uses gold answers, must never be a model feature (leakage)
        if it["condition"] == "inconsistent":
            a_orig, a_new = it["gold_answers"][0].lower(), it["gold_answers"][1].lower()
            rec["diag_orig_in_selected"] = any(a_orig in s.lower() for s in sents)
            rec["diag_new_in_selected"] = any(a_new in s.lower() for s in sents)
        f.write(json.dumps(rec) + "\n")
        f.flush()
        if n % 25 == 0 or n == len(todo):
            el = time.time() - start
            print(f"  {n}/{len(todo)} | {el/60:.1f} min | ~{el/n*(len(todo)-n)/60:.1f} min left")

# summary per threshold
rows = [json.loads(l) for l in open(OUT_PATH)]
for t in THRESHOLDS:
    key = f"nli_sym_t{int(t * 100)}"
    print(f"\nThreshold {t}:")
    by_cond = defaultdict(list)
    for r in rows:
        by_cond[r["condition"]].append(r[key])
    for c, v in sorted(by_cond.items()):
        print(f"  {c:13s} n={len(v):4d}  mean={sum(v)/len(v):.3f}  share>0.5={sum(x>0.5 for x in v)/len(v):.2f}")