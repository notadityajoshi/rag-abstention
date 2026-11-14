"""NLI contradiction signal (v3).

v2 (frozen results): top-10 question-relevant sentences from the whole context, aligned pairs
(TF-IDF similarity >= threshold), symmetric contradiction, max over pairs  -> nli_sym_tXX
v3 adds, in the same pass and without changing the v2 columns:
  * nli_cross_tXX : the same score restricted to pairs of sentences from DIFFERENT passages
  * a stage-by-stage diagnosis of every inconsistent item (selection / alignment / NLI)
Output: data/processed/nli_scores_v3.jsonl (the frozen nli_scores.jsonl is left untouched).
"""
import argparse
import json
import os
import re
import string
import time
from collections import Counter, defaultdict

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

MODEL = "cross-encoder/nli-deberta-v3-base"
TOP_K = 10
THRESHOLDS = [0.2, 0.3, 0.4, 0.5]   # TF-IDF similarity needed to count as "same claim"
PRIMARY_T = 0.3
IN_PATH = "data/processed/sample.jsonl"
OUT_PATH = "data/processed/nli_scores_v3.jsonl"


# ---------------- text helpers ----------------
def clean(text):
    text = re.sub(r"\[(PAR|TLE|SEP|DOC)\]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def normalise(text):
    text = text.lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def contains_answer(sentence, answer):
    s, a = normalise(sentence), normalise(answer)
    return bool(a) and re.search(r"\b" + re.escape(a) + r"\b", s) is not None


def get_sentences(passages, sent_tokenize):
    """Return (sentences, passage_index_of_each_sentence), in document order."""
    sents, pidx = [], []
    for p_i, p in enumerate(passages):
        for s in sent_tokenize(clean(p)):
            if len(s.split()) >= 4:
                sents.append(s.strip())
                pidx.append(p_i)
    return sents, pidx


def top_k_indices(question, sents, k=TOP_K):
    """Indices (in document order) of the k sentences most similar to the question."""
    if len(sents) <= k:
        return list(range(len(sents)))
    try:
        vec = TfidfVectorizer(stop_words="english").fit(sents + [question])
        sims = cosine_similarity(vec.transform([question]), vec.transform(sents))[0]
        idx = sims.argsort()[::-1][:k]
    except ValueError:  # empty vocabulary after stop-word removal
        idx = range(k)
    return sorted(int(i) for i in idx)


def sentence_sims(sents):
    """Pairwise TF-IDF cosine similarity between sentences."""
    try:
        m = TfidfVectorizer(stop_words="english").fit_transform(sents)
        return cosine_similarity(m)
    except ValueError:
        return [[0.0] * len(sents) for _ in sents]


# ---------------- scoring one item ----------------
def score_item(it, sent_tokenize, contradiction_probs):
    """Compute all NLI columns and diagnostics for one item.

    contradiction_probs(premises, hypotheses) -> list of P(contradiction); injected so the
    logic can be unit-tested without loading the model.
    """
    all_sents, all_pidx = get_sentences(it["passages"], sent_tokenize)
    keep = top_k_indices(it["question"], all_sents)
    sents = [all_sents[i] for i in keep]
    pidx = [all_pidx[i] for i in keep]
    ns = len(sents)
    sims = sentence_sims(sents) if ns > 1 else [[0.0]]

    cand = [(i, j) for i in range(ns) for j in range(i + 1, ns) if sims[i][j] >= min(THRESHOLDS)]
    sym = []
    if cand:
        k = len(cand)
        prem = [sents[i] for i, j in cand] + [sents[j] for i, j in cand]
        hyp = [sents[j] for i, j in cand] + [sents[i] for i, j in cand]
        probs = contradiction_probs(prem, hyp)
        sym = [min(probs[x], probs[x + k]) for x in range(k)]   # both directions must agree

    rec = {"item_id": it["item_id"], "condition": it["condition"], "subset": it["subset"],
           "n_passages": len(it["passages"]), "n_sentences": ns, "n_aligned_pairs": len(cand),
           "n_cross_pairs": sum(pidx[i] != pidx[j] for i, j in cand)}

    for t in THRESHOLDS:
        tag = int(t * 100)
        both = [s for s, (i, j) in zip(sym, cand) if sims[i][j] >= t]
        cross = [s for s, (i, j) in zip(sym, cand) if sims[i][j] >= t and pidx[i] != pidx[j]]
        rec[f"nli_sym_t{tag}"] = max(both) if both else 0.0        # v2 definition (unchanged)
        rec[f"nli_cross_t{tag}"] = max(cross) if cross else 0.0    # cross-passage pairs only

    # which kind of pair produced the v2 primary score?
    prim = [(s, pidx[i] != pidx[j]) for s, (i, j) in zip(sym, cand) if sims[i][j] >= PRIMARY_T]
    if prim:
        best = max(prim, key=lambda p: p[0])
        rec["primary_from_cross_pair"] = bool(best[1])
    else:
        rec["primary_from_cross_pair"] = None

    if sym:
        b = max(range(len(sym)), key=sym.__getitem__)
        i, j = cand[b]
        rec["top_pair"] = [sents[i], sents[j]]
        rec["top_pair_sim"] = float(sims[i][j])
        rec["top_pair_cross"] = bool(pidx[i] != pidx[j])
    else:
        rec["top_pair"], rec["top_pair_sim"], rec["top_pair_cross"] = [], 0.0, None

    # ---- DIAGNOSTIC ONLY (uses gold answers; never a model feature) ----
    if it["condition"] == "inconsistent" and len(it["gold_answers"]) >= 2:
        a_orig, a_new = it["gold_answers"][0], it["gold_answers"][1]
        loc_o = [n for n, s in enumerate(all_sents) if all_pidx[n] == 0 and contains_answer(s, a_orig)]
        loc_n = [n for n, s in enumerate(all_sents) if all_pidx[n] == 1 and contains_answer(s, a_new)]
        sel_o = [n for n, s in enumerate(sents) if pidx[n] == 0 and contains_answer(s, a_orig)]
        sel_n = [n for n, s in enumerate(sents) if pidx[n] == 1 and contains_answer(s, a_new)]
        ans_pairs = [(min(o, n), max(o, n)) for o in sel_o for n in sel_n]
        aligned = [p for p in ans_pairs if sims[p[0]][p[1]] >= PRIMARY_T]
        score_of = {c: s for c, s in zip(cand, sym)}
        ans_scores = [score_of[p] for p in aligned if p in score_of]
        best_ans = max(ans_scores) if ans_scores else None
        if not loc_o or not loc_n:
            stage = "answer_not_located"        # answer string not found verbatim; cannot attribute
        elif not sel_o or not sel_n:
            stage = "selection_failure"         # relevance filter dropped an answer sentence
        elif not aligned:
            stage = "alignment_failure"         # answer sentences selected but similarity < threshold
        elif best_ans is not None and best_ans > 0.5:
            stage = "answer_pair_detected"
        else:
            stage = "nli_failure"               # aligned answer pair scored <= 0.5
        rec["diag_stage"] = stage
        rec["diag_answer_pair_score"] = best_ans
        rec["diag_answer_pair_sim"] = max((float(sims[p[0]][p[1]]) for p in ans_pairs), default=None)
        if ans_pairs:
            p = max(ans_pairs, key=lambda q: sims[q[0]][q[1]])
            rec["diag_answer_pair"] = [sents[p[0]], sents[p[1]]]
    return rec


# ---------------- run ----------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", action="store_true", help="10 items per condition only")
    parser.add_argument("--device", default=None, help="force 'cpu' if mps errors")
    args = parser.parse_args()

    import nltk
    import torch
    from nltk.tokenize import sent_tokenize
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)

    device = args.device or ("mps" if torch.backends.mps.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL).to(device).eval()
    contra = next(i for i, l in model.config.id2label.items() if l.lower() == "contradiction")
    print(f"Device: {device} | labels: {model.config.id2label} | contradiction index: {contra}")

    @torch.no_grad()
    def contradiction_probs(premises, hypotheses, batch=16):
        out = []
        for i in range(0, len(premises), batch):
            enc = tok(premises[i:i + batch], hypotheses[i:i + batch], padding=True,
                      truncation=True, max_length=512, return_tensors="pt").to(device)
            probs = torch.softmax(model(**enc).logits, dim=-1)[:, contra]
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
            f.write(json.dumps(score_item(it, sent_tokenize, contradiction_probs)) + "\n")
            f.flush()
            if n % 25 == 0 or n == len(todo):
                el = time.time() - start
                print(f"  {n}/{len(todo)} | {el/60:.1f} min | ~{el/n*(len(todo)-n)/60:.1f} min left")

    rows = [json.loads(l) for l in open(OUT_PATH)]
    for key in ("nli_sym_t30", "nli_cross_t30"):
        print(f"\n{key}: share of items with score > 0.5")
        by = defaultdict(list)
        for r in rows:
            by[r["condition"]].append(r[key] > 0.5)
        for c, v in sorted(by.items()):
            print(f"  {c:13s} n={len(v):4d}  flagged={sum(v)/len(v):.3f}")

    src = Counter((r["condition"], r["primary_from_cross_pair"]) for r in rows if r["nli_sym_t30"] > 0.5)
    print("\nFlagged items (v2 primary): was the deciding pair cross-passage? (condition, cross) -> n")
    for k, v in sorted(src.items(), key=str):
        print(f"  {k}: {v}")

    inc = [r for r in rows if r["condition"] == "inconsistent" and "diag_stage" in r]
    print(f"\nInconsistent items by pipeline stage (n={len(inc)}):")
    for stage, n in Counter(r["diag_stage"] for r in inc).most_common():
        print(f"  {stage:22s} {n:4d}  ({n/len(inc):.1%})")
    missed = [r for r in inc if r["nli_cross_t30"] <= 0.5]
    print(f"\nMissed by the cross-passage signal: {len(missed)}; stage of each miss:")
    for stage, n in Counter(r["diag_stage"] for r in missed).most_common():
        print(f"  {stage:22s} {n:4d}")
    for stage in ("selection_failure", "alignment_failure", "nli_failure"):
        ex = [r for r in missed if r["diag_stage"] == stage and r.get("diag_answer_pair")][:2]
        for r in ex:
            print(f"\n  [{stage}] {r['subset']} sim={r['diag_answer_pair_sim']:.2f} "
                  f"score={r['diag_answer_pair_score']}")
            print("    A:", r["diag_answer_pair"][0][:170])
            print("    B:", r["diag_answer_pair"][1][:170])


if __name__ == "__main__":
    main()