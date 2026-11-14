"""Supervisor check 1: does the confidence score refer to the answer that was generated?

The frozen pipeline elicits P(Correct) in a separate call in which the model names its own
two best guesses. This script
  (a) measures how often that call's first guess is the same answer as the generated one, and
  (b) elicits a second confidence score that is explicitly about the generated answer.
Output: data/processed/confidence_given.jsonl (the frozen signals.jsonl is left untouched).
"""
import argparse
import json
import os
import re
import string
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from llm_signals import MODEL, call_parsed, extract_last_json, make_client

SAMPLE = "data/processed/sample.jsonl"
SIGNALS = "data/processed/signals.jsonl"
OUT_PATH = "data/processed/confidence_given.jsonl"

GIVEN_PROMPT = """Using only the context below, estimate the probability (between 0.0 and 1.0) that the proposed answer to the question is correct.

Context:
{context}

Question: {question}
Proposed answer: {answer}

Respond in JSON only: {{"p_correct": 0.0}}"""


def normalise(text):
    text = str(text).lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def same_answer(answer, guess):
    """Approximate string test: one normalised answer contains the other as whole words."""
    a, g = normalise(answer), normalise(guess)
    if not a or not g:
        return False
    return (re.search(r"\b" + re.escape(g) + r"\b", a) is not None
            or re.search(r"\b" + re.escape(a) + r"\b", g) is not None)


def parse_given(text):
    obj = extract_last_json(text)
    if obj is None:
        return None
    try:
        p = float(obj.get("p_correct"))
    except (TypeError, ValueError):
        return None
    return p if 0.0 <= p <= 1.0 else None


def process(client, item, sig):
    p, raw = call_parsed(client, GIVEN_PROMPT.format(
        context=item["context"], question=item["question"], answer=sig["answer"]),
        parse_given, json_mode=True, max_tokens=50)
    return {"item_id": sig["item_id"], "condition": sig["condition"],
            "confidence_given": p, "confidence_pcorrect": sig["confidence"],
            "same_answer": same_answer(sig["answer"], sig.get("guess1")),
            "answer": sig["answer"], "guess1": sig.get("guess1"),
            "committed": sig.get("committed"), "raw": raw, "model": MODEL}


def report(rows):
    print("\nDoes the P(Correct) call's first guess match the generated answer? (string match)")
    by = defaultdict(list)
    for r in rows:
        by[r["condition"]].append(r["same_answer"])
    for c, v in sorted(by.items()):
        print(f"  {c:13s} n={len(v):4d}  same answer={sum(v)/len(v):.3f}")
    allv = [r["same_answer"] for r in rows]
    print(f"  {'all':13s} n={len(allv):4d}  same answer={sum(allv)/len(allv):.3f}")

    ok = [r for r in rows if r["confidence_given"] is not None and r["confidence_pcorrect"] is not None]
    if len(ok) > 2:
        import numpy as np
        a = np.array([r["confidence_pcorrect"] for r in ok])
        b = np.array([r["confidence_given"] for r in ok])
        print(f"\nPearson correlation, P(Correct) vs answer-specific confidence: {np.corrcoef(a, b)[0, 1]:.3f}")
        for flag in (True, False):
            sub = [r for r in ok if r["same_answer"] == flag]
            if sub:
                d = np.mean([abs(r["confidence_pcorrect"] - r["confidence_given"]) for r in sub])
                print(f"  mean |difference| when same_answer={flag}: {d:.3f} (n={len(sub)})")
        print("Mean answer-specific confidence by condition:")
        byc = defaultdict(list)
        for r in ok:
            byc[r["condition"]].append(r["confidence_given"])
        for c, v in sorted(byc.items()):
            print(f"  {c:13s} {sum(v)/len(v):.3f}")
    print("Parse failures:", sum(r["confidence_given"] is None for r in rows))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true", help="first 30 items only")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    sample = {}
    for l in open(SAMPLE):
        r = json.loads(l)
        sample[r["item_id"]] = r
    sigs = [json.loads(l) for l in open(SIGNALS)]
    if args.pilot:
        sigs = sigs[:30]

    done = set()
    if os.path.exists(OUT_PATH):
        for l in open(OUT_PATH):
            r = json.loads(l)
            if r["confidence_given"] is not None:
                done.add(r["item_id"])
    todo = [s for s in sigs if s["item_id"] not in done]
    print(f"{len(sigs)} selected, {len(done)} complete, {len(todo)} to run")

    if todo:
        client = make_client()
        lock = threading.Lock()
        start, n = time.time(), 0
        with open(OUT_PATH, "a") as f, ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(process, client, sample[s["item_id"]], s): s["item_id"] for s in todo}
            for fut in as_completed(futures):
                try:
                    rec = fut.result()
                except Exception as e:
                    print(f"  FAILED {futures[fut]}: {type(e).__name__}: {e}")
                    continue
                with lock:
                    f.write(json.dumps(rec) + "\n")
                    f.flush()
                n += 1
                if n % 100 == 0 or n == len(todo):
                    el = time.time() - start
                    print(f"  {n}/{len(todo)} | {el/60:.1f} min | ~{el/n*(len(todo)-n)/60:.1f} min left")

    latest = {}
    for l in open(OUT_PATH):
        r = json.loads(l)
        latest[r["item_id"]] = r
    with open(OUT_PATH, "w") as f:
        for r in latest.values():
            f.write(json.dumps(r) + "\n")
    report(list(latest.values()))


if __name__ == "__main__":
    main()