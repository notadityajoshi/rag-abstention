import argparse
import json
import os
import re
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
from openai import OpenAI

USE_OPENROUTER = True                      # False = direct OpenAI API
MODEL = "openai/gpt-4o-mini" if USE_OPENROUTER else "gpt-4o-mini"
IN_PATH = "data/processed/sample.jsonl"
OUT_PATH = "data/processed/signals.jsonl"

# ---------------- prompts ----------------
# 1. Generator: Ming et al. (2025) default prompt WITHOUT the "unknown"/"conflict" add-ons
ANSWER_PROMPT = """You are an expert in retrieval-based question answering. Please respond with the exact answer, using only the information provided in the context.

Context:
{context}

Question: {question}
Answer:"""

# 2. Confidence: P(Correct), two best guesses with probabilities (Joren et al., 2025, Sec. 5.1)
PCORRECT_PROMPT = """Using only the context below, provide your two most likely answers to the question and the probability (between 0.0 and 1.0) that each is correct.

Context:
{context}

Question: {question}

Respond in JSON only: {{"guess1": "...", "p1": 0.0, "guess2": "...", "p2": 0.0}}"""

# 3. Sufficiency autorater: Joren et al. (2025), Appendix C.1 (1-shot), timestamp line removed
SUFF_PROMPT = """You are an expert LLM evaluator that excels at evaluating a QUESTION and REFERENCES.
Consider the following criteria:
Sufficient Context: 1 IF the CONTEXT is sufficient to infer the answer to the question and 0 IF the CONTEXT cannot be used to infer the answer to the question
First, output a list of step-by-step questions that would be used to arrive at a label for the criteria. Make sure to include questions about assumptions implicit in the QUESTION. Include questions about any mathematical calculations or arithmetic that would be required.
Next, answer each of the questions. Make sure to work step by step through any required mathematical calculations or arithmetic. Finally, use these answers to evaluate the criteria.
Output the ### EXPLANATION (Text). Then, use the EXPLANATION to output the ### EVALUATION (JSON)
EXAMPLE:
### QUESTION
In which year did the publisher of Roald Dahl's Guide to Railway Safety cease to exist?
### References
Roald Dahl's Guide to Railway Safety was published in 1991 by the British Railways Board. The British Railways Board had asked Roald Dahl to write the text of the booklet, and Quentin Blake to illustrate it, to help young people enjoy using the railways safely. The British Railways Board (BRB) was a nationalised industry in the United Kingdom that operated from 1963 to 2001. Until 1997 it was responsible for most railway services in Great Britain, trading under the brand name British Railways and, from 1965, British Rail. It did not operate railways in Northern Ireland, where railways were the responsibility of the Government of Northern Ireland.
### EXPLANATION
The context mentions that Roald Dahl's Guide to Railway Safety was published by the British Railways Board. It also states that the British Railways Board operated from 1963 to 2001, meaning the year it ceased to exist was 2001. Therefore, the context does provide a precise answer to the question.
### JSON
{{"Sufficient Context": 1}}
Remember the instructions: You are an expert LLM evaluator that excels at evaluating a QUESTION and REFERENCES. Consider the following criteria:
Sufficient Context: 1 IF the CONTEXT is sufficient to infer the answer to the question and 0 IF the CONTEXT cannot be used to infer the answer to the question
First, output a list of step-by-step questions that would be used to arrive at a label for the criteria. Make sure to include questions about assumptions implicit in the QUESTION. Include questions about any mathematical calculations or arithmetic that would be required.
Next, answer each of the questions. Make sure to work step by step through any required mathematical calculations or arithmetic. Finally, use these answers to evaluate the criteria.
Output the ### EXPLANATION (Text). Then, use the EXPLANATION to output the ### EVALUATION (JSON)
### QUESTION
{question}
### REFERENCES
{context}"""

# 4a. Judge for answerable controls: Joren et al. (2025), Appendix C.3 (LLMEval)
JUDGE_CONTROL_PROMPT = """===Task===
I need your help in evaluating an answer provided by an LLM against ground truth answers. Your task is to determine if the LLM's response matches the ground truth answers. Please analyze the provided data and make a decision.
===Instructions===
1. Carefully compare the "Predicted Answer" with the "Ground Truth Answers".
2. Consider the substance of the answers - look for equivalent information or correct answers. Do not focus on exact wording unless the exact wording is crucial to the meaning.
3. Your final decision should be based on whether the meaning and the vital facts of the "Ground Truth Answers" are present in the "Predicted Answer."
4. Categorize the answer as one of the following:
- "perfect": The answer is completely correct and matches the ground truth.
- "acceptable": The answer is partially correct or contains the main idea of the ground truth.
- "incorrect": The answer is wrong or contradicts the ground truth.
- "missing": The answer is "I don't know", "invalid question", or similar responses indicating lack of knowledge.
===Input Data===
- Question: {question}
- Predicted Answer: {answer}
- Ground Truth Answers: {gold}
===Output Format===
Provide your evaluation in the following format:
Explanation: (How you made the decision)
Decision: (One of "perfect", "acceptable", "incorrect", or "missing")
Please proceed with the evaluation."""

# 4b. Judge for inconsistent / unanswerable: did the model commit to one answer?
JUDGE_COMMIT_PROMPT = """You will see a question and a model's response. Decide whether the response COMMITS to a specific answer or ABSTAINS.

- COMMIT: the response states a specific answer to the question, even if it adds hedging.
- ABSTAIN: the response says it cannot answer, the information is not available or unknown, or it points out that the sources conflict and does not choose one answer. Listing several conflicting answers without choosing one is ABSTAIN.

Question: {question}
Response: {answer}

Respond in JSON only: {{"decision": "commit"}} or {{"decision": "abstain"}}"""


# ---------------- helpers ----------------
def extract_last_json(text):
    """Return the last {...} object in a text, or None."""
    for m in reversed(list(re.finditer(r"\{[^{}]*\}", text or ""))):
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            continue
    return None


def parse_suff(text):
    obj = extract_last_json(text)
    if obj is None:
        return None
    for k, v in obj.items():
        if "sufficient" in k.lower():
            try:
                v = int(v)
            except (TypeError, ValueError):
                return None
            return v if v in (0, 1) else None
    return None


def parse_pcorrect(text):
    obj = extract_last_json(text)
    if obj is None:
        return None
    try:
        p1 = float(obj.get("p1"))
    except (TypeError, ValueError):
        return None
    if not 0.0 <= p1 <= 1.0:
        return None
    return {"confidence": p1, "guess1": obj.get("guess1"),
            "guess2": obj.get("guess2"), "p2": obj.get("p2")}


def parse_control_decision(text):
    m = re.findall(r"Decision:\s*\"?\s*(perfect|acceptable|incorrect|missing)", text or "", re.I)
    return m[-1].lower() if m else None


def parse_commit(text):
    obj = extract_last_json(text)
    if obj is None:
        return None
    d = str(obj.get("decision", "")).lower()
    return d if d in ("commit", "abstain") else None


# ---------------- API ----------------
def make_client():
    load_dotenv()
    if USE_OPENROUTER:
        key = os.getenv("OPENROUTER_API_KEY")
        if not key:
            raise SystemExit("OPENROUTER_API_KEY missing: put it in .env")
        return OpenAI(api_key=key, base_url="https://openrouter.ai/api/v1",
                      max_retries=6, timeout=120)
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY missing: put it in .env")
    return OpenAI(max_retries=6, timeout=120)


def call(client, prompt, json_mode=False, max_tokens=800):
    kwargs = {"model": MODEL, "temperature": 0, "max_tokens": max_tokens,
              "messages": [{"role": "user", "content": prompt}]}
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    r = client.chat.completions.create(**kwargs)
    return r.choices[0].message.content or ""


def call_parsed(client, prompt, parser, json_mode=False, max_tokens=800, tries=2):
    """Call, parse; on parse failure retry once. Returns (parsed_or_None, raw_text)."""
    raw = ""
    for _ in range(tries):
        raw = call(client, prompt, json_mode, max_tokens)
        parsed = parser(raw)
        if parsed is not None:
            return parsed, raw
    return None, raw


def process(client, it):
    q, ctx = it["question"], it["context"]

    answer = call(client, ANSWER_PROMPT.format(context=ctx, question=q), max_tokens=200).strip()

    pc, pc_raw = call_parsed(client, PCORRECT_PROMPT.format(context=ctx, question=q),
                             parse_pcorrect, json_mode=True, max_tokens=200)

    suff, suff_raw = call_parsed(client, SUFF_PROMPT.format(context=ctx, question=q),
                                 parse_suff, max_tokens=1200)

    if it["condition"] == "control":
        dec, judge_raw = call_parsed(client, JUDGE_CONTROL_PROMPT.format(
            question=q, answer=answer, gold=it["gold_answers"]), parse_control_decision, max_tokens=300)
        correct = None if dec is None else int(dec in ("perfect", "acceptable"))
        committed = None if dec is None else int(dec != "missing")
    else:
        dec, judge_raw = call_parsed(client, JUDGE_COMMIT_PROMPT.format(question=q, answer=answer),
                                     parse_commit, json_mode=True, max_tokens=50)
        committed = None if dec is None else int(dec == "commit")
        correct = None if dec is None else int(dec == "abstain")   # committing = wrong (proposal rule)

    return {
        "item_id": it["item_id"], "condition": it["condition"], "subset": it["subset"],
        "answer": answer,
        "confidence": None if pc is None else pc["confidence"],
        "guess1": None if pc is None else pc["guess1"],
        "guess2": None if pc is None else pc["guess2"],
        "p2": None if pc is None else pc["p2"],
        "sufficiency": suff,
        "judge_decision": dec, "committed": committed, "correct": correct,
        "raw": {"pcorrect": pc_raw, "sufficiency": suff_raw, "judge": judge_raw},
        "model": MODEL,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true", help="10 items per condition")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

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
        for l in open(OUT_PATH):
            r = json.loads(l)
            if r["correct"] is not None and r["confidence"] is not None and r["sufficiency"] is not None:
                done.add(r["item_id"])          # incomplete rows are redone
    todo = [it for it in items if it["item_id"] not in done]
    print(f"{len(items)} selected, {len(done)} complete, {len(todo)} to run")

    client = make_client()
    lock = threading.Lock()
    start, n = time.time(), 0
    with open(OUT_PATH, "a") as f, ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process, client, it): it["item_id"] for it in todo}
        for fut in as_completed(futures):
            try:
                rec = fut.result()
            except Exception as e:                 # network/API failure: skip, re-run picks it up
                print(f"  FAILED {futures[fut]}: {type(e).__name__}: {e}")
                continue
            with lock:
                f.write(json.dumps(rec) + "\n")
                f.flush()
            n += 1
            if n % 25 == 0 or n == len(todo):
                el = time.time() - start
                print(f"  {n}/{len(todo)} | {el/60:.1f} min | ~{el/n*(len(todo)-n)/60:.1f} min left")

    # keep only the latest row per item (re-runs append)
    latest = {}
    for l in open(OUT_PATH):
        r = json.loads(l)
        latest[r["item_id"]] = r
    with open(OUT_PATH, "w") as f:
        for r in latest.values():
            f.write(json.dumps(r) + "\n")

    rows = list(latest.values())
    print("\nParse failures:",
          {k: sum(r[k] is None for r in rows) for k in ("confidence", "sufficiency", "correct")})
    print("Accuracy by condition:")
    for c in ("control", "inconsistent", "unanswerable"):
        v = [r["correct"] for r in rows if r["condition"] == c and r["correct"] is not None]
        if v:
            print(f"  {c:13s} n={len(v):4d}  correct={sum(v)/len(v):.2f}")
    print("Sufficiency=1 rate by condition:",
          {c: round(sum(r["sufficiency"] == 1 for r in rows if r["condition"] == c) /
                    max(1, sum(r["condition"] == c for r in rows)), 2)
           for c in ("control", "inconsistent", "unanswerable")})
    print("Judge decisions:", dict(Counter(r["judge_decision"] for r in rows)))


if __name__ == "__main__":
    main()