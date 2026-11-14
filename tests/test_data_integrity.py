"""Integrity tests on the generated experimental data (run after the pipeline)."""
import json
import os
import re
import string
from collections import Counter

import pytest

ROOT = os.path.join(os.path.dirname(__file__), "..")
SAMPLE = os.path.join(ROOT, "data/processed/sample.jsonl")
NLI = os.path.join(ROOT, "data/processed/nli_scores.jsonl")
SIGNALS = os.path.join(ROOT, "data/processed/signals.jsonl")


def load(path):
    if not os.path.exists(path):
        pytest.skip(f"{path} not generated yet")
    with open(path) as f:
        return [json.loads(l) for l in f]


def normalise(text):
    text = text.lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def test_condition_balance():
    counts = Counter(r["condition"] for r in load(SAMPLE))
    assert counts == {"inconsistent": 500, "control": 500, "unanswerable": 500}


def test_no_duplicate_questions_or_ids():
    rows = load(SAMPLE)
    assert len({r["item_id"] for r in rows}) == len(rows)
    assert len({r["source_qid"] for r in rows}) == len(rows)
    assert len({normalise(r["question"]) for r in rows}) == len(rows)


def test_passage_structure():
    for r in load(SAMPLE):
        assert all(p and p.strip() for p in r["passages"])
        expected = 1 if r["condition"] == "unanswerable" else 2
        assert len(r["passages"]) == expected, r["item_id"]


def test_control_distractor_has_no_gold_answer():
    for r in load(SAMPLE):
        if r["condition"] == "control":
            gold = normalise(r["gold_answers"][0])
            assert not re.search(r"\b" + re.escape(gold) + r"\b", normalise(r["passages"][1])), r["item_id"]


def test_nli_covers_sample_and_is_probability():
    ids = {r["item_id"] for r in load(SAMPLE)}
    rows = load(NLI)
    assert {r["item_id"] for r in rows} == ids
    for r in rows:
        for k in ("nli_sym_t20", "nli_sym_t30", "nli_sym_t40", "nli_sym_t50"):
            assert 0.0 <= r[k] <= 1.0


def test_signals_complete():
    ids = {r["item_id"] for r in load(SAMPLE)}
    rows = load(SIGNALS)
    assert {r["item_id"] for r in rows} == ids
    for r in rows:
        assert r["correct"] in (0, 1)
        assert r["sufficiency"] in (0, 1)
        assert 0.0 <= r["confidence"] <= 1.0
        