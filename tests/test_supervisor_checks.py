"""Unit tests for the supervisor-requested checks (no model or API needed)."""
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import confidence_given as CG  # noqa: E402
import nli_signal as N  # noqa: E402


def split(text):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def fake_nli(premises, hypotheses):
    """Contradiction only when one sentence says Sky and the other says Hulu."""
    out = []
    for p, h in zip(premises, hypotheses):
        clash = ("Sky" in p and "Hulu" in h) or ("Hulu" in p and "Sky" in h)
        out.append(0.95 if clash else 0.02)
    return out


ORIG = ("Setanta was awarded two of the six packages. Sky picked up the remaining four "
        "packages for 1.3bn. The league was founded long ago in England.")
EDIT = ORIG.replace("Sky", "Hulu")
QUESTION = "Who purchased the remaining four packages?"


def item(condition, passages, gold):
    return {"item_id": "x", "condition": condition, "subset": "SQuAD", "question": QUESTION,
            "passages": passages, "gold_answers": gold}


def test_cross_passage_conflict_is_detected_by_both_scores():
    r = N.score_item(item("inconsistent", [ORIG, EDIT], ["Sky", "Hulu"]), split, fake_nli)
    assert r["nli_sym_t30"] > 0.5 and r["nli_cross_t30"] > 0.5
    assert r["primary_from_cross_pair"] is True
    assert r["diag_stage"] == "answer_pair_detected"


def test_same_passage_conflict_is_ignored_by_cross_score():
    one = "Sky picked up the remaining four packages for 1.3bn. Hulu picked up the remaining four packages for 1.3bn."
    r = N.score_item(item("unanswerable", [one], ["unknown"]), split, fake_nli)
    assert r["nli_sym_t30"] > 0.5          # v2 score fires on a within-passage pair
    assert r["nli_cross_t30"] == 0.0       # cross-passage score cannot
    assert r["n_cross_pairs"] == 0


def test_selection_failure_is_attributed():
    filler = " ".join(f"The remaining four packages were purchased in round {i} by bidders." for i in range(12))
    r = N.score_item(item("inconsistent", [filler + " Sky was a broadcaster then.",
                                           filler + " Hulu was a broadcaster then."],
                          ["Sky", "Hulu"]), split, fake_nli)
    assert r["diag_stage"] in {"selection_failure", "alignment_failure", "nli_failure", "answer_pair_detected"}
    assert r["n_sentences"] <= N.TOP_K


def test_same_answer_matching():
    assert CG.same_answer("Sky", "sky.")
    assert CG.same_answer("The answer is Sky", "Sky")
    assert not CG.same_answer("Sky", "Hulu")
    assert not CG.same_answer("BSkyB", "Sky")      # whole words only
    assert not CG.same_answer("Sky", None)


def test_given_confidence_parser():
    assert CG.parse_given('{"p_correct": 0.7}') == 0.7
    assert CG.parse_given('{"p_correct": 1.4}') is None
    assert CG.parse_given("no json") is None
    CG.GIVEN_PROMPT.format(context="c", question="q", answer="a")