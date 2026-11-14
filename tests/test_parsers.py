"""Unit tests: parsing of LLM outputs into signals and labels."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import llm_signals as L  # noqa: E402


def test_sufficiency_parses_last_json():
    text = 'reasoning... {"Sufficient Context": 1} more reasoning ### JSON {"Sufficient Context": 0}'
    assert L.parse_suff(text) == 0


def test_sufficiency_accepts_string_digit():
    assert L.parse_suff('{"Sufficient Context": "1"}') == 1


def test_sufficiency_rejects_invalid():
    assert L.parse_suff("no json here") is None
    assert L.parse_suff('{"Sufficient Context": 5}') is None


def test_pcorrect_valid():
    out = L.parse_pcorrect('{"guess1": "Sky", "p1": 0.8, "guess2": "Hulu", "p2": 0.15}')
    assert out is not None
    assert out["confidence"] == 0.8 and out["guess2"] == "Hulu"


def test_pcorrect_out_of_range_rejected():
    assert L.parse_pcorrect('{"guess1": "Sky", "p1": 1.5}') is None
    assert L.parse_pcorrect('{"guess1": "Sky", "p1": "high"}') is None


def test_control_judge_takes_last_decision():
    text = 'Explanation: not "incorrect" at all.\nDecision: "Perfect"'
    assert L.parse_control_decision(text) == "perfect"


def test_commit_judge():
    assert L.parse_commit('{"decision": "abstain"}') == "abstain"
    assert L.parse_commit('{"decision": "maybe"}') is None


def test_prompts_format_without_error():
    L.ANSWER_PROMPT.format(context="c", question="q")
    L.PCORRECT_PROMPT.format(context="c", question="q")
    L.SUFF_PROMPT.format(context="c", question="q")
    L.JUDGE_CONTROL_PROMPT.format(question="q", answer="a", gold=["g"])
    L.JUDGE_COMMIT_PROMPT.format(question="q", answer="a")


def test_generator_prompt_has_no_task_instructions():
    """The generator must NOT be told about 'unknown'/'conflict' (Ming et al. sycophancy)."""
    p = L.ANSWER_PROMPT.lower()
    assert "unknown" not in p and "conflict" not in p