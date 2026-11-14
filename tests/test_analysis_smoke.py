"""Smoke test: the full analysis runs end to end on simulated labels and writes valid outputs."""
import os
import subprocess
import sys

import pandas as pd
import pytest

ROOT = os.path.join(os.path.dirname(__file__), "..")


def test_analysis_simulated_runs():
    if not os.path.exists(os.path.join(ROOT, "data/processed/nli_scores.jsonl")):
        pytest.skip("NLI scores not generated yet")
    res = subprocess.run([sys.executable, "src/analysis.py", "--simulate"], cwd=ROOT,
                         capture_output=True, text=True, timeout=600)
    assert res.returncode == 0, res.stderr[-2000:]
    m = pd.read_csv(os.path.join(ROOT, "results/simulated/metrics.csv"))
    assert set(m["config"]) >= {"C1_conf", "C2_conf_suff", "C3_conf_nli", "C4_conf_suff_nli"}
    assert m["auroc"].between(0, 1).all() and m["aurc"].between(0, 1).all()
    for f in ("rq1_tests.txt", "per_condition.csv", "per_subset.csv", "risk_coverage.png"):
        assert os.path.exists(os.path.join(ROOT, "results/simulated", f))