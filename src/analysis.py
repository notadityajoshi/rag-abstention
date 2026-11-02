import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import binomtest, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

NLI_KEY = "nli_sym_t30"          # primary threshold, frozen on pilot
COVERAGES = [0.75, 0.50, 0.25]
N_BOOT = 2000
SEED = 42

CONFIGS = {
    "C1_conf":           ["confidence"],
    "C2_conf_suff":      ["confidence", "sufficiency"],
    "C3_conf_nli":       ["confidence", "contradiction"],
    "C4_conf_suff_nli":  ["confidence", "sufficiency", "contradiction"],
}

parser = argparse.ArgumentParser()
parser.add_argument("--simulate", action="store_true",
                    help="FAKE confidence/sufficiency/correct labels to test the code")
args = parser.parse_args()
OUT = "results/simulated" if args.simulate else "results/final"
os.makedirs(OUT, exist_ok=True)


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f]


# ---------- 1. assemble one table: one row per item ----------
sample = pd.DataFrame(load_jsonl("data/processed/sample.jsonl"))[["item_id", "condition", "subset"]]
nli = pd.DataFrame(load_jsonl("data/processed/nli_scores.jsonl"))[["item_id", NLI_KEY]]
df = sample.merge(nli, on="item_id", how="inner").rename(columns={NLI_KEY: "contradiction"})

if args.simulate:
    rng = np.random.default_rng(SEED)
    p_correct = df["condition"].map({"control": 0.85, "inconsistent": 0.10, "unanswerable": 0.25})
    df["correct"] = (rng.random(len(df)) < p_correct).astype(int)
    df["confidence"] = np.where(df["correct"] == 1, rng.beta(8, 2, len(df)), rng.beta(6, 3, len(df)))
    p_suff = df["condition"].map({"control": 0.90, "inconsistent": 0.50, "unanswerable": 0.20})
    df["sufficiency"] = (rng.random(len(df)) < p_suff).astype(int)
else:
    sig = pd.DataFrame(load_jsonl("data/processed/signals.jsonl"))[
        ["item_id", "correct", "confidence", "sufficiency"]]
    df = df.merge(sig, on="item_id", how="inner")

df = df.dropna(subset=["correct", "confidence", "sufficiency", "contradiction"]).reset_index(drop=True)
y = df["correct"].to_numpy()
print(f"Items: {len(df)} | base accuracy (answer everything): {y.mean():.3f}")
print("By condition:", df.groupby("condition")["correct"].mean().round(3).to_dict())

# ---------- 2. out-of-fold P(correct) for each configuration ----------
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
scores = {}
for name, feats in CONFIGS.items():
    clf = LogisticRegression(max_iter=1000)
    scores[name] = cross_val_predict(clf, df[feats], y, cv=cv, method="predict_proba")[:, 1]
    df[f"score_{name}"] = scores[name]


# ---------- 3. metrics ----------
def risk_coverage(score, y):
    order = np.argsort(-score)                    # most confident first
    correct_sorted = y[order]
    k = np.arange(1, len(y) + 1)
    sel_acc = np.cumsum(correct_sorted) / k
    return k / len(y), 1 - sel_acc               # coverage, risk


def aurc(score, y):
    return risk_coverage(score, y)[1].mean()


def answered_mask(score, coverage):
    """Answer the top `coverage` share of items by score, abstain on the rest."""
    n_ans = int(round(coverage * len(score)))
    mask = np.zeros(len(score), dtype=bool)
    mask[np.argsort(-score)[:n_ans]] = True
    return mask


def sel_acc_at(score, y, coverage):
    m = answered_mask(score, coverage)
    return y[m].mean()


rng = np.random.default_rng(SEED)
boot_idx = [rng.integers(0, len(y), len(y)) for _ in range(N_BOOT)]


def boot_ci(fn):
    vals = [fn(idx) for idx in boot_idx]
    return np.percentile(vals, [2.5, 97.5])


rows = []
for name in CONFIGS:
    s = scores[name]
    row = {"config": name,
           "auroc": roc_auc_score(y, s),
           "aurc": aurc(s, y)}
    lo, hi = boot_ci(lambda i: roc_auc_score(y[i], s[i]))
    row["auroc_ci"] = f"[{lo:.3f}, {hi:.3f}]"
    for c in COVERAGES:
        row[f"sel_acc@{int(c*100)}"] = sel_acc_at(s, y, c)
    rows.append(row)
metrics = pd.DataFrame(rows)
metrics.to_csv(f"{OUT}/metrics.csv", index=False)
print("\n=== Overall metrics (AUROC: predicting correctness; AURC: lower is better) ===")
print(metrics.round(3).to_string(index=False))

# ---------- 4. key comparison C2 vs C4 ----------
s2, s4 = scores["C2_conf_suff"], scores["C4_conf_suff_nli"]
lo, hi = boot_ci(lambda i: roc_auc_score(y[i], s4[i]) - roc_auc_score(y[i], s2[i]))
d_auroc = roc_auc_score(y, s4) - roc_auc_score(y, s2)
lo2, hi2 = boot_ci(lambda i: aurc(s2[i], y[i]) - aurc(s4[i], y[i]))
d_aurc = aurc(s2, y) - aurc(s4, y)

# likelihood-ratio test on nested logistic