"""Descriptive figures and tables for the report. Reads frozen outputs; changes no results."""
import glob
import json
import os
import re
import string
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = "results/figures"
os.makedirs(OUT, exist_ok=True)
CONDS = ["control", "inconsistent", "unanswerable"]
MAIN = ["C1_conf", "C2_conf_suff", "C3_conf_nli", "C4_conf_suff_nli"]


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f]


def normalise(text):
    text = text.lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def contains_answer(passage, answer):
    p, a = normalise(passage), normalise(answer)
    return bool(a) and re.search(r"\b" + re.escape(a) + r"\b", p) is not None


sample = {r["item_id"]: r for r in load_jsonl("data/processed/sample.jsonl")}
nli = pd.DataFrame(load_jsonl("data/processed/nli_scores.jsonl"))
sig = pd.DataFrame(load_jsonl("data/processed/signals.jsonl"))

# ---------- Fig 1: NLI contradiction score by condition ----------
plt.figure(figsize=(6, 4))
data = [nli.loc[nli.condition == c, "nli_sym_t30"].to_numpy() for c in CONDS]
plt.boxplot(data, showfliers=False)
plt.xticks(range(1, 4), CONDS)
for i, d in enumerate(data, 1):
    plt.scatter(np.random.default_rng(0).normal(i, 0.06, len(d)), d, s=3, alpha=0.25)
plt.ylabel("Max symmetric contradiction probability")
plt.title("NLI contradiction signal by condition (threshold 0.3)")
plt.tight_layout()
plt.savefig(f"{OUT}/fig1_nli_by_condition.png", dpi=200)
plt.close()

# ---------- Fig 2: confidence by condition and correctness ----------
plt.figure(figsize=(7, 4))
labels, boxes = [], []
for c in CONDS:
    for corr in (1, 0):
        v = sig.loc[(sig.condition == c) & (sig.correct == corr), "confidence"].dropna().to_numpy()
        boxes.append(v if len(v) else np.array([np.nan]))
        labels.append(f"{c[:5]}\n{'correct' if corr else 'wrong'}\n(n={len(v)})")
plt.boxplot(boxes, showfliers=False)
plt.xticks(range(1, len(labels) + 1), labels, fontsize=8)
plt.ylabel("Verbalised confidence P(Correct)")
plt.title("Confidence by condition and correctness")
plt.tight_layout()
plt.savefig(f"{OUT}/fig2_confidence.png", dpi=200)
plt.close()

# ---------- Fig 3: per-condition wrong-answer and abstention rate at 50% coverage ----------
per = pd.read_csv("results/final/per_condition.csv")
per = per[(per.coverage == 0.5) & (per.config.isin(MAIN))]
fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
w = 0.2
x = np.arange(len(CONDS))
for ax, metric, title in [(axes[0], "wrong_answer_rate", "Wrong-answer rate"),
                          (axes[1], "abstain_rate", "Abstention rate")]:
    for k, cfg in enumerate(MAIN):
        vals = [per[(per.config == cfg) & (per.condition == c)][metric].iloc[0] for c in CONDS]
        ax.bar(x + (k - 1.5) * w, vals, w, label=cfg)
    ax.set_xticks(x)
    ax.set_xticklabels(CONDS)
    ax.set_title(f"{title} at 50% coverage")
axes[0].set_ylabel("Share of items in condition")
axes[1].legend(fontsize=8)
plt.tight_layout()
plt.savefig(f"{OUT}/fig3_per_condition.png", dpi=200)
plt.close()

# ---------- Table: NLI threshold sensitivity ----------
rows = []
for path in ["results/final/metrics.csv"] + sorted(glob.glob("results/sensitivity/*/metrics.csv")):
    key = "nli_sym_t30 (primary)" if "final" in path else path.split("/")[-2]
    m = pd.read_csv(path).set_index("config")
    auroc = {cfg: float(v) for cfg, v in m["auroc"].items()}
    aurc = {cfg: float(v) for cfg, v in m["aurc"].items()}
    rows.append({"nli_key": key,
                 "AUROC_C2": auroc["C2_conf_suff"], "AUROC_C4": auroc["C4_conf_suff_nli"],
                 "AUROC_C4_minus_C2": auroc["C4_conf_suff_nli"] - auroc["C2_conf_suff"],
                 "AURC_C2": aurc["C2_conf_suff"], "AURC_C4": aurc["C4_conf_suff_nli"]})
sens = pd.DataFrame(rows)
sens.to_csv(f"{OUT}/table_sensitivity.csv", index=False)
print("=== NLI threshold sensitivity ===")
print(sens.round(3).to_string(index=False))

# ---------- Table: which answer the model picks on inconsistent items ----------
choice = Counter()
for r in sig[sig.condition == "inconsistent"].itertuples():
    orig, new = sample[r.item_id]["gold_answers"][:2]
    has_o, has_n = contains_answer(r.answer, orig), contains_answer(r.answer, new)
    if r.committed == 0:
        choice["abstained / flagged conflict"] += 1
    elif has_o and not has_n:
        choice["original passage answer"] += 1
    elif has_n and not has_o:
        choice["edited passage answer"] += 1
    elif has_o and has_n:
        choice["both answers mentioned"] += 1
    else:
        choice["neither (paraphrase / other)"] += 1
tab = pd.DataFrame(choice.most_common(), columns=["model behaviour", "n"])
tab["share"] = tab["n"] / tab["n"].sum()
tab.to_csv(f"{OUT}/table_inconsistent_choice.csv", index=False)
print("\n=== Inconsistent items: which answer did the model give? (string match, approximate) ===")
print(tab.round(3).to_string(index=False))

# ---------- Table: descriptive summary per condition ----------
desc = sig.merge(nli[["item_id", "nli_sym_t30"]], on="item_id").groupby("condition").agg(
    n=("item_id", "count"), correct=("correct", "mean"), committed=("committed", "mean"),
    sufficiency_1=("sufficiency", "mean"), confidence_mean=("confidence", "mean"),
    nli_flag_rate=("nli_sym_t30", lambda v: (v > 0.5).mean()))
desc.to_csv(f"{OUT}/table_descriptive.csv")
print("\n=== Descriptive summary by condition ===")
print(desc.round(3).to_string())
print(f"\nSaved figures and tables to {OUT}/")