import argparse
import json
import os
import string

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, cross_val_predict

PRIMARY_NLI_KEY = "nli_sym_t30"  # primary threshold, frozen on pilot
COVERAGES = [0.75, 0.50, 0.25]
N_BOOT = 2000
SEED = 42

CONFIGS = {
    "C1_conf":           ["confidence"],
    "C2_conf_suff":      ["confidence", "sufficiency"],
    "C3_conf_nli":       ["confidence", "contradiction"],
    "C4_conf_suff_nli":  ["confidence", "sufficiency", "contradiction"],
    # robustness: does anything still help once the model KNOWS the condition?
    "R0_cond_oracle":    ["cond_inc", "cond_una"],
    "R1_cond_conf_suff": ["cond_inc", "cond_una", "confidence", "sufficiency"],
    "R2_cond_conf_suff_nli": ["cond_inc", "cond_una", "confidence", "sufficiency", "contradiction"],
}
MAIN = ["C1_conf", "C2_conf_suff", "C3_conf_nli", "C4_conf_suff_nli"]

parser = argparse.ArgumentParser()
parser.add_argument("--simulate", action="store_true",
                    help="FAKE confidence/sufficiency/correct labels to test the code")
parser.add_argument("--nli-key", default=PRIMARY_NLI_KEY,
                    help="NLI threshold column for sensitivity analysis, e.g. nli_sym_t40")
parser.add_argument("--confidence", choices=["pcorrect", "given"], default="pcorrect",
                    help="pcorrect = frozen P(Correct) call; given = confidence in the generated answer")
parser.add_argument("--cv", choices=["stratified", "grouped"], default="stratified",
                    help="grouped = items that share a passage are kept in the same fold")
args = parser.parse_args()
NLI_KEY = args.nli_key
DEFAULTS = args.confidence == "pcorrect" and args.cv == "stratified"
if args.simulate:
    OUT = "results/simulated"
elif DEFAULTS and NLI_KEY == PRIMARY_NLI_KEY:
    OUT = "results/final"
elif DEFAULTS and NLI_KEY.startswith("nli_sym_"):
    OUT = f"results/sensitivity/{NLI_KEY}"
else:
    OUT = f"results/variants/conf-{args.confidence}__{NLI_KEY}__cv-{args.cv}"
print(f"confidence: {args.confidence} | NLI signal: {NLI_KEY} | CV: {args.cv} -> output {OUT}")
os.makedirs(OUT, exist_ok=True)


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f]


# ---------- 1. assemble one table: one row per item ----------
sample_rows = load_jsonl("data/processed/sample.jsonl")
sample = pd.DataFrame(sample_rows)[["item_id", "condition", "subset"]]
# frozen columns (nli_sym_*) come from the frozen file; new columns (nli_cross_*) from the v3 run
nli_all = pd.DataFrame(load_jsonl("data/processed/nli_scores.jsonl"))
if NLI_KEY not in nli_all.columns:
    nli_all = pd.DataFrame(load_jsonl("data/processed/nli_scores_v3.jsonl"))
nli = nli_all[["item_id", NLI_KEY]]
df = sample.merge(nli, on="item_id", how="inner").rename(columns={NLI_KEY: "contradiction"})

if args.simulate:
    rng = np.random.default_rng(SEED)
    p_correct = df["condition"].map({"control": 0.85, "inconsistent": 0.10, "unanswerable": 0.25})
    df["correct"] = (rng.random(len(df)) < p_correct).astype(int)
    df["confidence"] = np.where(df["correct"] == 1, rng.beta(8, 2, len(df)), rng.beta(6, 3, len(df)))
    p_suff = df["condition"].map({"control": 0.90, "inconsistent": 0.50, "unanswerable": 0.20})
    df["sufficiency"] = (rng.random(len(df)) < p_suff).astype(int)
    df["committed"] = np.where(df["condition"] == "control", 1, 1 - df["correct"])
else:
    sig = pd.DataFrame(load_jsonl("data/processed/signals.jsonl"))[
        ["item_id", "correct", "confidence", "sufficiency", "committed"]]
    if args.confidence == "given":
        given = pd.DataFrame(load_jsonl("data/processed/confidence_given.jsonl"))[
            ["item_id", "confidence_given"]]
        sig = sig.drop(columns="confidence").merge(given, on="item_id", how="inner").rename(
            columns={"confidence_given": "confidence"})
    df = df.merge(sig, on="item_id", how="inner")

df = df.dropna(subset=["correct", "confidence", "sufficiency", "contradiction"]).reset_index(drop=True)
df["cond_inc"] = (df["condition"] == "inconsistent").astype(int)
df["cond_una"] = (df["condition"] == "unanswerable").astype(int)
y = df["correct"].to_numpy().astype(int)
print(f"Items: {len(df)} | base accuracy (answer everything): {y.mean():.3f}")
print("By condition:", df.groupby("condition")["correct"].mean().round(3).to_dict())

committed = df["committed"].to_numpy().astype(int)


# ---------- 1b. groups of related items (share a passage, or a near-identical one) ----------
def _norm(text):
    text = str(text).lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    return " ".join(text.split())


def related_groups(rows):
    """Union-find: two items are related if any of their passages has the same normalised text
    or the same first 120 normalised characters (catches edited copies of one source passage)."""
    parent = {r["item_id"]: r["item_id"] for r in rows}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    seen = {}
    for r in rows:
        for p in r.get("passages") or []:
            n = _norm(p)
            for key in (("full", n), ("head", n[:120])):
                if key in seen:
                    ra, rb = find(seen[key]), find(r["item_id"])
                    if ra != rb:
                        parent[ra] = rb
                else:
                    seen[key] = r["item_id"]
    return {i: find(i) for i in parent}


group_of = related_groups(sample_rows)
groups = df["item_id"].map(group_of).to_numpy()
sizes = pd.Series(groups).value_counts()
print(f"Related-item groups: {len(sizes)} groups for {len(df)} items | "
      f"items sharing a passage with another item: {int(sizes[sizes > 1].sum())} | largest group: {int(sizes.max())}")

# ---------- 2. out-of-fold P(correct) for each configuration ----------
if args.cv == "grouped":
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    cv_groups = groups
else:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    cv_groups = None
# how many related groups are split across folds under the chosen scheme?
fold_of = np.empty(len(df), dtype=int)
for k, (_, test_idx) in enumerate(cv.split(df[["confidence"]], y, cv_groups)):
    fold_of[test_idx] = k
split_groups = pd.DataFrame({"g": groups, "f": fold_of}).groupby("g")["f"].nunique()
print(f"Related groups split across folds: {int((split_groups > 1).sum())}")

scores = {}
for name, feats in CONFIGS.items():
    clf = LogisticRegression(max_iter=1000)
    proba = np.asarray(cross_val_predict(clf, df[feats], y, cv=cv, groups=cv_groups,
                                         method="predict_proba"))
    scores[name] = proba[:, 1]
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

# likelihood-ratio test on nested logistic models (full data, unpenalised, in-sample by definition)
def fit_llf(feats, mask=None):
    sub = df if mask is None else df[mask]
    yy = y if mask is None else y[mask]
    X = sub[feats].astype(float).to_numpy()
    m = LogisticRegression(C=1e12, max_iter=5000).fit(X, yy)   # huge C = effectively unpenalised
    p = np.clip(m.predict_proba(X)[:, 1], 1e-12, 1 - 1e-12)
    llf = float(np.sum(yy * np.log(p) + (1 - yy) * np.log(1 - p)))
    coefs = pd.Series(np.r_[m.intercept_, m.coef_[0]], index=["const"] + feats)
    return llf, coefs


def lr_test(small, big, mask=None):
    l_s, _ = fit_llf(CONFIGS[small], mask)
    l_b, coefs = fit_llf(CONFIGS[big], mask)
    stat = max(0.0, 2 * (l_b - l_s))
    dof = len(CONFIGS[big]) - len(CONFIGS[small])
    return stat, dof, chi2.sf(stat, dof), coefs


lr_stat, lr_df, lr_p, coefs4 = lr_test("C2_conf_suff", "C4_conf_suff_nli")
rb_stat, rb_df, rb_p, coefs_r2 = lr_test("R1_cond_conf_suff", "R2_cond_conf_suff_nli")

lines = [
    "=== RQ1: does contradiction add information beyond confidence + sufficiency? ===",
    f"AUROC C4 - C2 = {d_auroc:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]",
    f"AURC  C2 - C4 = {d_aurc:+.4f}  95% CI [{lo2:+.4f}, {hi2:+.4f}]  (positive = C4 better)",
    f"Likelihood-ratio test C2 vs C4: LR = {lr_stat:.2f}, df = {lr_df}, p = {lr_p:.4g}",
    f"C4 coefficients:\n{coefs4.round(3).to_string()}",
    "",
    "=== Robustness: does contradiction help BEYOND knowing the condition? ===",
    f"Likelihood-ratio test R1 vs R2: LR = {rb_stat:.2f}, df = {rb_df}, p = {rb_p:.4g}",
    f"R2 coefficients:\n{coefs_r2.round(3).to_string()}",
    "",
    "=== McNemar: wrong answers at fixed coverage, C2 vs C4 ===",
]
for c in COVERAGES:
    w2 = answered_mask(s2, c) & (y == 0)
    w4 = answered_mask(s4, c) & (y == 0)
    table = [[np.sum(w2 & w4), np.sum(w2 & ~w4)],
             [np.sum(~w2 & w4), np.sum(~w2 & ~w4)]]
    b, c_ = int(table[0][1]), int(table[1][0])
    p_mcn = binomtest(b, b + c_, 0.5).pvalue if b + c_ > 0 else 1.0
    lines.append(f"  coverage {int(c*100)}%: wrong C2={w2.sum()}, C4={w4.sum()} | "
                 f"discordant {b} vs {c_} | p = {p_mcn:.4g}")
text = "\n".join(lines)
print("\n" + text)
with open(f"{OUT}/rq1_tests.txt", "w") as f:
    f.write(text + "\n")

# ---------- 5. RQ2: behaviour per condition ----------
per = []
for name in CONFIGS:
    for c in COVERAGES:
        m = answered_mask(scores[name], c)
        for cond, g in df.groupby("condition"):
            idx = g.index.to_numpy()
            per.append({"config": name, "coverage": c, "condition": cond,
                        "abstain_rate": 1 - m[idx].mean(),
                        "wrong_answer_rate": (m[idx] & (y[idx] == 0)).mean()})
per = pd.DataFrame(per)
per.to_csv(f"{OUT}/per_condition.csv", index=False)
print("\n=== RQ2: per-condition behaviour at 50% coverage ===")
print(per[per.coverage == 0.5].pivot(index="config", columns="condition",
                                     values=["abstain_rate", "wrong_answer_rate"]).round(3).to_string())

# ---------- 5b. per source subset (proposal: analyse results per source dataset) ----------
sub_rows = []
for subset, g in df.groupby("subset"):
    idx = g.index.to_numpy()
    row = {"subset": subset, "n": len(idx), "accuracy": y[idx].mean()}
    for name in MAIN:
        row[name] = roc_auc_score(y[idx], scores[name][idx]) if len(set(y[idx])) > 1 else float("nan")
    row["C4_minus_C2"] = row["C4_conf_suff_nli"] - row["C2_conf_suff"]
    sub_rows.append(row)
per_subset = pd.DataFrame(sub_rows)
per_subset.to_csv(f"{OUT}/per_subset.csv", index=False)
print("\n=== AUROC per source subset ===")
print(per_subset.round(3).to_string(index=False))

# ---------- 5c. within-condition analysis (supervisor check 4) ----------
# Does contradiction predict correctness INSIDE a condition, or only tell conditions apart?
def safe_auc(yy, s):
    return roc_auc_score(yy, s) if len(set(yy)) > 1 and len(set(s)) > 1 else float("nan")


wc_rows = []
for cond, g in df.groupby("condition"):
    m = (df["condition"] == cond).to_numpy()
    yy = y[m]
    n_pos, n_neg = int(yy.sum()), int((1 - yy).sum())
    row = {"condition": cond, "n": int(m.sum()), "n_correct": n_pos, "n_wrong": n_neg,
           "auroc_confidence": safe_auc(yy, df.loc[m, "confidence"].to_numpy()),
           "auroc_sufficiency": safe_auc(yy, df.loc[m, "sufficiency"].to_numpy()),
           "auroc_no_contradiction": safe_auc(yy, -df.loc[m, "contradiction"].to_numpy()),
           "auroc_C2": safe_auc(yy, s2[m]), "auroc_C4": safe_auc(yy, s4[m])}
    if min(n_pos, n_neg) >= 10 and df.loc[m, "contradiction"].nunique() > 1:
        st, _, pv, cf = lr_test("C2_conf_suff", "C4_conf_suff_nli", m)
        row.update({"lr_stat": st, "lr_p": pv, "contradiction_coef": cf["contradiction"]})
    else:
        row.update({"lr_stat": float("nan"), "lr_p": float("nan"), "contradiction_coef": float("nan")})
    wc_rows.append(row)
within = pd.DataFrame(wc_rows)
within.to_csv(f"{OUT}/within_condition.csv", index=False)
print("\n=== Within-condition analysis (AUROC for correctness inside each condition) ===")
print(within.round(3).to_string(index=False))
print("lr_p: likelihood-ratio test for adding contradiction to confidence + sufficiency, "
      "fitted inside the condition (blank if fewer than 10 items in a class or the signal is constant)")

# ---------- 5d. generator abstention vs gate abstention (supervisor check 3) ----------
# (i) committed answers only: can the gate tell right from wrong when the generator DID answer?
cm = committed == 1
co_rows = []
for name in MAIN:
    co_rows.append({"config": name, "n": int(cm.sum()), "n_correct": int(y[cm].sum()),
                    "auroc_committed_only": safe_auc(y[cm], scores[name][cm])})
committed_only = pd.DataFrame(co_rows)
st_c, _, p_c, _ = lr_test("C2_conf_suff", "C4_conf_suff_nli", cm)
committed_only.to_csv(f"{OUT}/committed_only.csv", index=False)
print("\n=== Committed answers only (generator abstentions removed) ===")
print(committed_only.round(3).to_string(index=False))
print(f"Likelihood-ratio test C2 vs C4 on committed answers: LR = {st_c:.2f}, p = {p_c:.4g}")

# (ii) what each system finally returns, separating the two kinds of abstention
def outcomes(score, coverage):
    a = answered_mask(score, coverage)
    return {
        "answer shown, correct": int(np.sum(a & cm & (y == 1))),
        "answer shown, WRONG": int(np.sum(a & cm & (y == 0))),
        "generator abstained, passed through": int(np.sum(a & ~cm)),
        "gate abstained, blocked a wrong answer": int(np.sum(~a & cm & (y == 0))),
        "gate abstained, lost a correct answer": int(np.sum(~a & cm & (y == 1))),
        "gate abstained, generator had abstained": int(np.sum(~a & ~cm)),
    }


oc_rows = []
for c in COVERAGES:
    o2, o4 = outcomes(s2, c), outcomes(s4, c)
    for k in o2:
        oc_rows.append({"coverage": c, "outcome": k, "C2": o2[k], "C4": o4[k]})
outc = pd.DataFrame(oc_rows)
outc.to_csv(f"{OUT}/outcomes.csv", index=False)
print("\n=== Final system outcome at 50% coverage (items) ===")
print(outc[outc.coverage == 0.5][["outcome", "C2", "C4"]].to_string(index=False))

# ---------- 6. risk-coverage plot ----------
plt.figure(figsize=(6, 4))
for name in MAIN:
    cov, risk = risk_coverage(scores[name], y)
    plt.plot(cov, risk, label=f"{name} (AURC {aurc(scores[name], y):.3f})")
plt.xlabel("Coverage (share of questions answered)")
plt.ylabel("Risk (error rate on answered)")
plt.title("Risk-coverage" + (" [SIMULATED]" if args.simulate else ""))
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig(f"{OUT}/risk_coverage.png", dpi=200)

df.to_csv(f"{OUT}/item_level.csv", index=False)
print(f"\nSaved to {OUT}/")