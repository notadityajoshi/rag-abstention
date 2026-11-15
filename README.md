# Conflict-Aware Abstention in Retrieval-Augmented Generation

MSc Artificial Intelligence and Human Factors, Coventry University
Module 7005SCN Individual Research Project. Author: Aditya Joshi. Supervisor: Dr Manizheh Montazerian.
Ethics approval: P197033 (public secondary data only; no human participants).

## Purpose

Tests whether an explicit, training-free NLI contradiction signal improves selective question answering beyond model confidence and a context-sufficiency judgement (Joren et al., 2025), on FaithEval (Ming et al., 2025) inconsistent, unanswerable and answerable-control contexts.

Four abstention configurations are compared (logistic regression, 5-fold cross-validation):

| Config | Signals |
|---|---|
| C1 | confidence |
| C2 | confidence + sufficiency (Joren-style baseline) |
| C3 | confidence + NLI contradiction |
| C4 | confidence + sufficiency + NLI contradiction (proposed) |

Robustness configurations R0–R2 add the condition label as an oracle feature.

## Primary analysis and variants

After a supervisor code review (14 Nov 2026) the primary analysis was revised. The original (frozen) analysis is kept and reported beside it.

| | Original design (frozen, 12 Nov) | Primary analysis (revised, 14 Nov) |
|---|---|---|
| Confidence | P(Correct): model's own two best guesses with probabilities | Answer-specific: probability that the *generated* answer is correct |
| Cross-validation | Stratified 5-fold | Grouped stratified 5-fold (items sharing a passage stay in one fold) |
| Contradiction | `nli_sym_t30` (all aligned pairs) | `nli_sym_t30` (unchanged, pre-registered) |
| Output folder | `results/final/` | `results/variants/conf-given__nli_sym_t30__cv-grouped/` |

Why: the P(Correct) first guess matched the generated answer in only 68% of items, and 161 groups of related items were split across folds. A cross-passage-only contradiction score (`nli_cross_t30`) was added as a variant but not made primary, because it was chosen after seeing that it performed better.

## Repository structure

```
src/
  download_data.py        # download FaithEval subsets from Hugging Face
  inspect_data.py         # verify passage structure and subsets
  diagnose_split.py       # diagnose answer/passage matching failures
  build_sample.py         # deduplicated, stratified 1500-item sample (seed 42)
  nli_signal.py           # DeBERTa-v3 NLI signal: all-pairs and cross-passage scores,
                          #   plus failure-stage diagnosis (local, no API)
  analyse_nli.py          # NLI flag rates and miss diagnostics (original v2 run)
  label_sufficiency.py    # blind manual sufficiency labelling (50 items)
  llm_signals.py          # answers, P(Correct), sufficiency autorater, judges (API)
  confidence_given.py     # answer-specific confidence + P(Correct) agreement check (API)
  validate_autorater.py   # autorater vs manual labels (accuracy, Cohen's kappa)
  label_judge.py          # blind manual grading of 50 model answers
  validate_judge.py       # LLM judge vs manual grading (accuracy, Cohen's kappa)
  analysis.py             # AUROC, AURC, LR tests, McNemar, bootstrap CIs, per-condition,
                          #   per-subset, within-condition, committed-only, outcome breakdown
  report_figures.py       # descriptive figures and tables for the report
tests/
  test_parsers.py              # unit tests: LLM output parsing, prompt rules
  test_data_integrity.py       # sample balance, no leakage, signal completeness
  test_analysis_smoke.py       # end-to-end analysis run on simulated labels
  test_supervisor_checks.py    # cross-passage score, failure stages, answer matching
data/
  labels/sufficiency_manual.jsonl   # author's 50 blind manual sufficiency labels
  labels/judge_manual.jsonl         # author's 50 blind manual grading labels
  raw/, processed/                  # generated locally, not committed (see Data)
docs/
  labelling_guide.md                # sufficiency labelling rules
results/
  final/                            # original (frozen) analysis
  variants/                         # primary analysis and supervisor-review variants
  sensitivity/                      # original analysis at NLI thresholds 0.2 / 0.4 / 0.5
  figures/                          # report figures and descriptive tables
requirements.txt
.env.example
```

## Requirements

- macOS or Linux, Python 3.10+ (developed on Python 3.14, Apple Silicon)
- About 1 GB disk for the NLI model; no GPU required (uses Apple MPS if available, otherwise CPU)
- An OpenRouter API key (or OpenAI key) for `llm_signals.py` and `confidence_given.py` only. Total cost: under US$5.

## Installation

```bash
git clone https://github.com/notadityajoshi/rag-abstention.git
cd rag-abstention
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
mkdir -p data/raw data/processed
cp .env.example .env        # then paste your API key into .env
```

## Running the pipeline (in order)

```bash
# 1. data
python src/download_data.py                    # ~1 min
python src/build_sample.py                     # prints "ALL CHECKS PASSED"

# 2. signals
python src/nli_signal.py --pilot               # 30-item pilot
python src/nli_signal.py                       # full NLI run, ~10 min (resumable)
python src/llm_signals.py --pilot              # 30-item API pilot
python src/llm_signals.py                      # full API run, ~45 min (resumable)
PYTHONPATH=src python src/confidence_given.py  # answer-specific confidence (resumable)

# 3. human validation (interactive)
python src/validate_autorater.py
python src/label_judge.py
python src/validate_judge.py

# 4. analysis
python src/analysis.py --confidence given --cv grouped      # PRIMARY
python src/analysis.py                                      # original (frozen) design
python src/analysis.py --cv grouped                         # variants
python src/analysis.py --confidence given
python src/analysis.py --nli-key nli_cross_t30
python src/analysis.py --confidence given --nli-key nli_cross_t30 --cv grouped
for k in nli_sym_t20 nli_sym_t40 nli_sym_t50; do python src/analysis.py --nli-key $k; done

# 5. figures and tables
python src/report_figures.py
```

`analysis.py` options:

| Option | Values | Meaning |
|---|---|---|
| `--confidence` | `pcorrect` (default), `given` | P(Correct) or answer-specific confidence |
| `--cv` | `stratified` (default), `grouped` | grouped keeps items that share a passage in the same fold |
| `--nli-key` | `nli_sym_t20`…`t50`, `nli_cross_t20`…`t50` | contradiction score and alignment threshold |
| `--simulate` | flag | fake labels, to test the code without API data |

## Testing

```bash
python -m pytest tests/ -v
```
21 tests:
- `test_parsers.py` (9): parsing of every LLM output type; the generator prompt contains no "unknown"/"conflict" instruction.
- `test_data_integrity.py` (6): 500 items per condition, no duplicate IDs or questions, correct passage structure, no gold answer in control distractors, complete NLI and LLM signals (skipped until the data exist).
- `test_analysis_smoke.py` (1): the full analysis runs on simulated labels and writes valid outputs.
- `test_supervisor_checks.py` (5): cross-passage score ignores same-passage pairs, failure-stage attribution, answer matching, answer-specific confidence parsing.

All scripts that call models write results line by line and resume after interruption.

## Key settings

- **Sample:** 500 items per condition, stratified by FaithEval source subset, random seed 42; duplicate questions removed before sampling.
- **NLI:** `cross-encoder/nli-deberta-v3-base`; top-10 question-relevant sentences; only aligned sentence pairs (TF-IDF similarity ≥ 0.3, frozen on the pilot); symmetric contradiction (minimum of both directions); maximum over pairs. The cross-passage variant uses only pairs from different passages.
- **Generator and judges:** GPT-4o-mini, temperature 0, via OpenRouter (`openai/gpt-4o-mini`).
- **Generator prompt:** FaithEval default prompt without the "unknown"/"conflict" instructions.
- **Grading:** controls judged with the Joren et al. (2025) LLMEval prompt; on inconsistent and unanswerable items, committing to an answer counts as wrong and abstaining as correct.

## Results summary

Primary analysis (`results/variants/conf-given__nli_sym_t30__cv-grouped/`):

| Config | AUROC (95% CI) | AURC |
|---|---|---|
| C1 | 0.641 (0.615–0.668) | 0.486 |
| C2 | 0.647 (0.623–0.676) | 0.481 |
| C3 | 0.748 (0.723–0.771) | 0.409 |
| C4 | 0.751 (0.725–0.774) | 0.399 |

- C4 vs C2: AUROC +0.103 (95% CI 0.074–0.132); likelihood-ratio test p < 0.001.
- Wrong answers at 75 / 50 / 25% coverage: 518 → 523 (p = 0.68), 353 → 265 (p < 0.001), 171 → 133 (p = 0.012).
- Committed answers only (n = 1,301): AUROC 0.593 → 0.775.
- Within each condition, contradiction adds no information (LR p = 0.98, 0.38, 0.78); given the condition label, none either (R1 vs R2, p = 0.51). The signal detects the presence of conflict; it does not predict individual errors.

Robustness of the C4 − C2 gain:

| Variant | C4 − C2 AUROC (95% CI) |
|---|---|
| Original design (frozen) | +0.122 (0.096–0.149) |
| P(Correct), grouped folds | +0.128 (0.099–0.157) |
| Answer-specific confidence, stratified folds | +0.100 (0.073–0.127) |
| **Primary** | **+0.103 (0.074–0.132)** |
| Primary with cross-passage contradiction | +0.156 (0.124–0.188) |
| Original design, NLI thresholds 0.2–0.5 | +0.100 to +0.136 |

Validation and diagnostics:
- Sufficiency autorater vs manual labels: 88% agreement, Cohen's kappa 0.69.
- LLM judge vs manual grading: 98% agreement, Cohen's kappa 0.96.
- P(Correct) first guess matched the generated answer in 68% of items.
- Inconsistent items by detection stage: 51% detected; 16% selection, 14% NLI and 11% alignment failures; 7% not attributable.
- NLI v3 rerun reproduced the frozen all-pairs scores exactly (max difference 0.0).

## Known limitations

- FaithEval conflicts are synthetic entity swaps; real-world conflicts may be harder to detect.
- Correctness is strongly tied to condition; NLI acts as a conflict-presence detector, not an error predictor.
- The cross-passage score's advantage is partly structural: unanswerable items have one passage, so it is always 0 there.
- Sentence-level NLI misses edits that are different but not logically contradictory.
- The primary confidence measure and fold scheme were revised after the main run, at the supervisor's request; the original design is reported alongside.
- Single generator model; autorater agreement (kappa 0.69) is below the 93% accuracy reported by Joren et al.
- Original passage always appears first in inconsistent contexts, so position bias and prior knowledge cannot be separated.
- The model was accessed via OpenRouter because the OpenAI account was held on the free tier.

## Data and third-party resources

- FaithEval (Salesforce AI Research), Hugging Face: `Salesforce/FaithEval-inconsistent-v1.0` and `Salesforce/FaithEval-unanswerable-v1.0`. Not redistributed in this repository; download with `src/download_data.py`. See the dataset cards for licence terms.
- NLI model `cross-encoder/nli-deberta-v3-base` (Hugging Face); see its model card for licence terms.
- Prompts adapted from Joren et al. (2025, Appendix C) and Ming et al. (2025).
- No personal data is stored. API keys are kept in `.env`, which is git-ignored.

## References

- Joren, H., Zhang, J., Ferng, C.-S., Juan, D.-C., Taly, A., & Rashtchian, C. (2025). Sufficient context: A new lens on retrieval augmented generation systems. *ICLR 2025*.
- Ming, Y., Purushwalkam, S., Pandit, S., Ke, Z., Nguyen, X.-P., Xiong, C., & Joty, S. (2025). FaithEval: Can your language model stay faithful to context, even if "the moon is made of marshmallows". *ICLR 2025*.
- He, P., Gao, J., & Chen, W. (2023). DeBERTaV3: Improving DeBERTa using ELECTRA-style pre-training with gradient-disentangled embedding sharing. *ICLR 2023*.

## AI use

Developed with AI assistance (Claude, Anthropic), as declared in the project report. All manual labels are the author's.