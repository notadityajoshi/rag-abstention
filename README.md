# Conflict-Aware Abstention in Retrieval-Augmented Generation

MSc Artificial Intelligence and Human Factors, Coventry University
Module 7005SCN Individual Research Project. Author: Aditya Joshi. Supervisor: Dr Manizheh Montazerian.
Ethics approval: P197033 (public secondary data only; no human participants).

## Purpose

Tests whether an explicit, training-free NLI contradiction signal improves selective question answering beyond model confidence and a context-sufficiency judgement (Joren et al., 2025), on FaithEval (Ming et al., 2025) inconsistent, unanswerable and answerable-control contexts.

Four abstention configurations are compared (logistic regression, 5-fold cross-validation):

| Config | Signals |
|---|---|
| C1 | confidence (verbalised P(Correct)) |
| C2 | confidence + sufficiency (Joren-style baseline) |
| C3 | confidence + NLI contradiction |
| C4 | confidence + sufficiency + NLI contradiction (proposed) |

Robustness configurations R0–R2 add the condition label as an oracle feature.

## Repository structure

```
src/
  download_data.py        # download FaithEval subsets from Hugging Face
  inspect_data.py         # verify passage structure and subsets
  diagnose_split.py       # diagnose answer/passage matching failures
  build_sample.py         # deduplicated, stratified 1500-item sample (seed 42)
  nli_signal.py           # DeBERTa-v3 NLI contradiction signal (local, no API)
  analyse_nli.py          # NLI flag rates and miss diagnostics
  label_sufficiency.py    # blind manual sufficiency labelling (50 items)
  llm_signals.py          # answers, P(Correct), sufficiency autorater, judges (API)
  validate_autorater.py   # autorater vs manual labels (accuracy, Cohen's kappa)
  label_judge.py          # blind manual grading of 50 model answers
  validate_judge.py       # LLM judge vs manual grading (accuracy, Cohen's kappa)
  analysis.py             # AUROC, AURC, LR tests, McNemar, bootstrap CIs, per-condition/subset, plots
  report_figures.py       # descriptive figures and tables for the report
tests/
  test_parsers.py         # unit tests: LLM output parsing, prompt rules
  test_data_integrity.py  # sample balance, no leakage, signal completeness
  test_analysis_smoke.py  # end-to-end analysis run on simulated labels
data/
  labels/sufficiency_manual.jsonl   # author's 50 blind manual sufficiency labels
  labels/judge_manual.jsonl         # author's 50 blind manual grading labels
  raw/, processed/                  # generated locally, not committed (see Data)
docs/
  labelling_guide.md                # sufficiency labelling rules
results/
  final/                            # frozen results: metrics, tests, plot, item-level table
  sensitivity/                      # same analysis at NLI thresholds 0.2 / 0.4 / 0.5
  figures/                          # report figures and descriptive tables
requirements.txt
.env.example
```

## Requirements

- macOS or Linux, Python 3.10+ (developed on Python 3.14, Apple Silicon)
- About 1 GB disk for the NLI model; no GPU required (uses Apple MPS if available, otherwise CPU)
- An OpenRouter API key (or OpenAI key) for `llm_signals.py` only. Full run cost: under US$3.

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
python src/download_data.py              # ~1 min
python src/build_sample.py               # prints "ALL CHECKS PASSED"
python src/nli_signal.py --pilot         # 30-item pilot
python src/nli_signal.py                 # full NLI run, ~10 min (resumable)
python src/llm_signals.py --pilot        # 30-item API pilot
python src/llm_signals.py                # full API run, ~45 min (resumable)
python src/validate_autorater.py         # autorater vs manual labels
python src/label_judge.py                # manual grading of 50 answers (interactive)
python src/validate_judge.py             # LLM judge vs manual grading
python src/analysis.py                   # writes results/final/
for k in nli_sym_t20 nli_sym_t40 nli_sym_t50; do python src/analysis.py --nli-key $k; done
python src/report_figures.py             # writes results/figures/
```

## Testing

```bash
python -m pytest tests/ -v
```
- `test_parsers.py`: parsing of every LLM output type, and a check that the generator prompt contains no "unknown"/"conflict" instruction.
- `test_data_integrity.py`: 500 items per condition, no duplicate IDs or questions, correct passage structure, no gold answer in control distractors, complete NLI and LLM signals (skipped until the data exist).
- `test_analysis_smoke.py`: the full analysis runs on simulated labels and writes valid outputs.

To test the analysis code without any API calls (fake labels, written to `results/simulated/`):

```bash
python src/analysis.py --simulate
```

All scripts that call models write results line by line and resume after interruption.

## Key settings

- **Sample:** 500 items per condition, stratified by FaithEval source subset, random seed 42; duplicate questions removed before sampling.
- **NLI:** `cross-encoder/nli-deberta-v3-base`; top-10 question-relevant sentences; only aligned sentence pairs (TF-IDF similarity ≥ 0.3, frozen on the pilot); symmetric contradiction (minimum of both directions); maximum over pairs.
- **Generator and judges:** GPT-4o-mini, temperature 0, via OpenRouter (`openai/gpt-4o-mini`).
- **Generator prompt:** FaithEval default prompt without the "unknown"/"conflict" instructions.
- **Grading:** controls judged with the Joren et al. (2025) LLMEval prompt; on inconsistent and unanswerable items, committing to an answer counts as wrong and abstaining as correct.

## Results summary (`results/final/`)

| Config | AUROC | AURC |
|---|---|---|
| C1 | 0.605 | 0.454 |
| C2 | 0.595 | 0.470 |
| C3 | 0.718 | 0.393 |
| C4 | 0.717 | 0.393 |

- C4 vs C2: AUROC +0.122 (95% CI 0.096–0.149); likelihood-ratio test p < 0.001.
- Fewer wrong answers with C4 at 75%, 50% and 25% coverage (McNemar p < 0.001).
- Given the condition label, contradiction adds no further information (R1 vs R2, p = 0.27).
- Sufficiency autorater vs manual labels: 88% agreement, Cohen's kappa 0.69.
- LLM judge vs manual grading: 98% agreement, Cohen's kappa 0.96.
- Robust to NLI threshold (0.2–0.5): C4 − C2 AUROC between +0.100 and +0.136, all CIs exclude 0 (`results/figures/table_sensitivity.csv`).
- Test suite: 16/16 tests pass (`python -m pytest tests/ -v`).

## Known limitations

- FaithEval conflicts are synthetic entity swaps; real-world conflicts may be harder to detect.
- Correctness is strongly tied to condition; NLI acts mainly as a conflict-presence detector.
- Sentence-level NLI misses edits that are different but not logically contradictory.
- Single generator model; autorater agreement (kappa 0.69) is below the 93% accuracy reported by Joren et al.
- The model was accessed via OpenRouter because the OpenAI account was held on the free tier.
- Original passage always appears first in inconsistent contexts, so position bias and prior knowledge cannot be separated.

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

Developed with AI assistance (Claude, Anthropic), as declared in the project report. All design decisions, manual labels and interpretation of results are the author's.