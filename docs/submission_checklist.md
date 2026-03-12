# Submission Checklist (Tailored To Current Repo)

Use this checklist before finalizing the paper draft in `docs/paper.md`.

## 1. Freeze Artifacts
- [ ] Confirm these are the canonical result files for the paper:
- [ y ] `results/current/llama_baseline_responses.jsonl`
- [ y ] `results/current/llama_constitutional_responses.jsonl`
- [ y ] `results/current/gpt_baseline_responses.jsonl`
- [ y ] `results/current/gpt_constitutional_responses.jsonl`
- [ y ] `results/current/refused_for_review.csv`
- [ y ] `results/current/manual_review.csv`
- [ y ] `results/current/summary_analysis.csv`
- [ y ] `results/current/manual_review_summary.txt`
- [ y ] `results/current/disagreements_baseline.jsonl`
- [ y ] `results/current/disagreements_constitutional.jsonl`
- [ y ] `results/current/disagreements_llama_condition.jsonl`
- [ ] Confirm all old exploratory files remain in `results/archive/` and are not cited as final.

## 2. Data Integrity
- [ ] Verify row counts for each stage are documented in Methods.
- [ ] Verify that every reported denominator in `docs/paper.md` matches `results/current/summary_analysis.csv`.
- [ y ] Resolve or explicitly explain `condition=unknown` rows in `results/current/summary_analysis.csv`.
- [ ] Resolve or explicitly explain `category=unknown` rows (even if zero-count placeholders).
- [ ] Confirm no accidental duplicate rows by `(id, model, condition)` in analysis inputs.

## 3. Labeling and Review Quality
- [ ] In `docs/paper.md`, define all four behavioral labels exactly as used:
- [ ] `harmful_compliance`
- [ ] `soft_refusal_leakage`
- [ ] `safe_completion`
- [ ] `clean_refusal`
- [ ] Document the manual review protocol used for `results/current/manual_review.csv`.
- [ ] Report how `refused_for_review.csv` and `manual_review.csv` are combined (as implemented in `src/summary_analysis.py`).
- [ ] Add at least 1 example per failure type in the paper appendix or results section.

## 4. Statistical Strengthening
- [ ] Add confidence intervals (95%) for headline rates in Results.
- [ ] Add significance tests for core comparisons:
- [ ] Llama vs GPT refusal behavior.
- [ ] Baseline vs constitutional condition effects.
- [ ] Template-level differences.
- [ ] Report effect sizes (risk difference and/or odds ratio), not just percentages.
- [ ] Mark exploratory analyses explicitly as exploratory.

## 5. Robustness and Validity
- [ ] Include a section on LLM-as-judge limitations and possible bias.
- [ ] Include parse-failure handling rules and exclusion counts.
- [ ] Include template sensitivity caveat (framing can itself drive outcomes).
- [ ] Include category imbalance caveats if applicable.
- [ ] Include potential API/model drift caveat (time-dependent behavior).

## 6. Reproducibility Package
- [ ] Add exact command list to reproduce final tables from raw outputs.
- [ ] Record Python version and key package versions used for final run.
- [ ] Record exact model names used in final experiments:
- [ ] responder model(s)
- [ ] judge model(s)
- [ ] Save commit hash in `docs/results_round1.md` or paper appendix.
- [ ] Ensure scripts run from repo root without path edits.

## 7. Writing Quality Checks
- [ ] Every claim in `docs/paper.md` has a traceable source file.
- [ ] Results narrative distinguishes observed fact from interpretation.
- [ ] Limit overclaiming; explicitly state scope and external validity limits.
- [ ] Include a clear threat-to-validity section.
- [ ] Include an ethics/safety section for disclosure framing.

## 8. Pre-Submission Sanity Run
- [ ] Re-run final analysis script(s) to regenerate:
- [ ] `results/current/summary_analysis.csv`
- [ ] `results/current/manual_review_summary.txt`
- [ ] Confirm regenerated outputs match values cited in the paper.
- [ ] Spot-check 10 random rows across final outputs for field consistency.

## 9. Final Deliverables
- [ ] `docs/paper.md` updated with final numbers.
- [ ] `docs/results_round1.md` updated or superseded with final-round summary.
- [ ] `docs/submission_checklist.md` fully checked off.
- [ ] Archive snapshot retained and untouched.
