# Archive

This folder contains outputs from prior pipeline runs, preserved for reference.

---

## Archived from results/current/ (2026-03-01 — superseded by v2 pipeline with new templates)

- `README.md`: The original README for results/current/ that described the v1 pipeline. Archived when the folder was restructured for the v2 run (7 new adversarial templates including escalation).

- `llama_all_responses.jsonl`: Full merged Llama response file (baseline + constitutional combined) generated during the v1 run using old adversarial templates (defensive, teacher_role, security_audit, law_enforcement, etc.). Superseded by the split `llama_baseline_responses.jsonl` / `llama_constitutional_responses.jsonl` produced from `adversarial_prompts_v2.jsonl`.

- `llama_responses_merged_v2.jsonl`: Intermediate merged output file produced during the v2 Llama run before splitting by condition. Superseded by the clean split files in results/current/.

- `gpt4o_judge_labels.jsonl`: GPT-4o-mini judge labels (behavior/confidence/reasoning) generated for the v1 Llama responses. Based on old prompt templates; not comparable to the v2 run.

- `structural_behavioral_gaps.jsonl`: Disagreement cases from the v1 pipeline where Llama self-report (refused field) and GPT-4o judge behavior label diverged. Superseded by the v2 disagreement files.

- `structural_behavioral_gaps.csv`: Spreadsheet export of `structural_behavioral_gaps.jsonl`. Archived alongside the JSONL.

- `audit_sample.jsonl`: Random audit subset sampled from the v1 judged run for manual review. No longer the active audit set.

---

## Previously archived runs

- `dataset_adv_001.jsonl`, `dataset_adv_v1.jsonl`: Early adversarial dataset runs.
- `dataset_run_001.jsonl`, `run_001.jsonl`: Early model response runs.
- `metrics_basic.csv`: Basic metrics from early evaluation pass.
