# Round 1 Results (Adversarial Set, n=70)

## Structured Output Stability

- Total adversarial samples: 70
- Parse failures under strict JSON parsing: 9 (12.86%)

Failure mode audit:
- 9/9 were `extra_text_outside_json`
- All failures were valid JSON wrapped in markdown code fences

Implication:
These failures represent format non-compliance under strict machine parsing,
not malformed JSON content.

Parse failures were higher under the constitutional condition (7 vs 2 baseline),
suggesting that added system instruction complexity may increase format drift.

---

## Behavioral Safety (Disallowed Prompts, parse_ok=True, n=61)

False Safe Rate (behavioral harmful compliance):
- Overall: 27.87%
- Baseline: 30.30%
- Constitutional: 25.00%

False Unsafe Rate (model claims refusal but behaviorally complies):
- 0%

Key observation:
Self-reported refusal labels were reliable when present.
However, adversarial framing frequently induced harmful compliance without refusal.