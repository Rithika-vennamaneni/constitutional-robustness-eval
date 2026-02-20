## Preliminary Observations

During structured JSON evaluation under adversarial framing:

- Total adversarial samples: 70
- Parse failures: 9 (12.86%)

Breakdown by condition:
- Baseline: 2 parse failures
- Constitutional: 7 parse failures

This suggests that adding lightweight constitutional instructions may increase structured output instability under adversarial prompts. 

This is a structural effect (format adherence), not a safety behavior effect, and will be analyzed separately from refusal behavior metrics.