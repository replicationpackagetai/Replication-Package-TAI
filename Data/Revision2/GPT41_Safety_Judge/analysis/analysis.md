# GPT-4.1 Safety Judge Analysis

This analysis judges selected PEFT settings across 4 models and 3 rounds on HEx-PHI responses.

Settings:
- IA3.
- LoRA.
- LoRA (r=8).
- Prompt Tuning.
- P-Tuning.

Estimated GPT-4.1 API cost from returned usage: $29.68.

## Setting-Level Averages

| Setting | GPT unsafe | Llama Guard unsafe | Agreement | Kappa | LG-safe agreement | LG-unsafe agreement |
|---|---:|---:|---:|---:|---:|---:|
| IA3 | 21.1% | 17.8% | 90.0% | 0.639 | 90.1% | 73.8% |
| LoRA | 15.8% | 13.8% | 91.8% | 0.653 | 93.5% | 71.7% |
| LoRA (r=8) | 14.1% | 12.6% | 93.1% | 0.671 | 94.9% | 75.0% |
| Prompt Tuning | 28.5% | 28.8% | 86.0% | 0.588 | 88.5% | 69.5% |
| P-Tuning | 32.5% | 34.1% | 88.6% | 0.688 | 88.1% | 81.0% |

## Charts

![Unsafe rate by setting](charts/unsafe_rate_by_setting.svg)

![Agreement and kappa by setting](charts/agreement_kappa_by_setting.svg)

![Class-specific agreement by setting](charts/class_specific_agreement_by_setting.svg)

![GPT unsafe rate by model and setting](charts/gpt_unsafe_rate_by_model_setting.svg)
