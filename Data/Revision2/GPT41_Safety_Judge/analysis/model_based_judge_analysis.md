# Model-Based GPT-4.1 vs LLaMA-Guard 2 Analysis

This note summarizes the base-model-level comparison between GPT-4.1 and
LLaMA-Guard 2 on the five-setting HEx-PHI judge robustness check:

- Original IA3
- Original LoRA
- LoRA r=8
- Original Prompt Tuning
- Original P-Tuning

Each base model aggregates 4,950 prompt-response judgments:

```text
5 settings x 3 rounds x 330 prompts = 4,950 judgments per base model
```

## Model-Level Agreement

| Model | GPT unsafe | LG unsafe | Diff GPT-LG | Agreement | Kappa | McNemar p | Direction |
|---|---:|---:|---:|---:|---:|---:|---|
| Mistral | 48.7% | 43.2% | +5.5 pp | 82.7% | 0.654 | 5.77e-21 | GPT marks more unsafe |
| LLaMA | 7.8% | 9.7% | -1.9 pp | 93.7% | 0.607 | 1.94e-07 | LLaMA-Guard marks more unsafe |
| Gemma | 13.9% | 15.4% | -1.4 pp | 93.4% | 0.734 | 0.000107 | LLaMA-Guard marks more unsafe |
| Qwen | 19.1% | 17.5% | +1.6 pp | 89.8% | 0.658 | 0.00052 | GPT marks more unsafe |

## Model Ordering

Aggregated over all five judged settings, both judges produce the same
model-level unsafe-rate order:

```text
Mistral > Qwen > Gemma > LLaMA
```

Setting-level orderings:

| Setting | GPT-4.1 order | LLaMA-Guard 2 order |
|---|---|---|
| Original IA3 | Mistral > Qwen > Gemma > LLaMA | Mistral > Qwen > Gemma > LLaMA |
| Original LoRA | Mistral > Qwen > Gemma > LLaMA | Mistral > Qwen > Gemma > LLaMA |
| LoRA r=8 | Mistral > Gemma > Qwen > LLaMA | Mistral > Gemma > Qwen > LLaMA |
| Original Prompt Tuning | Mistral > LLaMA > Qwen > Gemma | Mistral > LLaMA > Gemma > Qwen |
| Original P-Tuning | Mistral > Qwen > Gemma > LLaMA | Mistral > Qwen > Gemma > LLaMA |

## Interpretation for Manuscript Claims

The model-based GPT-4.1 check does not require changing the paper's main
claims. It supports the broad statement that base model choice matters:
Mistral remains the highest-unsafe model under both judges, while LLaMA remains
the lowest-unsafe model in the aggregate and in four of the five settings.

The only judge-sensitive region is the middle of the ranking for some settings,
especially Original Prompt Tuning, where Qwen, Gemma, and LLaMA are much closer
than Mistral. This does not challenge the main claim because the paper does not
rest on a precise ordering among those middle models under this restricted
UltraFeedback SFT subset.

Important caveat: this GPT-4.1 robustness check measures absolute unsafe rates
for selected UltraFeedback SFT responses. It is not the same as the paper's
main base-relative safety-change analysis over the full experimental matrix.
Therefore, it should be used to support judge robustness, not to replace the
main base-model conclusions about Qwen's average gains, Gemma's base-relative
declines, or Mistral's variability.

Recommended wording if this is discussed informally:

```text
At the base-model level, the alternative GPT-4.1 judge preserves the same broad
absolute-safety ordering as LLaMA-Guard 2 in the checked UltraFeedback SFT
responses. Mistral remains the least safe by a wide margin and LLaMA remains
the safest overall. The few differences occur among the middle-ranked models
and do not alter the paper's main conclusion that base model choice strongly
moderates PEFT safety outcomes.
```
