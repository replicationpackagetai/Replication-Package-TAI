# McNemar Interpretation for GPT-4.1 vs LLaMA-Guard 2

McNemar's test is used because GPT-4.1 and LLaMA-Guard 2 judge the same
prompt-response pairs. The safe/unsafe outcomes are therefore paired binary
decisions, not two independent samples.

For each response, the relevant 2x2 table is:

| | GPT unsafe | GPT safe |
|---|---:|---:|
| LLaMA-Guard unsafe | both unsafe | LLaMA-Guard unsafe, GPT safe |
| LLaMA-Guard safe | LLaMA-Guard safe, GPT unsafe | both safe |

The test uses only the discordant cells:

- \(b\): LLaMA-Guard safe, GPT unsafe
- \(c\): LLaMA-Guard unsafe, GPT safe

The null hypothesis is \(b=c\): the two judges may disagree, but the
disagreements are symmetric. A small p-value means the disagreements have a
systematic direction. It does not mean the judges have low agreement, and it
does not mean the setting-level conclusions fail. It should be interpreted
together with agreement, Cohen's kappa, and the percentage-point difference in
unsafe rates.

For effect size, we also compute the common McNemar/Pearson \(\phi\) value from
the uncorrected McNemar chi-square statistic:

\[
\chi^2 = \frac{(b-c)^2}{b+c}, \qquad
\phi = \sqrt{\frac{\chi^2}{N}},
\]

where \(N\) is the total number of paired judgments in the row. We also report
the matched-pair odds ratio \(b/c\), which gives the direction of the
discordance imbalance.

## Aggregate Result

Across all five checked settings:

| Judgments | GPT unsafe | LG unsafe | Diff GPT-LG | Agreement | Discordant | b | c | Phi | OR b/c | McNemar p | Direction |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 19,800 | 22.4% | 21.4% | +1.0 pp | 89.9% | 10.1% | 1,094 | 905 | 0.030 | 1.21 | 2.58e-05 | GPT more unsafe |

The p-value is small because the sample is large and the discordance is
directional: GPT marks 189 more responses unsafe than LLaMA-Guard in the
discordant cells. The effect size is small: the overall unsafe-rate difference
is +1.0 percentage point, and the \(\phi\) effect size is very small
(\(\phi=0.030\)).

## By Setting

| Setting | GPT unsafe | LG unsafe | Diff GPT-LG | Agreement | Discordant | b | c | Phi | OR b/c | McNemar p | Direction |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| IA3 | 21.1% | 17.8% | +3.3 pp | 90.0% | 10.0% | 262 | 133 | 0.103 | 1.97 | 8.16e-11 | GPT more unsafe |
| LoRA | 15.8% | 13.8% | +1.9 pp | 91.8% | 8.2% | 201 | 124 | 0.068 | 1.62 | 2.29e-05 | GPT more unsafe |
| LoRA (r=8) | 14.1% | 12.6% | +1.5 pp | 93.1% | 6.9% | 167 | 107 | 0.058 | 1.56 | 0.000347 | GPT more unsafe |
| Prompt Tuning | 28.5% | 28.8% | -0.3 pp | 86.0% | 14.0% | 271 | 283 | 0.008 | 0.96 | 0.640 | LLaMA-Guard more unsafe |
| P-Tuning | 32.5% | 34.1% | -1.6 pp | 88.6% | 11.4% | 193 | 258 | 0.049 | 0.75 | 0.00254 | LLaMA-Guard more unsafe |

## By Model

| Model | GPT unsafe | LG unsafe | Diff GPT-LG | Agreement | Discordant | b | c | Phi | OR b/c | McNemar p | Direction |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Mistral | 48.7% | 43.2% | +5.5 pp | 82.7% | 17.2% | 563 | 290 | 0.133 | 1.94 | 5.77e-21 | GPT more unsafe |
| LLaMA | 7.8% | 9.7% | -1.9 pp | 93.7% | 6.3% | 109 | 201 | 0.074 | 0.54 | 1.94e-07 | LLaMA-Guard more unsafe |
| Gemma | 13.9% | 15.4% | -1.4 pp | 93.4% | 6.6% | 129 | 200 | 0.056 | 0.65 | 0.000107 | LLaMA-Guard more unsafe |
| Qwen | 19.1% | 17.5% | +1.6 pp | 89.8% | 10.2% | 293 | 214 | 0.050 | 1.37 | 0.00052 | GPT more unsafe |

## Interpretation

The small p-values do not indicate that GPT-4.1 and LLaMA-Guard 2 produce
completely different distributions. They indicate that, for some settings, one
judge is consistently stricter in one direction:

- GPT-4.1 is stricter for the adapter-based settings.
- LLaMA-Guard 2 is slightly stricter for P-Tuning.
- Prompt Tuning has no significant marginal difference between judges.

This does not change the paper's claims. The percentage-point differences are
limited, the conventional \(\phi\) effect sizes are small, agreement remains
high, and the setting-level unsafe rates preserve the broad PEFT-family
ordering: LoRA is lowest, IA3 is intermediate, and prompt-based methods are less
safe.

Useful manuscript-style wording:

```text
Exact McNemar tests show that some judge disagreements are directionally
asymmetric, especially because the paired sample contains 19,800 judgments.
However, these significant marginal differences are small in absolute
unsafe-rate terms and do not alter the setting-level ordering: GPT-4.1 and
LLaMA-Guard 2 remain strongly aligned in unsafe rates, and both preserve the
broad distinction between adapter-based and prompt-based PEFT.
```
