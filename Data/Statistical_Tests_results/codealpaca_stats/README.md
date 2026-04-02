# CodeAlpaca summaries and task statistics

This directory contains the corrected CodeAlpaca summary tables and statistical tests.


## Validation
- Each Round1/2/3 summary contains the same 32 CodeAlpaca configurations for both `hex_safe_overall_summary.csv` and `bbq_overall_summary.csv`.
- The corrected mean table is computed directly from the 96 per-round rows.
- Each averaged configuration must appear exactly 3 times, once in each round.
- The pairing key for task comparison is:
  - `model_name`
  - `peft_method`
  - `learning_rate`
  - `epochs`

## Output files
- `codealpaca_bias_safety_abs_diff_rounds.csv`: 96 rows (32 configurations x 3 rounds)
- `codealpaca_bias_safety_abs_diff_mean.csv`: 32 rows (means over the same configuration across Round1/2/3)
- `codealpaca_task_stat_tests.csv`: corrected Wilcoxon signed-rank test results
- `codealpaca_task_stat_tests_report.txt`: compact textual summary

## Metric convention
- Safety diffs are computed as:
  - `safe_ratio * 100 - base_model_safety_percentage`
- Fairness diffs are computed against the matching base model.
- Bias differences use absolute bias magnitudes before subtraction.
