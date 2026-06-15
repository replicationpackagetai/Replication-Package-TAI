# Revision 2 PEFT Hyperparameter Sensitivity Data

This directory contains the compact data and statistical outputs for the targeted PEFT hyperparameter sensitivity analysis reported in the revised appendices.

Scope:
- Dataset/strategy: UltraFeedback SFT.
- Learning rate and epochs: 2e-5, 5 epochs.
- Base models: Qwen2.5-7B-Instruct, Meta-Llama-3-8B-Instruct, Mistral-7B-Instruct-v0.3, and gemma-7b-it.
- Repeated runs: Round1, Round2, and Round3, all using seed 42. The CSV files in this directory report values after averaging over the three repeated runs when applicable.

Files:
- `numeric_sensitivity_delta_by_model.csv`: model-level changes compared with the corresponding base model for LoRA, Prompt-Tuning, and P-Tuning settings.
- `numeric_sensitivity_delta_by_peft_setting.csv`: averages over the four base models for the same settings.
- `grouped_peft_setting_boxplot_data.csv`: data used for the grouped appendix boxplots.
- `paired_setting_tests/`: Friedman and Wilcoxon signed-rank test outputs for LoRA rank and virtual-token-count comparisons.
- `ia3_target_sensitivity/`: summary and paired tests for IA3 target-module placement.

The corresponding portable analysis script is `Code/Revision2/peft_sensitivity_tests.py`.
