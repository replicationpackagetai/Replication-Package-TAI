# Replication Package for Anonymous TAI Submission

This repository contains the replication materials for the anonymous submission "Efficiency at What Cost? Safety and Fairness in Parameter-Efficient Fine-Tuning of LLMs".

## Repository Structure

- `Appendices/`
  - `5-Appendices.pdf`: compiled appendix file submitted as supplementary material.

- `Code/`
  - `Fine-Tuning/`: scripts for PEFT fine-tuning used in the study.
  - `Benchmark/`: scripts for inference, safety evaluation, fairness evaluation, utility evaluation, and statistical testing.

- `Data/`
  - `BBQ-Lite_Modified/`: corrected version of BBQ-Lite used in the fairness evaluation.
  - `Models_Reponses/`: benchmark outputs for the main conversational experiments.
  - `Fine-Tuned_Models/`: archived fine-tuned model artifacts and adapter-related files.
  - `HuggingFace_Models/`: model metadata and adapter configuration files collected for the study.
  - `Utility_test_datasets/`: test subsets used for utility evaluation.
  - `CodeAlpaca/`: benchmark outputs, moderation outputs, archived fine-tuned artifacts, and per-round summaries for the benign coding-task experiments.
  - `Statistical_Tests_results/`: statistical outputs for both the main study and the CodeAlpaca experiments, including averaged summaries across rounds.
