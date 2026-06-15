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


## Revision 2 Additions

The `Data/Revision2/`, `Code/Revision2/`, and `Figures/Revision2/` directories contain the compact materials added for the second revision.

- `Data/Revision2/PEFT_Hyperparameter_Sensitivity/` contains the averaged PEFT-setting sensitivity summaries and paired statistical-test outputs for LoRA rank, Prompt-Tuning and P-Tuning virtual-token counts, and IA3 target-module placement.
- `Data/Revision2/GPT41_Safety_Judge/` contains the GPT-4.1 alternative safety-judge outputs and aggregate agreement/statistical summaries against LLaMA-Guard 2.
- `Code/Revision2/` contains portable scripts for recomputing the PEFT sensitivity tests and GPT-4.1 judge analysis. These scripts use command-line inputs and environment variables rather than cluster-specific paths.
- `Figures/Revision2/` contains the final figures used for the new revision appendices.

The full Revision 2 benchmark-output archive and sanitized final adapter archive are provided under `Data/Revision2/PEFT_Hyperparameter_Sensitivity/` and are tracked with Git LFS.
