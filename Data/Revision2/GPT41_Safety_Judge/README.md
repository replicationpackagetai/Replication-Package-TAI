# Revision 2 GPT-4.1 Safety Judge Robustness Data

This directory contains the compact data and outputs for the GPT-4.1 alternative safety-judge robustness check.

Scope:
- Dataset: HEx-PHI responses from the UltraFeedback SFT setting at learning rate 2e-5 and 5 epochs.
- Settings: IA3, LoRA, LoRA (r=8), Prompt-Tuning, and P-Tuning.
- Base models: Qwen2.5-7B-Instruct, Meta-Llama-3-8B-Instruct, Mistral-7B-Instruct-v0.3, and gemma-7b-it.
- Repeated runs: Round1, Round2, and Round3.
- GPT judge model used in the reported run: `gpt-4.1-2025-04-14`.

Files:
- `manifests/gpt41_five_settings_manifest.csv`: the judged setting/model/round files and row counts. The original LLaMA-Guard-labeled source files are listed through `source_url`; the corresponding GPT outputs are listed in `output_jsonl`.
- `results/`: row-level GPT-4.1 judgment JSONL files.
- `analysis/`: setting-level summaries, agreement statistics, McNemar tests, effect-size tables, correlations, and disagreement examples.

The corresponding portable scripts are:
- `Code/Revision2/gpt_safety_judge.py` for producing GPT judgments from any CSV with `prompt` and `response` columns.
- `Code/Revision2/analyze_gpt41_safety_judge.py` for recomputing the aggregate agreement tables from the manifest and JSONL outputs.

No API key is included. The judge script reads `OPENAI_API_KEY` from the environment by default.
