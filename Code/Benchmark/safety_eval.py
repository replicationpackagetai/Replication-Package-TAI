#!/usr/bin/env python3
"""
Llama Guard 2 Safety Assessment

Evaluates model responses for safety using a moderation model
(default: meta-llama/Meta-Llama-Guard-2-8B). It expects CSV files that
contain two columns:
    - prompt
    - response

For each row, the script forms a chat with the user's prompt and the
assistant's response, runs the moderation model, and appends:
    - Moderation_Result
    - Hazard_Types  

Example Usage:
    python moderate_llamaguard.py \
        --files path/to/outputs1.csv path/to/outputs2.csv \
        --model_id meta-llama/Meta-Llama-Guard-2-8B
"""

import os
import json
import argparse
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def parse_args():
    """
    Parse command-line arguments for Llama Guard 2 moderation.
    """
    p = argparse.ArgumentParser(description="Moderate CSV files with Llama Guard 2.")
    p.add_argument("--files", "-f", nargs="+", required=True,
                   help="Paths to input CSV files containing 'prompt' and 'response' columns.")
    p.add_argument("--model_id", "-m", type=str, default="meta-llama/Meta-Llama-Guard-2-8B",
                   help="HF model id for the moderation model.")
    p.add_argument("--token", "-t", type=str, default=None,
                   help="HF token (optional). If not provided, will try config or env.")
    p.add_argument("--config", "-c", type=str, default=None,
                   help="Optional JSON config file containing {'hf_token': '...'}")
    p.add_argument("--local_files_only", action="store_true",
                   help="Load model/tokenizer only from local cache.")
    p.add_argument("--max_new_tokens", type=int, default=100,
                   help="Maximum new tokens for moderation generation.")
    return p.parse_args()


def load_token(cli_token: str | None, cfg_path: str | None) -> str | None:
    """
    Resolve Hugging Face token from CLI, config file, or environment.
    """
    if cli_token:
        return cli_token
    if cfg_path and os.path.exists(cfg_path):
        with open(cfg_path, "r") as f:
            cfg = json.load(f) or {}
        if "hf_token" in cfg:
            return cfg["hf_token"]
    return os.environ.get("HF_TOKEN", None)


def build_tokenizer(model_id: str, token: str | None, local_only: bool):
    """
    Build tokenizer with safe defaults for padding.
    """
    tok = AutoTokenizer.from_pretrained(
        model_id,
        token=token,
        use_fast=True,
        local_files_only=local_only
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token or tok.unk_token
    return tok


def build_model(model_id: str, token: str | None, local_only: bool):
    """
    Load the moderation model with appropriate dtype and device map.
    """
    torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    device_map = "auto" if torch.cuda.is_available() else None
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        token=token,
        local_files_only=local_only,
        torch_dtype=torch_dtype,
        device_map=device_map,
    )
    model.eval()
    return model


def moderate(chat, tokenizer, model, max_new_tokens: int, device: str):
    """
    Run moderation on a single chat transcript.
    """
    input_ids = tokenizer.apply_chat_template(
        chat,
        return_tensors="pt",
        add_generation_prompt=True
    ).to(device)

    with torch.inference_mode():
        outputs = model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
        )

    # Skip the prompt portion in the generated sequence
    prompt_len = input_ids.shape[-1]
    text = tokenizer.decode(outputs[0][prompt_len:], skip_special_tokens=True)
    return text.strip()


def extract_hazards(moderation_text: str) -> str:
    """
    Best-effort hazard extraction from the moderation output.
    Keeps it simple to avoid overfitting to any specific template.
    """
    if not moderation_text or not isinstance(moderation_text, str):
        return ""
    text = moderation_text.lower()
    if "unsafe" not in text:
        return ""
    # Heuristic: return the second line if present, else empty
    parts = [ln.strip() for ln in moderation_text.splitlines() if ln.strip()]
    return parts[1] if len(parts) > 1 else ""


def process_file(file_path: str, tokenizer, model, device: str, max_new_tokens: int):
    """
    Process a single CSV file and write a moderated output CSV.
    """
    df = pd.read_csv(file_path)
    if "prompt" not in df.columns or "response" not in df.columns:
        raise ValueError("CSV must contain 'prompt' and 'response' columns.")

    results = []
    for _, row in df.iterrows():
        chat = [
            {"role": "user", "content": row["prompt"]},
            {"role": "assistant", "content": row["response"]},
        ]
        try:
            mod = moderate(chat, tokenizer, model, max_new_tokens=max_new_tokens, device=device)
        except Exception as e:
            print(f"Moderation error on file {file_path}: {e}")
            mod = ""
        results.append(mod)

    df["Moderation_Result"] = results
    df["Hazard_Types"] = [extract_hazards(x) for x in results]

    out_path = file_path.replace(".csv", "_moderated.csv")
    df.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")


def main():
    """
    Entry point: load model/tokenizer once, then moderate each file.
    """
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    token = load_token(args.token, args.config)
    tokenizer = build_tokenizer(args.model_id, token, args.local_files_only)
    model = build_model(args.model_id, token, args.local_files_only)

    for fp in args.files:
        try:
            process_file(fp, tokenizer, model, device=device, max_new_tokens=args.max_new_tokens)
        except Exception as e:
            print(f"Could not process {fp}: {e}")


if __name__ == "__main__":
    main()
