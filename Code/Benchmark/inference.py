#!/usr/bin/env python3
"""
Unified Inference Script for SFT/DPO-tuned Models and Base Models

Consolidates inference routines for LoRA, IA3, Prompt-Tuning and P-Tuning
across multiple evaluation datasets:

    - Safety:   Hex-PHI dataset  (--dataset hex)
    - Fairness: BBQ-Lite dataset (--dataset bbq)
    - Utility:  Test sets        (--dataset test)

Adapters supported via AutoPeftModelForCausalLM:
    - IA3, LoRA, Prompt-Tuning, P-Tuning
If --model_dir points to a base model (no adapter files), a plain
AutoModelForCausalLM is loaded.

Example:
    python inference.py --dataset hex  --model_dir <path> --input_dir <path>
    python inference.py --dataset bbq  --model_dir <path> --input_dir <path>
    python inference.py --dataset test --model_dir <path> --input_dir <path>
"""

import os
import re
import csv
import json
import argparse
import torch
import pandas as pd
from tqdm import tqdm
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import AutoPeftModelForCausalLM


def parse_args():
    """
    Parse command-line arguments for unified PEFT/base inference.
    """
    p = argparse.ArgumentParser(description="Unified inference script for PEFT-tuned or base models.")
    p.add_argument("--model_dir", "-m", type=str, required=True,
                   help="Path to model or PEFT adapter directory.")
    p.add_argument("--input_dir", "-i", type=str, required=True,
                   help="Input directory or dataset path (varies by dataset).")
    p.add_argument("--output_file", "-o", type=str,
                   help="Optional path for output file.")
    p.add_argument("--gemma", "-g", type=int, default=0,
                   help="Set to 1 for Gemma-family chat template.")
    p.add_argument("--dataset", "-d", type=str, required=True,
                   choices=["hex", "bbq", "test"],
                   help="Dataset identifier: hex | bbq | test.")
    p.add_argument("--train_dataset", "-t", type=str, default="ultrachat_200k",
                   help="Training dataset name used for output folder naming.")
    return p.parse_args()


def clean_response(raw_response: str, prompt: str) -> str:
    """
    Remove role prefixes and prompt echoes from model generations.
    """
    parts = raw_response.split("assistant")
    if len(parts) > 2:
        cleaned = "assistant".join(parts[2:]).strip()
    elif len(parts) == 2:
        cleaned = parts[1].strip()
    else:
        cleaned = raw_response.strip()
    cleaned = cleaned.lstrip(":").strip()
    if cleaned.startswith(prompt):
        cleaned = cleaned[len(prompt):].strip()
    return cleaned


def generate_response(model, tokenizer, prompt, gemma=0,
                      max_new_tokens=500, do_sample=True,
                      temperature=0.01, top_p=0.9):
    """
    Generate a response for a prompt using a chat template.
    """
    if gemma:
        messages = [{"role": "user", "content": prompt}]
    else:
        messages = [{"role": "system", "content": ""}, {"role": "user", "content": prompt}]
    input_ids = tokenizer.apply_chat_template(
        messages, truncation=True, add_generation_prompt=True, return_tensors="pt"
    ).to("cuda")

    with torch.inference_mode():
        outputs = model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p
        )
    raw_response = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    return clean_response(raw_response, prompt), raw_response


# ------------------------- Fairness: BBQ -------------------------

def format_bbq_prompt(example):
    """
    Construct a multiple-choice prompt for the BBQ dataset.
    """
    return (
        f"Read the given context, and choose the correct answer to the question "
        f"from options (A), (B), or (C). Respond with a single alphabet.\n"
        f"Context: {example['context']}\n"
        f"Question: {example['question']}\n"
        f"(A): {example['ans0']}\n(B): {example['ans1']}\n(C): {example['ans2']}\n"
        f"Answer:"
    )


def clean_bbq_response(raw_response: str, prompt: str) -> str:
    """
    Extract a valid option label (A/B/C) from model output.
    """
    cleaned = clean_response(raw_response, prompt)
    cleaned = re.sub(r'user\n.*?Answer:\nmodel\n', '', cleaned, flags=re.S)
    m = re.search(r'\b[\(]?(A|B|C)[\)]?\b', cleaned, re.I)
    return m.group(1).upper() if m else "X"


def process_bbq(model, tokenizer, input_dir, output_file, gemma):
    """
    Run inference on BBQ categories and write CSV output.
    """
    categories = [
        "Age", "Disability_status", "Gender_identity", "Nationality",
        "Physical_appearance", "Race_ethnicity", "Religion", "SES", "Sexual_orientation"
    ]
    rows = []
    for cat in categories:
        path = os.path.join(input_dir, f"{cat}.json")
        with open(path, "r") as f:
            data = json.load(f)
        for ex in tqdm(data["examples"], desc=f"Processing {cat}"):
            prompt = format_bbq_prompt(ex)
            _, raw = generate_response(model, tokenizer, prompt, gemma)
            resp = clean_bbq_response(raw, prompt)
            rows.append({
                "category": cat,
                "example_id": ex["example_id"],
                "prompt": prompt,
                "raw_response": raw,
                "response": resp
            })
    pd.DataFrame(rows).to_csv(output_file, index=False)


# -------------------------- Safety: Hex --------------------------

def process_hex(model, tokenizer, input_dir, output_file, gemma):
    """
    Process Hex-PHI CSV files and write CSV output.
    """
    rows = []
    for i in range(1, 12):
        fp = os.path.join(input_dir, f"category_{i}.csv")
        with open(fp, "r") as f:
            reader = csv.reader(f)
            prompts = [row[0] for row in reader]
        for prompt in tqdm(prompts, desc=f"Processing category_{i}.csv"):
            resp, _ = generate_response(model, tokenizer, prompt, gemma)
            rows.append([prompt, resp])
    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["prompt", "response"])
        writer.writerows(rows)


# -------------------------- Utility: Test ------------------------

def process_test(model, tokenizer, dataset_dir, output_file, gemma):
    """
    Load test dataset from disk, run inference on a subset, and save results.
    """
    ds = load_from_disk(dataset_dir)
    subset = ds.select(range(0, 100))
    rows = []
    for ex in tqdm(subset, desc="Processing test dataset"):
        prompt = ex["prompt"]
        resp, _ = generate_response(model, tokenizer, prompt, gemma)
        rows.append({"example_id": ex["prompt_id"], "prompt": prompt, "response": resp})
    pd.DataFrame(rows).to_csv(output_file, index=False)


# ---------------------------- Main -------------------------------

def is_peft_adapter(path: str) -> bool:
    """
    Heuristic to detect a PEFT adapter directory.
    """
    names = ("adapter_config.json", "adapter_model.bin", "adapter_model.safetensors")
    return any(os.path.exists(os.path.join(path, n)) for n in names)


def main():
    """
    Load tokenizer and model (PEFT adapter or base), dispatch dataset processing,
    write outputs to a consistent directory structure.
    """
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_path = args.model_dir
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token

    if is_peft_adapter(model_path):
        model = AutoPeftModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        )

    model = model.to(device)
    model.eval()

    out_root = f"IA3_{args.dataset}_responses/{args.train_dataset}/"
    os.makedirs(out_root, exist_ok=True)

    if not args.output_file:
        name = args.model_dir.rstrip("/").split("/")[-1]
        args.output_file = os.path.join(out_root, f"{name}_{args.dataset}.csv")

    if args.dataset == "hex":
        process_hex(model, tokenizer, args.input_dir, args.output_file, args.gemma)
    elif args.dataset == "bbq":
        process_bbq(model, tokenizer, args.input_dir, args.output_file, args.gemma)
    elif args.dataset == "test":
        process_test(model, tokenizer, args.input_dir, args.output_file, args.gemma)
    else:
        raise SystemExit("Unsupported dataset type.")

    print(f"Inference complete. Results saved to {args.output_file}")


if __name__ == "__main__":
    main()
