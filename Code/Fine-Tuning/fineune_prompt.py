#!/usr/bin/env python3
"""
Unified Prompt Tuning Training Script for SFT and DPO

- Uses PromptTuningConfig from PEFT.
- Accepts unified config JSON/YAML with blocks: prompt_tuning, lora, ia3, p_tuning, trainer, etc.

Dataset expectations:
- UltraFeedback_binarized or UltraChat_200k loaded via datasets.load_from_disk as a DatasetDict
- SFT splits: train_sft / test_sft with "messages"
- DPO splits: train_prefs / test_prefs with "chosen"/"rejected" (lists of messages or {"messages": [...]})
"""

import os
import json
import yaml
import random
import logging
import argparse
from typing import Optional, Dict, Any

import torch
from datasets import load_from_disk, DatasetDict
from transformers import AutoTokenizer, TrainingArguments
from peft import PromptTuningConfig
from trl import SFTTrainer, DPOTrainer, DPOConfig

LOG = logging.getLogger("train_prompt")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)


def set_seed_opt(seed: Optional[int] = None):
    if seed is None:
        return
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

def load_config(path: str) -> Dict[str, Any]:
    if path.endswith((".yaml", ".yml")):
        with open(path, "r") as f:
            return yaml.safe_load(f) or {}
    elif path.endswith(".json"):
        with open(path, "r") as f:
            return json.load(f) or {}
    else:
        raise SystemExit("Config must be .yaml/.yml or .json")

def deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out.get(k, {}), v)
        else:
            out[k] = v
    return out

def infer_default_train_pct(data_dir: str, task: str) -> float | None:
    """
    If no train_percentage is provided, use your defaults based on dataset+task:
      - SFT: ultrafeedback = 34%, ultrachat = 10%
      - DPO: ultrafeedback = 34%
    Otherwise return None (caller may skip sampling).
    """
    name = (data_dir or "").lower()
    is_ufb = "ultrafeedback" in name
    is_uc  = "ultrachat" in name
    if "ultrafeedback" in name: 
        return 34.0
    elif "ultrachat" in name:  
        return 10.0
    return None

def sample_split(raw: DatasetDict, split: str, percentage: float, seed: int = 42) -> DatasetDict:
    """
    Replace `split` with a random subset of size percentage% (0 < p <= 100).
    Other splits remain unchanged.
    """
    if not isinstance(raw, DatasetDict):
        raise ValueError("Expected a DatasetDict.")
    if split not in raw:
        raise KeyError(f"Split '{split}' not found in dataset.")
    if not (0 < percentage <= 100):
        raise ValueError("percentage must be in (0, 100].")

    random.seed(seed)
    n = len(raw[split])
    k = max(1, int((percentage / 100.0) * n))
    idx = random.sample(range(n), k)

    out = {}
    for ksplit, dset in raw.items():
        out[ksplit] = dset.select(idx) if ksplit == split else dset
    return DatasetDict(out)


def apply_chat_template_to_messages(messages, tokenizer: AutoTokenizer, gemma_safe: bool):
    msgs = messages
    if not gemma_safe:
        if len(msgs) == 0 or msgs[0].get("role") != "system":
            msgs = [{"role": "system", "content": ""}] + msgs
    return tokenizer.apply_chat_template(msgs, tokenize=False)

def map_for_sft(example, tokenizer: AutoTokenizer, gemma_safe: bool):
    example["text"] = apply_chat_template_to_messages(example["messages"], tokenizer, gemma_safe)
    return example

def map_for_dpo(example, tokenizer: AutoTokenizer, gemma_safe: bool):
    def _msgs(x):
        if isinstance(x, dict) and 'messages' in x:
            return x['messages']
        return x  # assume list
    chosen_msgs = _msgs(example["chosen"])
    rejected_msgs = _msgs(example["rejected"])
    example["chosen"] = apply_chat_template_to_messages(chosen_msgs, tokenizer, gemma_safe)
    example["rejected"] = apply_chat_template_to_messages(rejected_msgs, tokenizer, gemma_safe)
    return example


def build_tokenizer(model_name: str, cfg: Dict[str, Any]) -> AutoTokenizer:
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token or tok.unk_token
    if "padding_side" in cfg:
        tok.padding_side = cfg["padding_side"]
    return tok

def build_prompt_tuning_config(cfg: Dict[str, Any]) -> PromptTuningConfig:
    return PromptTuningConfig(**cfg)

def filter_kwargs(allowed_keys, user_dict):
    return {k: user_dict[k] for k in allowed_keys if k in user_dict}

def allowed_training_args():
    return {
        "output_dir","overwrite_output_dir","do_eval","evaluation_strategy",
        "per_device_train_batch_size","per_device_eval_batch_size",
        "gradient_accumulation_steps","gradient_checkpointing",
        "gradient_checkpointing_kwargs",
        "learning_rate","lr_scheduler_type","max_steps","num_train_epochs",
        "logging_steps","log_level","logging_strategy",
        "save_steps","save_total_limit","save_strategy","seed",
        "report_to"
    }

def allowed_dpo_args():
    return allowed_training_args().union({"beta","max_length","max_prompt_length"})

def require_keys(cfg: Dict[str, Any], keys):
    missing = [k for k in keys if k not in cfg or cfg[k] is None]
    if missing:
        raise SystemExit(f"Missing required config keys: {missing}")

# ----------------------------
# Runners
# ----------------------------

def run_sft(cfg: Dict[str, Any]):
    require_keys(cfg, ["model_name","data_dir","output_dir","train_split"])
    set_seed_opt(cfg.get("seed"))

    tokenizer = build_tokenizer(cfg["model_name"], cfg)

    raw = load_from_disk(cfg["data_dir"])

    # sampling (applies to train split only)
    train_split = cfg["train_split"]
    eval_split  = cfg.get("eval_split", None)
    seed        = cfg.get("seed", 42)
    pct         = cfg.get("train_percentage")
    if pct is None:
        pct = infer_default_train_pct(cfg.get("data_dir",""), task="sft")
    if pct:
        raw = sample_split(raw, split=train_split, percentage=pct, seed=seed)

    ds = raw.map(lambda ex: map_for_sft(ex, tokenizer, cfg.get("gemma_safe", False)),
                 num_proc=cfg.get("num_proc", None),
                 desc="Applying chat template (SFT)")

    trainer_kwargs = filter_kwargs(allowed_training_args(), cfg.get("trainer", {}))
    trainer_kwargs["output_dir"] = cfg["output_dir"]
    args = TrainingArguments(**trainer_kwargs)

    peft_conf = build_prompt_tuning_config(cfg.get("prompt_tuning", {}))

    trainer = SFTTrainer(
        model=cfg["model_name"],
        model_init_kwargs=dict(
            torch_dtype="auto",
            use_cache=False,
            device_map=cfg.get("device_map", "auto")
        ),
        args=args,
        train_dataset=ds[train_split],
        eval_dataset=ds[eval_split] if eval_split and eval_split in ds else None,
        dataset_text_field="text",
        tokenizer=tokenizer,
        packing=True,
        peft_config=peft_conf,
        max_seq_length=tokenizer.model_max_length,
    )

    LOG.info("Starting SFT (Prompt Tuning) ...")
    result = trainer.train()
    trainer.save_state()
    trainer.save_model(cfg["output_dir"])
    trainer.log_metrics("train", result.metrics)
    trainer.save_metrics("train", result.metrics)
    LOG.info("SFT complete. Saved to %s", cfg["output_dir"])

def run_dpo(cfg: Dict[str, Any]):
    require_keys(cfg, ["model_name","data_dir","output_dir","train_split"])
    set_seed_opt(cfg.get("seed"))

    tokenizer = build_tokenizer(cfg["model_name"], cfg)

    raw = load_from_disk(cfg["data_dir"])

    # sampling (applies to train split only)
    train_split = cfg["train_split"]
    eval_split  = cfg.get("eval_split", None)
    seed        = cfg.get("seed", 42)
    pct         = cfg.get("train_percentage")
    if pct is None:
        pct = infer_default_train_pct(cfg.get("data_dir",""), task="dpo")
    if pct:
        raw = sample_split(raw, split=train_split, percentage=pct, seed=seed)

    ds = raw.map(lambda ex: map_for_dpo(ex, tokenizer, cfg.get("gemma_safe", False)),
                 num_proc=cfg.get("num_proc", None),
                 desc="Applying chat template (DPO)")

    dpo_kwargs = filter_kwargs(allowed_dpo_args(), cfg.get("trainer", {}))
    dpo_kwargs["output_dir"] = cfg["output_dir"]
    args = DPOConfig(**dpo_kwargs)

    peft_conf = build_prompt_tuning_config(cfg.get("prompt_tuning", {}))

    trainer = DPOTrainer(
        model=cfg["model_name"],
        model_init_kwargs=dict(
            torch_dtype="bfloat16",
            use_cache=False,
            local_files_only=cfg.get("local_files_only", True),
            device_map=cfg.get("device_map", "auto")
        ),
        ref_model=None,
        args=args,
        train_dataset=ds[train_split],
        eval_dataset=ds[eval_split] if eval_split and eval_split in ds else None,
        peft_config=peft_conf,
        tokenizer=tokenizer
    )

    LOG.info("Starting DPO (Prompt Tuning) ...")
    result = trainer.train()
    trainer.save_state()
    trainer.save_model(cfg["output_dir"])
    trainer.log_metrics("train", result.metrics)
    trainer.save_metrics("train", result.metrics)
    LOG.info("DPO complete. Saved to %s", cfg["output_dir"])


def parse_cli():
    p = argparse.ArgumentParser(description="Unified Prompt Tuning trainer for SFT or DPO.")
    p.add_argument("--task", choices=["sft", "dpo"], required=True, help="Training paradigm.")
    p.add_argument("--config", type=str, required=True, help="YAML or JSON config file path.")
    p.add_argument("--model_name", type=str, help="HF model ID (overrides config).")
    p.add_argument("--data_dir", type=str, help="Path to load_from_disk dataset.")
    p.add_argument("--output_dir", type=str, help="Where to save model/artifacts.")
    p.add_argument("--seed", type=int, help="Random seed.")
    p.add_argument("--train_split", type=str, help="Training split name.")
    p.add_argument("--eval_split", type=str, help="Evaluation split name (optional).")
    p.add_argument("--train_percentage", type=float, help="Percentage of the training split to sample (0< p <=100).")
    p.add_argument("--device_map", type=str, help="Device map (e.g. 'auto').")
    p.add_argument("--local_files_only", type=str, help="For DPO model_init_kwargs; 'true'/'false'.")
    return p.parse_args()

def _to_bool(v):
    s = str(v).strip().lower()
    if s in ("true","1","yes","y"): return True
    if s in ("false","0","no","n"): return False
    return v

def main():
    args = parse_cli()
    file_cfg = load_config(args.config)
    cli_cfg = {k: v for k, v in vars(args).items() if v is not None}
    cfg = deep_update(file_cfg, cli_cfg)
    if isinstance(cfg.get("local_files_only"), str):
        cfg["local_files_only"] = _to_bool(cfg["local_files_only"])

    if args.task == "sft":
        run_sft(cfg)
    else:
        run_dpo(cfg)

if __name__ == "__main__":
    main()
