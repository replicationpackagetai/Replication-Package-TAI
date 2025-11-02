#!/usr/bin/env python3
"""
Utility Evaluation via LLM-as-Judge

Scores model responses using an external judge model
(default: gpt-4o-2024-08-06) and appends the results to CSV files. Each input CSV
must contain two columns:
    - prompt
    - response

For each row, the judge produces:
    - judgment  (free-form explanation + rating line)
    - rating    (extracted integer from "Rating: [[N]]")

Caching is used to avoid re-judging identical (prompt, response) pairs.
Optionally, the cache can be prefilled from an existing directory of
judged CSVs (with *_judged.csv suffix).

Example:
    python evaluate_utility.py \
      --input_dir test_results/round3_dpo \
      --cache_file judgment_cache.json \
      --model gpt-4o
"""

import os
import re
import json
import time
import argparse
import hashlib
from pathlib import Path

import pandas as pd
from tqdm import tqdm

# OpenAI client (expects OPENAI_API_KEY in the environment by default)
from openai import OpenAI


def parse_args():
    """
    Parse command-line arguments for utility evaluation.
    """
    p = argparse.ArgumentParser(description="Evaluate utility with an LLM judge.")
    p.add_argument("--input_dir", "-i", type=str, required=True,
                   help="Directory containing CSV files to judge.")
    p.add_argument("--model", "-m", type=str, default="gpt-4o-2024-08-06",
                   help="Judge model id (OpenAI).")
    p.add_argument("--cache_file", "-c", type=str, default="judgment_cache.json",
                   help="Path to the JSON cache file.")
    p.add_argument("--prefill_base_dir", type=str, default=None,
                   help="Optional directory of pre-judged CSVs (prefills cache).")
    p.add_argument("--api_key", type=str, default=None,
                   help="Optional API key (overrides OPENAI_API_KEY).")
    return p.parse_args()


def normalize_text(text: str) -> str:
    """
    Normalize text for stable hashing: strip, lowercase, collapse whitespace.
    """
    if not isinstance(text, str):
        return ""
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def pair_hash(prompt: str, response: str) -> str:
    """
    Deterministic hash for a (prompt, response) pair used in caching.
    """
    combined = normalize_text(prompt) + normalize_text(response)
    return hashlib.sha256(combined.encode("utf-8")).hexdigest()


def extract_rating(text: str) -> str | None:
    """
    Extract rating from judge output. Expects a line like:  Rating: [[N]]
    """
    if not isinstance(text, str):
        return None
    m = re.search(r"Rating[:\s\*]*\[\[\s*(\d+)\s*\]\]", text, re.IGNORECASE)
    return m.group(1) if m else None


def load_cache(cache_path: str) -> dict:
    """
    Load judgment cache from disk, or return an empty dict.
    """
    if os.path.exists(cache_path):
        with open(cache_path, "r") as f:
            return json.load(f)
    return {}


def save_cache(cache_path: str, cache: dict):
    """
    Persist judgment cache to disk.
    """
    with open(cache_path, "w") as f:
        json.dump(cache, f, indent=2)


def prefill_cache_from_dir(cache: dict, base_dir: str) -> None:
    """
    Prefill cache entries from existing judged CSVs in base_dir.
    Expects columns: prompt, response, judgment, rating.
    """
    paths = Path(base_dir).rglob("*_judged.csv")
    count = 0
    for path in paths:
        df = pd.read_csv(path)
        if not {"prompt", "response", "judgment", "rating"}.issubset(df.columns):
            continue
        for _, row in df.iterrows():
            h = pair_hash(row.get("prompt", ""), row.get("response", ""))
            if h and h not in cache:
                cache[h] = {
                    "judgment": row.get("judgment", ""),
                    "rating": str(row.get("rating")) if pd.notna(row.get("rating")) else None,
                }
                count += 1
    print(f"Prefill: added {count} cached judgments from {base_dir}.")


def judge_once(client: OpenAI, model: str, prompt: str, response: str) -> str:
    """
    Query the judge model once and return the raw textual judgment.
    """
    content = (
        "Please act as an impartial judge and evaluate the quality of the response "
        "provided by an AI assistant to the user question displayed below. Your "
        "evaluation should consider factors such as helpfulness, relevance, accuracy, "
        "depth, and level of detail. Begin with a short explanation. Afterward, you "
        'must rate the response on a scale of 1 to 10 using the exact format:\n'
        '"Rating: [[N]]"\n\n'
        "[Question]\n"
        f"{prompt}\n\n"
        "[The Start of Assistant's Answer]\n"
        f"{response}\n"
        "[The End of Assistant's Answer]"
    )
    out = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful, impartial evaluator."},
            {"role": "user", "content": content},
        ],
    )
    return out.choices[0].message.content


def get_judgment(client: OpenAI, model: str, cache: dict, prompt: str, response: str) -> tuple[str, str | None, bool]:
    """
    Return (judgment_text, rating_str, from_cache). Uses and updates cache.
    """
    h = pair_hash(prompt, response)
    if h in cache:
        entry = cache[h]
        return entry.get("judgment", ""), entry.get("rating"), True

    text = judge_once(client, model, prompt, response)
    rating = extract_rating(text)
    cache[h] = {"judgment": text, "rating": rating}
    return text, rating, False


def list_input_csvs(root_dir: str) -> list[Path]:
    """
    List all CSV files under root_dir excluding *_judged.csv outputs.
    """
    return [p for p in Path(root_dir).rglob("*.csv") if not str(p).endswith("_judged.csv")]


def process_directory(input_dir: str, model: str, cache_path: str, prefill_dir: str | None, api_key: str | None):
    """
    Evaluate all CSVs in input_dir with the specified judge model.
    Writes *_judged.csv files alongside the originals and updates cache.
    """
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key

    cache = load_cache(cache_path)
    if prefill_dir:
        prefill_cache_from_dir(cache, prefill_dir)
        save_cache(cache_path, cache)

    client = OpenAI()
    files = list_input_csvs(input_dir)
    print(f"Found {len(files)} CSV file(s) to judge under {input_dir}.")

    for path in files:
        path = Path(path)
        out_path = path.with_name(path.stem + "_judged.csv")
        if out_path.exists():
            print(f"Skipping existing output: {out_path}")
            continue

        df = pd.read_csv(path)
        if not {"prompt", "response"}.issubset(df.columns):
            print(f"Skipping {path} (missing 'prompt' or 'response').")
            continue

        judgments, ratings = [], []
        t0 = time.time()
        cached_rows = []

        for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Judging {path.name}"):
            jtxt, rnum, from_cache = get_judgment(client, model, cache, row["prompt"], row["response"])
            judgments.append(jtxt)
            ratings.append(rnum)
            if from_cache:
                cached_rows.append(idx)

        df["judgment"] = judgments
        df["rating"] = ratings
        df.to_csv(out_path, index=False)
        save_cache(cache_path, cache)

        dt = time.time() - t0
        print(f"Wrote {out_path} in {dt:.2f}s; reused {len(cached_rows)} cached row(s).")


def main():
    """
    Entry point: configure judge and process the input directory.
    """
    args = parse_args()
    process_directory(
        input_dir=args.input_dir,
        model=args.model,
        cache_path=args.cache_file,
        prefill_dir=args.prefill_base_dir,
        api_key=args.api_key,
    )


if __name__ == "__main__":
    main()
