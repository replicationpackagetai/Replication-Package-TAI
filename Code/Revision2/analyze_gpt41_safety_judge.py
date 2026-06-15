#!/usr/bin/env python3
"""Aggregate GPT-4.1 and LLaMA-Guard safety labels for the R2 judge check."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, stdev

import numpy as np
from scipy.stats import pearsonr, spearmanr

PRICE_PER_MILLION = {"input": 2.00, "output": 8.00}

SETTING_LABELS = {
    "original_ia3": "IA3",
    "original_lora": "LoRA",
    "lora_r8": "LoRA (r=8)",
    "original_prompt": "Prompt Tuning",
    "original_ptuning": "P-Tuning",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="Data/Revision2/GPT41_Safety_Judge/manifests/gpt41_five_settings_manifest.csv")
    parser.add_argument("--output_dir", default="Data/Revision2/GPT41_Safety_Judge/recomputed_analysis")
    parser.add_argument("--base_dir", default=".", help="Base directory for relative JSONL paths in the manifest.")
    return parser.parse_args()


def label(value: object) -> str:
    text = str(value or "").strip().lower()
    if text.startswith("unsafe"):
        return "unsafe"
    if text.startswith("safe"):
        return "safe"
    return "unknown"


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def cohen_kappa(pairs: list[tuple[str, str]]) -> float:
    pairs = [(a, b) for a, b in pairs if a in {"safe", "unsafe"} and b in {"safe", "unsafe"}]
    n = len(pairs)
    if not n:
        return math.nan
    observed = sum(a == b for a, b in pairs) / n
    left = Counter(a for a, _ in pairs)
    right = Counter(b for _, b in pairs)
    expected = sum((left[x] / n) * (right[x] / n) for x in ["safe", "unsafe"])
    return (observed - expected) / (1 - expected) if expected != 1 else math.nan


def exact_mcnemar_p(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    lower = sum(math.comb(n, k) for k in range(0, min(b, c) + 1)) / (2 ** n)
    return min(1.0, 2.0 * lower)


def phi_from_mcnemar(b: int, c: int, n_total: int) -> float:
    if n_total == 0 or b + c == 0:
        return 0.0
    chi2 = ((b - c) ** 2) / (b + c)
    return math.sqrt(chi2 / n_total)


def summarize_manifest_row(row: dict, base_dir: Path) -> dict:
    path = base_dir / row["output_jsonl"]
    records = [r for r in read_jsonl(path) if not r.get("error")]
    expected = int(row.get("row_count") or len(records))
    if len(records) != expected:
        raise ValueError(f"{path} has {len(records)} successful rows; expected {expected}")
    pairs = [(label(r.get("llamaguard_label")), label(r.get("gpt_label"))) for r in records]
    counts = Counter(pairs)
    ss = counts[("safe", "safe")]
    su = counts[("safe", "unsafe")]
    us = counts[("unsafe", "safe")]
    uu = counts[("unsafe", "unsafe")]
    n = ss + su + us + uu
    input_tokens = sum(int((r.get("usage") or {}).get("prompt_tokens") or 0) for r in records)
    output_tokens = sum(int((r.get("usage") or {}).get("completion_tokens") or 0) for r in records)
    return {
        "setting": row["setting"],
        "setting_label": SETTING_LABELS.get(row["setting"], row.get("setting_label", row["setting"])),
        "round": row["round"],
        "model": row["model"],
        "n": n,
        "lg_safe_gpt_safe": ss,
        "lg_safe_gpt_unsafe": su,
        "lg_unsafe_gpt_safe": us,
        "lg_unsafe_gpt_unsafe": uu,
        "gpt_unsafe_rate": (su + uu) / n,
        "llamaguard_unsafe_rate": (us + uu) / n,
        "agreement": (ss + uu) / n,
        "cohens_kappa": cohen_kappa(pairs),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "estimated_cost_usd": (input_tokens * PRICE_PER_MILLION["input"] + output_tokens * PRICE_PER_MILLION["output"]) / 1_000_000,
    }


def aggregate(rows: list[dict], keys: list[str]) -> list[dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[k] for k in keys)].append(row)
    out = []
    for key, group in groups.items():
        record = dict(zip(keys, key))
        if "setting" in keys:
            record["setting_label"] = group[0]["setting_label"]
        record["units"] = len(group)
        record["n_total"] = sum(r["n"] for r in group)
        for col in ["gpt_unsafe_rate", "llamaguard_unsafe_rate", "agreement", "cohens_kappa"]:
            vals = [float(r[col]) for r in group]
            record[f"mean_{col}"] = mean(vals)
            record[f"sem_{col}"] = stdev(vals) / math.sqrt(len(vals)) if len(vals) > 1 else 0.0
        record["estimated_cost_usd"] = sum(float(r["estimated_cost_usd"]) for r in group)
        out.append(record)
    return sorted(out, key=lambda r: tuple(str(r[k]) for k in keys))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def agreement_tests(per_file: list[dict]) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in per_file:
        grouped[row["setting"]].append(row)
    rows = []
    for setting, group in grouped.items():
        ss = sum(r["lg_safe_gpt_safe"] for r in group)
        su = sum(r["lg_safe_gpt_unsafe"] for r in group)
        us = sum(r["lg_unsafe_gpt_safe"] for r in group)
        uu = sum(r["lg_unsafe_gpt_unsafe"] for r in group)
        n = ss + su + us + uu
        p = exact_mcnemar_p(su, us)
        rows.append({
            "setting": setting,
            "setting_label": SETTING_LABELS.get(setting, setting),
            "n_total": n,
            "lg_safe_gpt_safe": ss,
            "lg_safe_gpt_unsafe": su,
            "lg_unsafe_gpt_safe": us,
            "lg_unsafe_gpt_unsafe": uu,
            "gpt_unsafe_rate": (su + uu) / n,
            "llamaguard_unsafe_rate": (us + uu) / n,
            "agreement": (ss + uu) / n,
            "cohens_kappa": cohen_kappa([("safe", "safe")] * ss + [("safe", "unsafe")] * su + [("unsafe", "safe")] * us + [("unsafe", "unsafe")] * uu),
            "mcnemar_exact_p": p,
            "mcnemar_phi": phi_from_mcnemar(su, us, n),
            "mcnemar_significant_p_lt_0.05": bool(p < 0.05),
        })
    return sorted(rows, key=lambda r: r["setting"])


def correlations(per_file: list[dict]) -> list[dict]:
    gpt = np.array([r["gpt_unsafe_rate"] for r in per_file], dtype=float)
    lg = np.array([r["llamaguard_unsafe_rate"] for r in per_file], dtype=float)
    pearson = pearsonr(gpt, lg)
    spearman = spearmanr(gpt, lg)
    return [{
        "unit": "setting_round_model",
        "n_units": len(per_file),
        "metric_pair": "gpt_unsafe_rate_vs_llamaguard_unsafe_rate",
        "pearson_r": float(pearson.statistic),
        "pearson_p": float(pearson.pvalue),
        "spearman_rho": float(spearman.statistic),
        "spearman_p": float(spearman.pvalue),
    }]


def main() -> None:
    args = parse_args()
    base_dir = Path(args.base_dir)
    output_dir = Path(args.output_dir)
    with Path(args.manifest).open("r", encoding="utf-8", newline="") as handle:
        manifest = list(csv.DictReader(handle))
    per_file = [summarize_manifest_row(row, base_dir) for row in manifest]
    write_csv(output_dir / "per_model_round.csv", per_file)
    write_csv(output_dir / "by_setting.csv", aggregate(per_file, ["setting"]))
    write_csv(output_dir / "by_setting_model.csv", aggregate(per_file, ["setting", "model"]))
    write_csv(output_dir / "judge_agreement_tests.csv", agreement_tests(per_file))
    write_csv(output_dir / "judge_unsafe_rate_correlations.csv", correlations(per_file))
    print(f"Wrote GPT-4.1 judge analysis to {output_dir}")


if __name__ == "__main__":
    main()
