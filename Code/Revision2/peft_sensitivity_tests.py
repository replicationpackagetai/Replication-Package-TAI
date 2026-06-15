#!/usr/bin/env python3
"""Run paired non-parametric tests for the Revision 2 PEFT sensitivity analysis.

The script expects the compact CSV files distributed under
Data/Revision2/PEFT_Hyperparameter_Sensitivity. Rows should already be averaged
across the three repeated runs for each base-model/configuration pair.
"""

from __future__ import annotations

import argparse
import math
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, norm, wilcoxon

METRICS = [
    "delta_safe_ratio",
    "delta_accuracy_ambig",
    "delta_accuracy_disambig",
    "delta_bias_score_amb_abs",
    "delta_bias_score_dis_abs",
]

METRIC_LABELS = {
    "delta_safe_ratio": "Change in Safety",
    "delta_accuracy_ambig": "Change in Accuracy AMB",
    "delta_accuracy_disambig": "Change in Accuracy DIS",
    "delta_bias_score_amb_abs": "Change in Bias Score AMB",
    "delta_bias_score_dis_abs": "Change in Bias Score DIS",
}

ORDERED_LEVELS = {
    "lora": ["r4", "r8", "r16"],
    "prompt": ["v20", "v50", "v100"],
    "ptuning": ["v20", "v50", "v100"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--numeric_by_model",
        default="Data/Revision2/PEFT_Hyperparameter_Sensitivity/numeric_sensitivity_delta_by_model.csv",
        help="CSV containing model-level LoRA/Prompt/P-Tuning sensitivity values.",
    )
    parser.add_argument(
        "--ia3_by_model",
        default="Data/Revision2/PEFT_Hyperparameter_Sensitivity/ia3_target_sensitivity/ia3_target_sensitivity_by_model.csv",
        help="CSV containing model-level IA3 target-module sensitivity values.",
    )
    parser.add_argument(
        "--output_dir",
        default="Data/Revision2/PEFT_Hyperparameter_Sensitivity/recomputed_tests",
        help="Directory where recomputed test tables are written.",
    )
    return parser.parse_args()


def effect_size_r(p_value: float, diffs: np.ndarray) -> float:
    diffs = np.asarray(diffs, dtype=float)
    n_nonzero = int(np.sum(diffs != 0))
    if n_nonzero == 0 or not np.isfinite(p_value) or p_value <= 0:
        return math.nan
    z_value = -norm.ppf(p_value / 2.0)
    return float(z_value / math.sqrt(n_nonzero))


def bonferroni(p_values: list[float]) -> list[float]:
    m = len(p_values)
    return [min(1.0, p * m) if np.isfinite(p) else math.nan for p in p_values]


def complete_matrix(df: pd.DataFrame, method: str, levels: list[str], metric: str) -> pd.DataFrame:
    sub = df[df["peft_method"].eq(method)]
    wide = sub.pivot_table(index="model", columns="setting_value", values=metric, aggfunc="mean")
    return wide.reindex(columns=levels).dropna()


def run_three_level_tests(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    omnibus_rows = []
    pairwise_rows = []
    prompt_two_level_rows = []

    for method, levels in ORDERED_LEVELS.items():
        for metric in METRICS:
            complete = complete_matrix(df, method, levels, metric)
            if len(complete) < 2:
                continue
            values = [complete[level].to_numpy(dtype=float) for level in levels]
            stat, p_value = friedmanchisquare(*values)
            kendalls_w = stat / (len(complete) * (len(levels) - 1))
            omnibus_rows.append({
                "peft_method": method,
                "metric": metric,
                "metric_label": METRIC_LABELS[metric],
                "levels": " vs ".join(levels),
                "n_blocks": len(complete),
                "test": "Friedman",
                "statistic": float(stat),
                "p_value": float(p_value),
                "effect_size_kendalls_w": float(kendalls_w),
                "significant_p_lt_0.05": bool(p_value < 0.05),
            })

            raw_pairwise = []
            for a, b in combinations(levels, 2):
                diff = complete[a].to_numpy(dtype=float) - complete[b].to_numpy(dtype=float)
                w_stat, w_p = wilcoxon(complete[a], complete[b])
                raw_pairwise.append((a, b, diff, float(w_stat), float(w_p)))
            adjusted = bonferroni([row[4] for row in raw_pairwise])
            for (a, b, diff, w_stat, w_p), p_adj in zip(raw_pairwise, adjusted):
                pairwise_rows.append({
                    "peft_method": method,
                    "metric": metric,
                    "metric_label": METRIC_LABELS[metric],
                    "comparison": f"{a} vs {b}",
                    "n_blocks": len(complete),
                    "test": "Wilcoxon signed-rank",
                    "statistic": w_stat,
                    "p_value": w_p,
                    "p_value_bonferroni": p_adj,
                    "effect_size_r": effect_size_r(w_p, diff),
                    "mean_diff_a_minus_b": float(np.mean(diff)),
                    "median_diff_a_minus_b": float(np.median(diff)),
                    "significant_bonferroni_p_lt_0.05": bool(p_adj < 0.05),
                })

            if method == "prompt":
                two = complete_matrix(df, method, ["v50", "v100"], metric)
                if len(two) >= 2:
                    diff = two["v50"].to_numpy(dtype=float) - two["v100"].to_numpy(dtype=float)
                    w_stat, w_p = wilcoxon(two["v50"], two["v100"])
                    prompt_two_level_rows.append({
                        "peft_method": method,
                        "metric": metric,
                        "metric_label": METRIC_LABELS[metric],
                        "comparison": "v50 vs v100",
                        "n_blocks": len(two),
                        "test": "Wilcoxon signed-rank",
                        "statistic": float(w_stat),
                        "p_value": float(w_p),
                        "effect_size_r": effect_size_r(float(w_p), diff),
                        "mean_diff_v50_minus_v100": float(np.mean(diff)),
                        "median_diff_v50_minus_v100": float(np.median(diff)),
                        "significant_p_lt_0.05": bool(w_p < 0.05),
                    })

    return pd.DataFrame(omnibus_rows), pd.DataFrame(pairwise_rows), pd.DataFrame(prompt_two_level_rows)


def run_ia3_tests(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    rows = []
    for metric in METRICS:
        wide = df.pivot_table(index="model", columns="setting_value", values=metric, aggfunc="mean")
        wide = wide.reindex(columns=["q_v_down", "k_v_down"]).dropna()
        if len(wide) < 2:
            continue
        diff = wide["q_v_down"].to_numpy(dtype=float) - wide["k_v_down"].to_numpy(dtype=float)
        stat, p_value = wilcoxon(wide["q_v_down"], wide["k_v_down"])
        rows.append({
            "metric": metric,
            "metric_label": METRIC_LABELS[metric],
            "comparison": "q_v_down vs k_v_down",
            "n_blocks": len(wide),
            "test": "Wilcoxon signed-rank",
            "statistic": float(stat),
            "p_value": float(p_value),
            "effect_size_r": effect_size_r(float(p_value), diff),
            "mean_diff_q_minus_k": float(np.mean(diff)),
            "median_diff_q_minus_k": float(np.median(diff)),
            "significant_p_lt_0.05": bool(p_value < 0.05),
        })
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    numeric_df = pd.read_csv(args.numeric_by_model)
    omnibus, pairwise, prompt_two = run_three_level_tests(numeric_df)
    ia3 = run_ia3_tests(args.ia3_by_model)

    omnibus.to_csv(output_dir / "paired_friedman_tests.csv", index=False)
    pairwise.to_csv(output_dir / "paired_wilcoxon_tests.csv", index=False)
    prompt_two.to_csv(output_dir / "prompt_v50_v100_wilcoxon_tests.csv", index=False)
    ia3.to_csv(output_dir / "ia3_target_sensitivity_wilcoxon_tests.csv", index=False)
    print(f"Wrote PEFT sensitivity tests to {output_dir}")


if __name__ == "__main__":
    main()
