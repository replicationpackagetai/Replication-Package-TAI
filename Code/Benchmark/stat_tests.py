#!/usr/bin/env python3
"""
Paired Statistical Tests for Fairness/Safety/Utility Experiments

This script runs paired nonparametric tests across factor levels:
  - 2 groups  -> Wilcoxon signed-rank test
  - ≥3 groups -> Friedman test + post-hoc Wilcoxon with p-value correction
Optional:
  - --compare_to_zero -> one-sample Wilcoxon per group vs 0 (baseline)

It expects a single “wide” CSV that includes:
  - factor columns (e.g., dataset, peft_method, model_name, strategy, learning_rate, epochs)
  - per-row identifiers (these same factor columns, except the tested factor)
  - metric columns, following either:
      absolute:   "<Category>_<metric>"
      diff-mode:  "diff_<Category>_<metric>"
    Example: "diff_Age_accuracy_ambig", "diff_bias_score_dis"

Outputs (per factor):
  - <prefix>__tests_<factor>.csv       : main tests (Wilcoxon/Friedman)
  - <prefix>__posthoc_<factor>.csv     : post-hoc pairwise Wilcoxon (if applicable)
  - <prefix>__vszero_<factor>.csv      : one-sample Wilcoxon vs 0 (if --compare_to_zero)

Rollups:
  - <prefix>__tests_ALL.csv
  - <prefix>__posthoc_ALL.csv
  - <prefix>__vszero_ALL.csv

Example:
  python run_stats_tests.py \
    --input_csv final_final/bias_safety_utility_abs_diff_rel_signflip_minus_outliers_mean.csv \
    --factors peft_method,model_name \
    --metrics accuracy_ambig,accuracy_disambig,bias_score_amb,bias_score_dis \
    --categories Age,Disability_status,Gender_identity,Nationality,Physical_appearance,Race_ethnicity,Religion,SES,Sexual_orientation \
    --mode diff \
    --alpha 0.05 \
    --correction bonferroni \
    --compare_to_zero \
    --output_prefix stats_tests
"""

import argparse
from itertools import combinations
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, friedmanchisquare, norm
from statsmodels.stats.multitest import multipletests


ALL_FACTORS = ["dataset", "peft_method", "model_name", "strategy", "learning_rate", "epochs"]

DEFAULT_CATEGORIES = [
    "Age", "Disability_status", "Gender_identity", "Nationality",
    "Physical_appearance", "Race_ethnicity", "Religion", "SES", "Sexual_orientation"
]


# --------------------------- CLI ---------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Paired nonparametric tests across factor levels.")
    p.add_argument("--input_csv", "-i", type=str, required=True, help="Path to the input CSV.")
    p.add_argument("--factors", "-f", type=str, required=True,
                   help="Comma-separated list of factor columns to test (e.g., 'peft_method,model_name').")
    p.add_argument("--metrics", "-m", type=str, required=True,
                   help="Comma-separated list of base metrics (e.g., 'accuracy_ambig,bias_score_dis').")
    p.add_argument("--categories", "-c", type=str, default=",".join(DEFAULT_CATEGORIES),
                   help="Comma-separated list of categories (for fairness metrics).")
    p.add_argument("--mode", type=str, default="diff", choices=["diff", "absolute"],
                   help="Use 'diff' to look for columns 'diff_<Category>_<metric>'.")
    p.add_argument("--alpha", type=float, default=0.05, help="Significance level.")
    p.add_argument("--correction", type=str, default="bonferroni",
                   choices=["bonferroni", "holm"], help="P-value correction for post-hoc tests.")
    p.add_argument("--compare_to_zero", action="store_true",
                   help="Also run one-sample Wilcoxon tests vs 0 for each group.")
    p.add_argument("--output_prefix", type=str, default="stats_tests",
                   help="Prefix for output CSV filenames.")
    return p.parse_args()


# ------------------------- Helpers -------------------------

def col_name(cat: str, metric: str, mode: str) -> str:
    """Build the metric column name."""
    return f"diff_{cat}_{metric}" if mode == "diff" else f"{cat}_{metric}"


def align_groups_by_keys(groups: Dict[str, pd.DataFrame], keys: List[str]) -> Dict[str, pd.DataFrame]:
    """
    Align group DataFrames to the intersection of key tuples, keeping identical row order across groups.
    Returns dict of aligned DataFrames indexed by a MultiIndex built from `keys`.
    """
    key_sets = []
    keyed = {}
    for name, g in groups.items():
        gi = g.dropna(subset=keys) if keys else g.copy()
        if keys:
            gi = gi.set_index(keys, drop=False)
        keyed[name] = gi
        key_sets.append(set(gi.index.to_list()) if keys else set(range(len(gi))))

    common = set.intersection(*key_sets) if key_sets else set()
    if not common:
        return {name: g.iloc[0:0].copy() for name, g in keyed.items()}

    common_sorted = sorted(common)
    aligned = {name: gi.loc[common_sorted] if keys else gi.iloc[common_sorted] for name, gi in keyed.items()}
    return aligned


def effect_size_from_wilcoxon(p: float, diffs: np.ndarray) -> float:
    """
    Compute effect size r from Wilcoxon two-sided p-value and non-zero diffs.
    r = |z| / sqrt(n_nonzero)
    """
    nonzero = np.sum(diffs != 0)
    if nonzero == 0 or p is None or np.isnan(p) or p <= 0:
        return np.nan
    z = -norm.ppf(p / 2.0)  # two-sided
    return float(z / np.sqrt(nonzero))


# --------------------- Test Runners ------------------------

def run_wilcoxon(a: np.ndarray, b: np.ndarray) -> Tuple[float, float, float, str]:
    """
    Paired Wilcoxon signed-rank test. Returns (stat, p, effect_size_r, direction).
    """
    stat, p = wilcoxon(a, b)
    diffs = a - b
    r = effect_size_from_wilcoxon(p, diffs)
    mean_diff = float(np.mean(diffs))
    direction = "group0>group1" if mean_diff > 0 else "group1>group0" if mean_diff < 0 else "tie"
    return float(stat), float(p), r, direction


def run_friedman(arrs: List[np.ndarray]) -> Tuple[float, float, float]:
    """
    Friedman test across k>=3 paired groups.
    Returns (stat, p, Kendall_W).
    """
    stat, p = friedmanchisquare(*arrs)
    k = len(arrs)
    n = len(arrs[0]) if arrs and len(arrs[0]) else 0
    W = float(stat / (n * (k - 1))) if (n and k > 1) else float("nan")
    return float(stat), float(p), W


def posthoc_wilcoxon(arrs: List[np.ndarray], names: List[str], correction: str) -> pd.DataFrame:
    """
    Pairwise Wilcoxon for all group pairs with multiple-testing correction.
    """
    rows, pvals, meta = [], [], []
    for i, j in combinations(range(len(arrs)), 2):
        a, b = arrs[i], arrs[j]
        stat, p = wilcoxon(a, b)
        diffs = a - b
        r = effect_size_from_wilcoxon(p, diffs)
        direction = f"{names[i]}>{names[j]}" if np.mean(diffs) > 0 else f"{names[j]}>{names[i]}" if np.mean(diffs) < 0 else "tie"
        pvals.append(p)
        meta.append((i, j, stat, r, direction))
    if not pvals:
        return pd.DataFrame()
    reject, p_adj, _, _ = multipletests(pvals, method=correction)
    for k, (i, j, stat, r, direction) in enumerate(meta):
        rows.append({
            "group_i": names[i],
            "group_j": names[j],
            "statistic": float(stat),
            "p_value": float(pvals[k]),
            "p_value_adj": float(p_adj[k]),
            "significant": bool(reject[k]),
            "effect_size_r": float(r),
            "direction": direction,
        })
    return pd.DataFrame(rows)


def one_sample_wilcoxon(x: np.ndarray) -> Tuple[float, float, float]:
    """
    One-sample Wilcoxon vs 0. Returns (stat, p, effect_size_r).
    """
    stat, p = wilcoxon(x)  # tests median(x) != 0
    r = effect_size_from_wilcoxon(p, x)  # compare to zero => diffs = x - 0 = x
    return float(stat), float(p), float(r)


# -------------------- Orchestration ------------------------

def analyze_factor(df: pd.DataFrame,
                   factor: str,
                   metrics: List[str],
                   categories: List[str],
                   mode: str,
                   alpha: float,
                   correction: str,
                   compare_to_zero: bool) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Run Wilcoxon (2 groups) or Friedman + post-hoc Wilcoxon (≥3 groups)
    for each (category, metric) under the given factor.
    Optionally, run one-sample Wilcoxon vs 0 per group.

    Returns:
      main_df  : primary test results
      post_df  : post-hoc pairwise results (may be empty)
      vs0_df   : one-sample vs 0 results per group (may be empty)
    """
    grouped = {name: g.copy() for name, g in df.groupby(factor)}
    if len(grouped) < 2 and not compare_to_zero:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    keys = [c for c in ALL_FACTORS if c != factor and c in df.columns]

    main_rows, post_rows, vs0_rows = [], [], []

    # Align groups for paired testing
    if len(grouped) >= 2:
        aligned = align_groups_by_keys(grouped, keys)
        aligned = {k: v for k, v in aligned.items() if len(v) > 0}
        names = list(aligned.keys())
        k = len(names)
    else:
        aligned, names, k = {}, [], 0

    # For vs-zero, we use each group's own rows (no alignment needed)
    for cat in categories:
        for metric in metrics:
            col = col_name(cat, metric, mode)

            # --------- Paired tests across groups (Wilcoxon/Friedman) ---------
            if k >= 2 and all(col in g.columns for g in aligned.values()):
                arrs = [pd.to_numeric(aligned[n][col], errors="coerce").to_numpy() for n in names]
                mask = np.ones_like(arrs[0], dtype=bool)
                for a in arrs:
                    mask &= ~np.isnan(a)
                arrs = [a[mask] for a in arrs]
                if any(len(a) == 0 for a in arrs):
                    pass
                else:
                    if k == 2:
                        stat, p, r, direction = run_wilcoxon(arrs[0], arrs[1])
                        main_rows.append({
                            "factor": factor,
                            "category": cat,
                            "metric": metric,
                            "groups": f"{names[0]} vs {names[1]}",
                            "test": "wilcoxon",
                            "statistic": stat,
                            "p_value": p,
                            "significant": bool(p < alpha),
                            "effect_size_r": r,
                            "direction": direction,
                            "n": int(len(arrs[0])),
                        })
                    else:
                        stat, p, W = run_friedman(arrs)
                        main_rows.append({
                            "factor": factor,
                            "category": cat,
                            "metric": metric,
                            "groups": ",".join(names),
                            "test": "friedman",
                            "statistic": stat,
                            "p_value": p,
                            "significant": bool(p < alpha),
                            "kendalls_W": W,
                            "n": int(len(arrs[0])),
                        })
                        if p < alpha:
                            post = posthoc_wilcoxon(arrs, names, correction)
                            if not post.empty:
                                post.insert(0, "factor", factor)
                                post.insert(1, "category", cat)
                                post.insert(2, "metric", metric)
                                post_rows.append(post)

            # --------- One-sample vs 0, per group (optional) ---------
            if compare_to_zero:
                for gname, gdf in grouped.items():
                    if col not in gdf.columns:
                        continue
                    x = pd.to_numeric(gdf[col], errors="coerce").dropna().to_numpy()
                    if x.size == 0:
                        continue
                    stat, p, r = one_sample_wilcoxon(x)
                    vs0_rows.append({
                        "factor": factor,
                        "group": gname,
                        "category": cat,
                        "metric": metric,
                        "test": "wilcoxon_one_sample",
                        "statistic": stat,
                        "p_value": p,
                        "significant": bool(p < alpha),
                        "effect_size_r": r,
                        "n": int(x.size),
                        "mean": float(np.mean(x)),
                        "median": float(np.median(x)),
                    })

    main_df = pd.DataFrame(main_rows)
    post_df = pd.concat(post_rows, ignore_index=True) if post_rows else pd.DataFrame()
    vs0_df = pd.DataFrame(vs0_rows)
    return main_df, post_df, vs0_df


def main():
    args = parse_args()
    df = pd.read_csv(args.input_csv)

    factors = [x.strip() for x in args.factors.split(",") if x.strip()]
    metrics = [x.strip() for x in args.metrics.split(",") if x.strip()]
    categories = [x.strip() for x in args.categories.split(",") if x.strip()]

    all_main, all_post, all_vs0 = [], [], []

    for factor in factors:
        if factor not in df.columns:
            continue
        main_df, post_df, vs0_df = analyze_factor(
            df=df,
            factor=factor,
            metrics=metrics,
            categories=categories,
            mode=args.mode,
            alpha=args.alpha,
            correction=args.correction,
            compare_to_zero=args.compare_to_zero
        )

        main_path = f"{args.output_prefix}__tests_{factor}.csv"
        post_path = f"{args.output_prefix}__posthoc_{factor}.csv"
        vs0_path = f"{args.output_prefix}__vszero_{factor}.csv"

        main_df.to_csv(main_path, index=False)
        post_df.to_csv(post_path, index=False)
        vs0_df.to_csv(vs0_path, index=False)

        print(f"Wrote: {main_path} ({len(main_df)} rows)")
        print(f"Wrote: {post_path} ({len(post_df)} rows)")
        print(f"Wrote: {vs0_path} ({len(vs0_df)} rows)")

        all_main.append(main_df)
        if not post_df.empty:
            all_post.append(post_df)
        if not vs0_df.empty:
            all_vs0.append(vs0_df)

    if any(len(df_) for df_ in all_main):
        combined_main = pd.concat(all_main, ignore_index=True)
        combined_main.to_csv(f"{args.output_prefix}__tests_ALL.csv", index=False)
        print(f"Wrote: {args.output_prefix}__tests_ALL.csv ({len(combined_main)} rows)")

    if any(len(df_) for df_ in all_post):
        combined_post = pd.concat(all_post, ignore_index=True)
        combined_post.to_csv(f"{args.output_prefix}__posthoc_ALL.csv", index=False)
        print(f"Wrote: {args.output_prefix}__posthoc_ALL.csv ({len(combined_post)} rows)")

    if any(len(df_) for df_ in all_vs0):
        combined_vs0 = pd.concat(all_vs0, ignore_index=True)
        combined_vs0.to_csv(f"{args.output_prefix}__vszero_ALL.csv", index=False)
        print(f"Wrote: {args.output_prefix}__vszero_ALL.csv ({len(combined_vs0)} rows)")


if __name__ == "__main__":
    main()
