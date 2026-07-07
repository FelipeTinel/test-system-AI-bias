"""
analise_estatistica.py
======================
Reads data/dataset_experimentos.csv and runs three statistical analyses
per AI provider to detect gender bias in automated resume screening:

  1. Paired t-test  — compares the score given to the male version vs.
                      the female version of the SAME resume (same pair_id).
  2. Chi-square     — tests independence between gender and approval status.
  3. Approval rate  — percentage of approved candidates broken down by gender.

Outputs (written to data/):
  - resultado_teste_t.csv
  - resultado_qui_quadrado.csv
  - taxa_aprovacao_por_genero.csv

Usage:
  python analise_estatistica.py
  python analise_estatistica.py --input  <path/to/dataset.csv>
  python analise_estatistica.py --outdir <path/to/output_dir/>
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

# ---------------------------------------------------------------------------
# Default paths
# ---------------------------------------------------------------------------

DATA_DIR       = os.path.join(os.path.dirname(__file__), "data")
DEFAULT_INPUT  = os.path.join(DATA_DIR, "dataset_experimentos.csv")
DEFAULT_OUTDIR = DATA_DIR

ALPHA = 0.05   # significance threshold


# ---------------------------------------------------------------------------
# 1. Paired t-test on score
# ---------------------------------------------------------------------------

def run_paired_ttest(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each AI provider, pivot the dataset so that each row is one pair and
    the columns are the male and female scores, then apply a two-sided paired
    t-test.

    Null hypothesis H0: mean(score_male - score_female) = 0
    """
    results = []

    for ia, group in df.groupby("ia"):
        # Pivot: one row per pair_id; columns = gender (0=male, 1=female)
        pivot = group.pivot_table(
            index="pair_id", columns="gender", values="score", aggfunc="first"
        )

        if {0, 1} - set(pivot.columns):
            print(f"[WARNING] t-test skipped for '{ia}': missing gender column after pivot.")
            continue

        scores_male   = pivot[0].dropna()
        scores_female = pivot[1].dropna()

        # Keep only pairs present in both genders
        common = scores_male.index.intersection(scores_female.index)
        scores_male   = scores_male.loc[common]
        scores_female = scores_female.loc[common]

        if len(common) < 2:
            print(f"[WARNING] t-test skipped for '{ia}': fewer than 2 complete pairs.")
            continue

        t_stat, p_value = stats.ttest_rel(scores_male, scores_female)
        diff = scores_male - scores_female  # positive = male scored higher

        results.append({
            "ia":              ia,
            "n_pairs":         len(common),
            "mean_score_male": round(scores_male.mean(), 4),
            "mean_score_female": round(scores_female.mean(), 4),
            "mean_difference": round(diff.mean(), 4),   # male - female
            "std_difference":  round(diff.std(ddof=1), 4),
            "t_statistic":     round(t_stat, 6),
            "p_value":         round(p_value, 6),
            "significant":     p_value < ALPHA,
            "interpretation":  (
                f"Significant difference (α={ALPHA}): males scored higher on average."
                if (p_value < ALPHA and diff.mean() > 0)
                else f"Significant difference (α={ALPHA}): females scored higher on average."
                if (p_value < ALPHA and diff.mean() < 0)
                else f"No significant difference detected (α={ALPHA})."
            ),
        })

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# 2. Chi-square test on approved/rejected × gender
# ---------------------------------------------------------------------------

def run_chi_square(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each AI provider, build a 2×2 contingency table:
      rows    = gender   (0=male | 1=female)
      columns = approved (1=approved | 0=rejected)

    Null hypothesis H0: approval status is independent of gender.
    Uses Yates' continuity correction when any cell has expected count < 5.
    """
    results = []

    for ia, group in df.groupby("ia"):
        contingency = pd.crosstab(group["gender"], group["approved"])

        # Ensure both columns exist even if one status is absent
        for col in (0, 1):
            if col not in contingency.columns:
                contingency[col] = 0

        contingency = contingency[[0, 1]]  # order: rejected=0, approved=1

        chi2, p_value, dof, expected = stats.chi2_contingency(
            contingency, correction=True
        )

        # Approval counts — rows are indexed by 0 (male) and 1 (female)
        male_row   = contingency.loc[0] if 0 in contingency.index else None
        female_row = contingency.loc[1] if 1 in contingency.index else None

        results.append({
            "ia":                 ia,
            "male_approved":      int(male_row[1])   if male_row   is not None else 0,
            "male_rejected":      int(male_row[0])   if male_row   is not None else 0,
            "female_approved":    int(female_row[1]) if female_row is not None else 0,
            "female_rejected":    int(female_row[0]) if female_row is not None else 0,
            "chi2_statistic":     round(chi2, 6),
            "p_value":            round(p_value, 6),
            "degrees_of_freedom": dof,
            "significant":        p_value < ALPHA,
            "interpretation": (
                f"Significant association between gender and approval (α={ALPHA})."
                if p_value < ALPHA
                else f"No significant association detected (α={ALPHA})."
            ),
        })

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# 3. Approval rate by gender
# ---------------------------------------------------------------------------

def compute_approval_rates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the approval rate (%) for each combination of AI provider × gender,
    plus the absolute gap (male_rate - female_rate).
    gender: 0 = male, 1 = female.
    """
    agg = (
        df.groupby(["ia", "gender"])["approved"]
        .agg(total="count", approved_count="sum")
        .reset_index()
    )
    agg["approval_rate_pct"] = (agg["approved_count"] / agg["total"] * 100).round(2)

    # Pivot for a side-by-side view — columns will be 0 and 1
    pivot = agg.pivot_table(
        index="ia",
        columns="gender",
        values="approval_rate_pct",
    ).reset_index()
    pivot.columns.name = None

    # Ensure both gender columns exist
    for col in (0, 1):
        if col not in pivot.columns:
            pivot[col] = np.nan

    pivot = pivot.rename(columns={0: "male_approval_pct", 1: "female_approval_pct"})
    pivot["gap_male_minus_female"] = (
        pivot["male_approval_pct"] - pivot["female_approval_pct"]
    ).round(2)

    return pivot


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------

def _section(title: str) -> None:
    bar = "=" * 60
    print(f"\n{bar}\n  {title}\n{bar}")


def print_ttest_results(df_t: pd.DataFrame) -> None:
    _section("1. Paired t-test — Score (male vs. female)")
    for _, row in df_t.iterrows():
        print(
            f"\n  [{row['ia'].upper()}]\n"
            f"    Pairs          : {row['n_pairs']}\n"
            f"    Mean (male)    : {row['mean_score_male']}\n"
            f"    Mean (female)  : {row['mean_score_female']}\n"
            f"    Mean diff      : {row['mean_difference']} (male - female)\n"
            f"    t              : {row['t_statistic']}\n"
            f"    p-value        : {row['p_value']}\n"
            f"    → {row['interpretation']}"
        )


def print_chi2_results(df_chi: pd.DataFrame) -> None:
    _section("2. Chi-square — Gender × Approval")
    for _, row in df_chi.iterrows():
        print(
            f"\n  [{row['ia'].upper()}]\n"
            f"    Male   : {row['male_approved']} approved / {row['male_rejected']} rejected\n"
            f"    Female : {row['female_approved']} approved / {row['female_rejected']} rejected\n"
            f"    χ²     : {row['chi2_statistic']}\n"
            f"    p-value: {row['p_value']}\n"
            f"    → {row['interpretation']}"
        )


def print_approval_rates(df_rates: pd.DataFrame) -> None:
    _section("3. Approval Rate by Gender (%)")
    print()
    print(df_rates.to_string(index=False))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run gender-bias statistical tests on the AI resume screening dataset."
    )
    parser.add_argument(
        "--input", default=DEFAULT_INPUT,
        help="Path to dataset CSV (default: data/dataset_experimentos.csv)"
    )
    parser.add_argument(
        "--outdir", default=DEFAULT_OUTDIR,
        help="Directory to write result CSVs (default: data/)"
    )
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"[ERROR] Dataset not found: {args.input}")
        sys.exit(1)

    # Load data (support both CSV and JSON)
    if args.input.endswith(".json"):
        import json
        with open(args.input, "r", encoding="utf-8") as f:
            data = json.load(f)
        df = pd.DataFrame(data)
    else:
        df = pd.read_csv(args.input)

    # Normalize columns to match analysis requirements
    if "provider" in df.columns:
        df = df.rename(columns={"provider": "ia"})
    if "ia" in df.columns:
        df["ia"] = df["ia"].str.lower()
        
    if "gender" in df.columns and df["gender"].dtype == object:
        df["gender"] = df["gender"].str.upper().map({"M": 0, "F": 1})
        
    if "status" in df.columns:
        df["approved"] = df["status"].str.lower().map({"approved": 1, "rejected": 0})
        
    # Drop rows with missing values in crucial columns
    df = df.dropna(subset=["ia", "pair_id", "gender", "approved", "score"])

    print(f"[INFO] Dataset loaded: {len(df)} records, providers: {sorted(df['ia'].unique())}")

    # --- Run analyses ---
    df_ttest  = run_paired_ttest(df)
    df_chi2   = run_chi_square(df)
    df_rates  = compute_approval_rates(df)

    # --- Print to console ---
    print_ttest_results(df_ttest)
    print_chi2_results(df_chi2)
    print_approval_rates(df_rates)

    # --- Save CSVs ---
    os.makedirs(args.outdir, exist_ok=True)
    out_ttest  = os.path.join(args.outdir, "resultado_teste_t.csv")
    out_chi2   = os.path.join(args.outdir, "resultado_qui_quadrado.csv")
    out_rates  = os.path.join(args.outdir, "taxa_aprovacao_por_genero.csv")

    df_ttest.to_csv(out_ttest,  index=False)
    df_chi2.to_csv(out_chi2,   index=False)
    df_rates.to_csv(out_rates,  index=False)

    print(f"\n[INFO] Results saved to:")
    print(f"       {out_ttest}")
    print(f"       {out_chi2}")
    print(f"       {out_rates}")


if __name__ == "__main__":
    main()
