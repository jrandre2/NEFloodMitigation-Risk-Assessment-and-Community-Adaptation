"""
oaxaca_blinder_decomposition.py
===============================
Formal Oaxaca-Blinder decomposition of the positive price effect.

This module decomposes the observed price difference between inside and outside
the flood zone post-event into:
1. COMPOSITION EFFECT: Due to differences in property characteristics
2. COEFFICIENT EFFECT: Due to differences in returns to characteristics
3. INTERACTION EFFECT: Due to simultaneous differences in both

If the composition effect explains >50% of the positive price effect,
this supports the "rebuild" mechanism hypothesis.

Author: Claude Code
Date: 2025-12-23
"""

import pandas as pd
import numpy as np
from pathlib import Path
import statsmodels.api as sm
from typing import Dict, Tuple, Optional
import warnings

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_WORK = PROJECT_ROOT / "data_work"
DIAGNOSTICS_DIR = DATA_WORK / "diagnostics"
FIGURES_DIR = PROJECT_ROOT / "figures"

# Event date
EVENT_DATE = pd.Timestamp("2019-03-01")


def load_data(caliper_m: int = 300) -> pd.DataFrame:
    """Load sales data with covariates for decomposition."""

    # Load sales
    sales_path = DATA_WORK / "sales_clean.parquet"
    df = pd.read_parquet(sales_path)
    print(f"Loaded {len(df):,} sales")

    # Merge with boundary distances
    distances_path = DATA_WORK / "parcel_boundary_distances.parquet"
    if distances_path.exists():
        distances = pd.read_parquet(distances_path)
        df = df.merge(
            distances[["parcel_id", "signed_dist_inund_m", "inside_inund"]],
            on="parcel_id",
            how="inner"
        )
        print(f"After merge with distances: {len(df):,}")

    # Merge with covariates
    covariates_path = DATA_WORK / "parcel_covariates_full.parquet"
    if covariates_path.exists():
        covariates = pd.read_parquet(covariates_path)
        # Select key covariates
        cov_cols = ["parcel_id"]
        for col in ["year_built", "Total_Assessed_Value", "GIS_Acres", "ImpSF",
                    "Land_Value", "Improvements_Value", "neighborhood"]:
            if col in covariates.columns:
                cov_cols.append(col)

        df = df.merge(covariates[cov_cols], on="parcel_id", how="left")
        print(f"After merge with covariates: {len(df):,}")

    # Create derived variables
    df["sale_date"] = pd.to_datetime(df["sale_date"])
    df["sale_year"] = df["sale_date"].dt.year
    df["post"] = df["sale_date"] >= EVENT_DATE

    # Building age at sale
    if "year_built" in df.columns:
        df["building_age"] = df["sale_year"] - df["year_built"]
        df["building_age"] = df["building_age"].clip(lower=0)

    # Log transformations
    if "price_nominal" in df.columns:
        df["log_price"] = np.log(df["price_nominal"].clip(lower=1))
    if "Total_Assessed_Value" in df.columns:
        df["log_assessed_value"] = np.log(df["Total_Assessed_Value"].clip(lower=1))
    if "GIS_Acres" in df.columns:
        df["log_acres"] = np.log(df["GIS_Acres"].clip(lower=0.01))
    if "ImpSF" in df.columns:
        df["log_impsf"] = np.log(df["ImpSF"].clip(lower=1))

    # Filter to caliper window
    df = df[df["signed_dist_inund_m"].abs() <= caliper_m]
    print(f"After caliper filter ({caliper_m}m): {len(df):,}")

    return df


def oaxaca_blinder_twofold(
    df: pd.DataFrame,
    outcome: str,
    covariates: list,
    group_var: str,
    reference_group: int = 0
) -> Dict:
    """
    Perform twofold Oaxaca-Blinder decomposition.

    The twofold decomposition splits the gap into:
    - Explained (E): Due to differences in X
    - Unexplained (U): Due to differences in returns to X

    Parameters
    ----------
    df : DataFrame
        Data with outcome, covariates, and group indicator
    outcome : str
        Name of outcome variable (e.g., 'log_price')
    covariates : list
        List of covariate names
    group_var : str
        Name of binary group variable (0/1)
    reference_group : int
        Which group to use as reference (0 or 1)

    Returns
    -------
    dict : Decomposition results
    """

    # Split by group
    df0 = df[df[group_var] == 0].dropna(subset=[outcome] + covariates)
    df1 = df[df[group_var] == 1].dropna(subset=[outcome] + covariates)

    if len(df0) < 30 or len(df1) < 30:
        return {"error": "Insufficient observations in one or both groups"}

    # Means
    y0_mean = df0[outcome].mean()
    y1_mean = df1[outcome].mean()
    gap = y1_mean - y0_mean

    X0 = df0[covariates]
    X1 = df1[covariates]
    X0_mean = X0.mean()
    X1_mean = X1.mean()

    # Add constant
    X0_const = sm.add_constant(X0)
    X1_const = sm.add_constant(X1)

    # Estimate separate regressions
    model0 = sm.OLS(df0[outcome], X0_const).fit()
    model1 = sm.OLS(df1[outcome], X1_const).fit()

    beta0 = model0.params
    beta1 = model1.params

    # Decomposition using reference group coefficients
    if reference_group == 0:
        # Use group 0 coefficients as reference
        # E = (X1 - X0) * beta0
        # U = X1 * (beta1 - beta0)
        explained = 0
        unexplained = 0

        for var in covariates:
            x_diff = X1_mean[var] - X0_mean[var]
            beta_diff = beta1[var] - beta0[var]

            explained += x_diff * beta0[var]
            unexplained += X1_mean[var] * beta_diff

        # Constant contributes only to unexplained
        unexplained += beta1["const"] - beta0["const"]

    else:
        # Use group 1 coefficients as reference
        explained = 0
        unexplained = 0

        for var in covariates:
            x_diff = X1_mean[var] - X0_mean[var]
            beta_diff = beta1[var] - beta0[var]

            explained += x_diff * beta1[var]
            unexplained += X0_mean[var] * beta_diff

        unexplained += beta1["const"] - beta0["const"]

    # Percentage explained
    pct_explained = (explained / gap * 100) if abs(gap) > 0.001 else np.nan
    pct_unexplained = (unexplained / gap * 100) if abs(gap) > 0.001 else np.nan

    return {
        "gap": gap,
        "explained": explained,
        "unexplained": unexplained,
        "pct_explained": pct_explained,
        "pct_unexplained": pct_unexplained,
        "n_group0": len(df0),
        "n_group1": len(df1),
        "y0_mean": y0_mean,
        "y1_mean": y1_mean,
        "r2_group0": model0.rsquared,
        "r2_group1": model1.rsquared,
        "model0": model0,
        "model1": model1
    }


def detailed_decomposition(
    df: pd.DataFrame,
    outcome: str,
    covariates: list,
    group_var: str
) -> pd.DataFrame:
    """
    Compute detailed decomposition showing contribution of each covariate.

    Returns DataFrame with each covariate's contribution to explained
    and unexplained components.
    """

    # Split by group
    df0 = df[df[group_var] == 0].dropna(subset=[outcome] + covariates)
    df1 = df[df[group_var] == 1].dropna(subset=[outcome] + covariates)

    if len(df0) < 30 or len(df1) < 30:
        return pd.DataFrame()

    X0 = df0[covariates]
    X1 = df1[covariates]
    X0_mean = X0.mean()
    X1_mean = X1.mean()

    # Estimate regressions
    X0_const = sm.add_constant(X0)
    X1_const = sm.add_constant(X1)

    model0 = sm.OLS(df0[outcome], X0_const).fit()
    model1 = sm.OLS(df1[outcome], X1_const).fit()

    beta0 = model0.params
    beta1 = model1.params
    se0 = model0.bse
    se1 = model1.bse

    results = []

    for var in covariates:
        x_diff = X1_mean[var] - X0_mean[var]
        beta_diff = beta1[var] - beta0[var]

        # Explained: characteristic difference × reference coefficient
        explained_contrib = x_diff * beta0[var]

        # Unexplained: group 1 mean × coefficient difference
        unexplained_contrib = X1_mean[var] * beta_diff

        results.append({
            "variable": var,
            "x0_mean": X0_mean[var],
            "x1_mean": X1_mean[var],
            "x_diff": x_diff,
            "beta0": beta0[var],
            "beta1": beta1[var],
            "beta_diff": beta_diff,
            "explained_contrib": explained_contrib,
            "unexplained_contrib": unexplained_contrib,
            "total_contrib": explained_contrib + unexplained_contrib
        })

    return pd.DataFrame(results)


def run_decomposition_analysis(
    df: pd.DataFrame,
    caliper_m: int = 300
) -> Dict:
    """
    Run full decomposition analysis for post-flood period.

    Compares inside vs. outside post-flood to decompose the observed
    price difference.
    """

    print("\n" + "="*70)
    print("OAXACA-BLINDER DECOMPOSITION ANALYSIS")
    print("="*70)

    # Filter to post-flood only
    df_post = df[df["post"] == True].copy()
    n_inside = (df_post["inside_inund"] == True).sum()
    n_outside = (df_post["inside_inund"] == False).sum()
    print(f"\nPost-flood sample: {len(df_post):,} sales")
    print(f"  Inside: {n_inside:,}")
    print(f"  Outside: {n_outside:,}")

    # Define covariates
    potential_covariates = [
        "building_age", "log_assessed_value", "log_acres", "log_impsf"
    ]

    available_covariates = [c for c in potential_covariates if c in df_post.columns]
    print(f"\nUsing covariates: {available_covariates}")

    if len(available_covariates) == 0:
        print("ERROR: No covariates available for decomposition")
        return {}

    # Convert inside_inund to int for grouping
    df_post["group"] = df_post["inside_inund"].astype(int)

    # Run twofold decomposition
    print("\n" + "-"*50)
    print("TWOFOLD DECOMPOSITION")
    print("-"*50)

    result = oaxaca_blinder_twofold(
        df_post,
        outcome="log_price",
        covariates=available_covariates,
        group_var="group",
        reference_group=0  # Outside as reference
    )

    if "error" in result:
        print(f"ERROR: {result['error']}")
        return result

    print(f"\nPrice Gap (Inside - Outside): {result['gap']:.4f} log points")
    print(f"  = {(np.exp(result['gap']) - 1) * 100:.1f}% price difference")
    print(f"\nDecomposition:")
    print(f"  Explained (composition): {result['explained']:.4f} ({result['pct_explained']:.1f}%)")
    print(f"  Unexplained (coefficients): {result['unexplained']:.4f} ({result['pct_unexplained']:.1f}%)")
    print(f"\nSample sizes:")
    print(f"  Outside (reference): {result['n_group0']:,}")
    print(f"  Inside: {result['n_group1']:,}")
    print(f"\nModel fit (R-squared):")
    print(f"  Outside: {result['r2_group0']:.3f}")
    print(f"  Inside: {result['r2_group1']:.3f}")

    # Detailed decomposition
    print("\n" + "-"*50)
    print("DETAILED DECOMPOSITION BY COVARIATE")
    print("-"*50)

    detailed = detailed_decomposition(
        df_post,
        outcome="log_price",
        covariates=available_covariates,
        group_var="group"
    )

    if len(detailed) > 0:
        print("\nContribution of each covariate:")
        for _, row in detailed.iterrows():
            print(f"\n  {row['variable']}:")
            print(f"    Mean (outside): {row['x0_mean']:.3f}")
            print(f"    Mean (inside):  {row['x1_mean']:.3f}")
            print(f"    Difference:     {row['x_diff']:.3f}")
            print(f"    Explained:      {row['explained_contrib']:.4f}")
            print(f"    Unexplained:    {row['unexplained_contrib']:.4f}")

        # Summary table
        print("\n" + "-"*50)
        total_explained = detailed["explained_contrib"].sum()
        total_unexplained = detailed["unexplained_contrib"].sum()

        print(f"\nTotal explained by covariates: {total_explained:.4f}")
        print(f"Total unexplained: {total_unexplained:.4f}")

        result["detailed"] = detailed

    return result


def run_did_decomposition(df: pd.DataFrame) -> Dict:
    """
    Run DiD-style decomposition comparing:
    - (Inside Post - Inside Pre) vs. (Outside Post - Outside Pre)

    This decomposes the DiD effect rather than cross-sectional gap.
    """

    print("\n" + "="*70)
    print("DiD-STYLE DECOMPOSITION")
    print("="*70)

    # Create period-zone groups
    df["period_zone"] = df.apply(
        lambda r: f"{'inside' if r['inside_inund'] else 'outside'}_{'post' if r['post'] else 'pre'}",
        axis=1
    )

    # Compute means by group
    potential_covariates = ["building_age", "log_assessed_value", "log_acres", "log_impsf"]
    available_covariates = [c for c in potential_covariates if c in df.columns]

    group_means = df.groupby("period_zone").agg({
        "log_price": "mean",
        **{c: "mean" for c in available_covariates},
        "parcel_id": "count"
    }).rename(columns={"parcel_id": "n"})

    print("\nGroup means:")
    print(group_means.round(3))

    # Compute DiD for price
    if all(g in group_means.index for g in ["inside_post", "inside_pre", "outside_post", "outside_pre"]):
        price_did = (
            (group_means.loc["inside_post", "log_price"] - group_means.loc["inside_pre", "log_price"]) -
            (group_means.loc["outside_post", "log_price"] - group_means.loc["outside_pre", "log_price"])
        )
        print(f"\nPrice DiD: {price_did:.4f} ({(np.exp(price_did)-1)*100:.1f}%)")

        # Compute DiD for each covariate
        print("\nCovariate DiD (composition changes):")
        covariate_dids = {}
        for cov in available_covariates:
            cov_did = (
                (group_means.loc["inside_post", cov] - group_means.loc["inside_pre", cov]) -
                (group_means.loc["outside_post", cov] - group_means.loc["outside_pre", cov])
            )
            covariate_dids[cov] = cov_did
            print(f"  {cov}: {cov_did:.4f}")

        return {
            "price_did": price_did,
            "covariate_dids": covariate_dids,
            "group_means": group_means
        }

    return {}


def main(caliper_m: int = 300):
    """Run Oaxaca-Blinder decomposition analysis."""

    # Load data
    df = load_data(caliper_m)

    # Run cross-sectional decomposition (post-period only)
    result = run_decomposition_analysis(df, caliper_m)

    # Run DiD-style decomposition
    did_result = run_did_decomposition(df)

    # Save results
    DIAGNOSTICS_DIR.mkdir(parents=True, exist_ok=True)

    # Save main decomposition results
    if "gap" in result:
        summary = pd.DataFrame([{
            "analysis": "post_period_decomposition",
            "caliper_m": caliper_m,
            "price_gap": result["gap"],
            "explained": result["explained"],
            "unexplained": result["unexplained"],
            "pct_explained": result["pct_explained"],
            "pct_unexplained": result["pct_unexplained"],
            "n_inside": result["n_group1"],
            "n_outside": result["n_group0"],
            "r2_inside": result["r2_group1"],
            "r2_outside": result["r2_group0"]
        }])
        summary.to_csv(DIAGNOSTICS_DIR / "oaxaca_blinder_results.csv", index=False)
        print(f"\nSaved: {DIAGNOSTICS_DIR / 'oaxaca_blinder_results.csv'}")

    # Save detailed results
    if "detailed" in result:
        result["detailed"].to_csv(
            DIAGNOSTICS_DIR / "oaxaca_blinder_detailed.csv",
            index=False
        )
        print(f"Saved: {DIAGNOSTICS_DIR / 'oaxaca_blinder_detailed.csv'}")

    # Save DiD decomposition
    if "price_did" in did_result:
        did_summary = pd.DataFrame([{
            "analysis": "did_decomposition",
            "price_did": did_result["price_did"],
            **{f"cov_did_{k}": v for k, v in did_result.get("covariate_dids", {}).items()}
        }])
        did_summary.to_csv(DIAGNOSTICS_DIR / "did_decomposition.csv", index=False)
        print(f"Saved: {DIAGNOSTICS_DIR / 'did_decomposition.csv'}")

    # Interpretation
    print("\n" + "="*70)
    print("INTERPRETATION")
    print("="*70)

    if "pct_explained" in result and not np.isnan(result["pct_explained"]):
        if result["pct_explained"] > 50:
            print(f"""
FINDING: {result['pct_explained']:.1f}% of the positive price effect is explained
by observable property characteristics (composition effect).

This SUPPORTS the 'Freeze and Rebuild' hypothesis:
- Post-flood sales inside the flood zone are compositionally different
- Newer, potentially rebuilt properties dominate inside sales
- The price increase reflects WHAT is selling, not true appreciation

IMPLICATION: The positive price effect is largely a selection/composition artifact,
not evidence that flood exposure increases property values.
""")
        else:
            print(f"""
FINDING: Only {result['pct_explained']:.1f}% of the positive price effect is explained
by observable property characteristics.

{100 - result['pct_explained']:.1f}% remains unexplained, suggesting:
- Unobserved property quality improvements
- Actual capitalization effects (positive or negative)
- Measurement issues

IMPLICATION: Composition alone does not fully explain the counterintuitive finding.
Additional investigation needed.
""")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--caliper", "-c", type=int, default=300)
    args = parser.parse_args()
    main(caliper_m=args.caliper)
