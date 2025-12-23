"""
seller_selection_analysis.py
============================
Analyze who sells post-flood inside the flood zone.

Key Questions:
1. What characteristics predict selling post-flood?
2. Are distressed properties more or less likely to sell?
3. How does propensity to sell differ inside vs outside?
4. What does selection on unobservables look like?

Author: Claude Code
Date: 2025-12-23
"""

import pandas as pd
import numpy as np
from pathlib import Path
import statsmodels.api as sm
from scipy import stats
import warnings

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_WORK = PROJECT_ROOT / "data_work"
DIAGNOSTICS_DIR = DATA_WORK / "diagnostics"
FIGURES_DIR = PROJECT_ROOT / "figures"

# Event date
EVENT_DATE = pd.Timestamp("2019-03-01")


def load_data(caliper_m: int = 300) -> pd.DataFrame:
    """Load parcel-level data with sale indicators."""

    # Try enriched panel data first (has boundary distances)
    panel_path = DATA_WORK / "panel_parcel_month_enriched.parquet"
    if panel_path.exists():
        df = pd.read_parquet(panel_path)
        print(f"Loaded from enriched panel: {len(df):,} records")

        # Filter to caliper window
        if "signed_dist_inund_m" in df.columns:
            df = df[df["signed_dist_inund_m"].abs() <= caliper_m]
            print(f"After caliper filter ({caliper_m}m): {len(df):,} records")

        return df

    # Try basic panel
    panel_path = DATA_WORK / "panel_parcel_month.parquet"
    if panel_path.exists():
        df = pd.read_parquet(panel_path)
        print(f"Loaded from panel: {len(df):,} records")

        # Merge with boundary distances
        distances_path = DATA_WORK / "parcel_boundary_distances.parquet"
        if distances_path.exists():
            distances = pd.read_parquet(distances_path)
            df = df.merge(
                distances[["parcel_id", "signed_dist_inund_m", "inside_inund"]],
                on="parcel_id",
                how="inner"
            )
            print(f"After merge with distances: {len(df):,} records")

        # Filter to caliper window
        if "signed_dist_inund_m" in df.columns:
            df = df[df["signed_dist_inund_m"].abs() <= caliper_m]
            print(f"After caliper filter ({caliper_m}m): {len(df):,} records")

        return df

    # Fallback to sales data
    sales_path = DATA_WORK / "sales_clean.parquet"
    if sales_path.exists():
        df = pd.read_parquet(sales_path)
        print(f"Loaded from sales_clean: {len(df):,} records")

        # Merge with boundary distances
        distances_path = DATA_WORK / "parcel_boundary_distances.parquet"
        if distances_path.exists():
            distances = pd.read_parquet(distances_path)
            df = df.merge(
                distances[["parcel_id", "signed_dist_inund_m", "inside_inund"]],
                on="parcel_id",
                how="inner"
            )
            print(f"After merge with distances: {len(df):,} records")

        if "signed_dist_inund_m" in df.columns:
            df = df[df["signed_dist_inund_m"].abs() <= caliper_m]
            print(f"After caliper filter ({caliper_m}m): {len(df):,} records")

        return df

    raise FileNotFoundError("No suitable data found")


def create_seller_indicator(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create indicator for whether a parcel sold in post-period.
    Aggregates panel to parcel-level.
    """
    print("\n" + "="*60)
    print("CREATING SELLER INDICATORS")
    print("="*60)

    if "parcel_id" not in df.columns:
        print("No parcel_id column - cannot aggregate")
        return df

    # Identify post-period sales
    if "sold_this_month" in df.columns and "post" in df.columns:
        # Panel data: aggregate to parcel level
        post_df = df[df["post"] == True]

        parcel_sales = post_df.groupby("parcel_id").agg({
            "sold_this_month": "sum"
        }).reset_index()
        parcel_sales.columns = ["parcel_id", "n_sales_post"]
        parcel_sales["sold_post"] = parcel_sales["n_sales_post"] > 0

        # Get parcel characteristics (first observation per parcel)
        parcel_chars = df.groupby("parcel_id").first().reset_index()

        # Merge
        result = parcel_chars.merge(parcel_sales, on="parcel_id", how="left")
        result["n_sales_post"] = result["n_sales_post"].fillna(0)
        result["sold_post"] = result["sold_post"].fillna(False)

        print(f"Created parcel-level data: {len(result):,} parcels")
        print(f"  Sold in post-period: {result['sold_post'].sum():,} ({result['sold_post'].mean()*100:.1f}%)")

        return result

    return df


def estimate_propensity_model(df: pd.DataFrame) -> dict:
    """
    Estimate propensity to sell as function of characteristics.
    Uses probit/logit model.
    """
    print("\n" + "="*60)
    print("PROPENSITY TO SELL MODEL")
    print("="*60)

    if "sold_post" not in df.columns:
        print("No sold_post indicator - run create_seller_indicator first")
        return {}

    # Covariates for propensity model
    covariates = [
        "log_assessed_value", "building_age", "log_impsf",
        "log_acres", "elevation_m", "inside_inund"
    ]

    # Filter to available
    available = [c for c in covariates if c in df.columns]

    if len(available) < 2:
        print(f"Insufficient covariates. Available: {available}")
        return {}

    # Prepare data
    model_df = df.dropna(subset=available + ["sold_post"]).copy()

    # Ensure inside_inund is numeric
    if "inside_inund" in model_df.columns:
        model_df["inside_inund"] = model_df["inside_inund"].astype(int)

    X = sm.add_constant(model_df[available])
    y = model_df["sold_post"].astype(int)

    try:
        # Logit model
        logit_model = sm.Logit(y, X)
        logit_result = logit_model.fit(disp=0, maxiter=100)

        print(f"\nLogit Model for P(Sell in Post-Period)")
        print("-" * 50)
        print(f"N = {len(model_df):,}")
        print(f"Pseudo R² = {logit_result.prsquared:.4f}")
        print(f"\nCoefficients:")

        results_list = []
        for var in logit_result.params.index:
            coef = logit_result.params[var]
            se = logit_result.bse[var]
            pval = logit_result.pvalues[var]
            odds_ratio = np.exp(coef)

            sig = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.1 else ""
            print(f"  {var}: {coef:.4f} (SE: {se:.4f}), OR: {odds_ratio:.3f} {sig}")

            results_list.append({
                "variable": var,
                "coefficient": coef,
                "std_error": se,
                "odds_ratio": odds_ratio,
                "pvalue": pval
            })

        # Key finding: inside_inund effect
        if "inside_inund" in logit_result.params:
            inside_coef = logit_result.params["inside_inund"]
            inside_or = np.exp(inside_coef)
            inside_pval = logit_result.pvalues["inside_inund"]

            print(f"\n*** KEY RESULT ***")
            print(f"Inside flood zone effect on selling:")
            print(f"  Coefficient: {inside_coef:.4f}")
            print(f"  Odds Ratio: {inside_or:.3f}")
            print(f"  P-value: {inside_pval:.4f}")

            if inside_coef < 0:
                print(f"  => Parcels inside are {(1-inside_or)*100:.1f}% LESS likely to sell")
            else:
                print(f"  => Parcels inside are {(inside_or-1)*100:.1f}% MORE likely to sell")

        return {
            "model": "logit",
            "n_obs": len(model_df),
            "pseudo_r2": logit_result.prsquared,
            "results": pd.DataFrame(results_list)
        }

    except Exception as e:
        print(f"Model estimation failed: {e}")
        return {}


def compare_sellers_vs_holders(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compare characteristics of those who sold vs those who held.
    """
    print("\n" + "="*60)
    print("SELLERS VS HOLDERS COMPARISON")
    print("="*60)

    if "sold_post" not in df.columns:
        print("No sold_post indicator")
        return pd.DataFrame()

    # Covariates to compare
    covariates = [
        "log_assessed_value", "building_age", "log_impsf",
        "log_acres", "elevation_m"
    ]

    available = [c for c in covariates if c in df.columns]

    results = []

    # Compare inside flood zone only
    inside = df[df["inside_inund"] == True]
    sellers = inside[inside["sold_post"] == True]
    holders = inside[inside["sold_post"] == False]

    print(f"\nInside Flood Zone:")
    print(f"  Sellers: {len(sellers):,}")
    print(f"  Holders: {len(holders):,}")
    print("-" * 50)

    for cov in available:
        seller_vals = sellers[cov].dropna()
        holder_vals = holders[cov].dropna()

        if len(seller_vals) > 1 and len(holder_vals) > 1:
            seller_mean = seller_vals.mean()
            holder_mean = holder_vals.mean()
            diff = seller_mean - holder_mean
            diff_pct = diff / holder_mean * 100 if holder_mean != 0 else np.nan

            tstat, pval = stats.ttest_ind(seller_vals, holder_vals)

            sig = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.1 else ""
            print(f"  {cov}: sellers={seller_mean:.2f}, holders={holder_mean:.2f}, "
                  f"diff={diff_pct:+.1f}% {sig}")

            results.append({
                "covariate": cov,
                "zone": "inside",
                "seller_mean": seller_mean,
                "holder_mean": holder_mean,
                "difference": diff,
                "diff_pct": diff_pct,
                "tstat": tstat,
                "pval": pval,
                "n_sellers": len(seller_vals),
                "n_holders": len(holder_vals)
            })

    # Also do outside for comparison
    outside = df[df["inside_inund"] == False]
    sellers_out = outside[outside["sold_post"] == True]
    holders_out = outside[outside["sold_post"] == False]

    print(f"\nOutside Flood Zone (Control):")
    print(f"  Sellers: {len(sellers_out):,}")
    print(f"  Holders: {len(holders_out):,}")
    print("-" * 50)

    for cov in available:
        seller_vals = sellers_out[cov].dropna()
        holder_vals = holders_out[cov].dropna()

        if len(seller_vals) > 1 and len(holder_vals) > 1:
            seller_mean = seller_vals.mean()
            holder_mean = holder_vals.mean()
            diff = seller_mean - holder_mean
            diff_pct = diff / holder_mean * 100 if holder_mean != 0 else np.nan

            tstat, pval = stats.ttest_ind(seller_vals, holder_vals)

            sig = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.1 else ""
            print(f"  {cov}: sellers={seller_mean:.2f}, holders={holder_mean:.2f}, "
                  f"diff={diff_pct:+.1f}% {sig}")

            results.append({
                "covariate": cov,
                "zone": "outside",
                "seller_mean": seller_mean,
                "holder_mean": holder_mean,
                "difference": diff,
                "diff_pct": diff_pct,
                "tstat": tstat,
                "pval": pval,
                "n_sellers": len(seller_vals),
                "n_holders": len(holder_vals)
            })

    return pd.DataFrame(results)


def test_distress_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Test if distress indicators predict selling.
    Distress proxies: low assessed value, old building, small lot
    """
    print("\n" + "="*60)
    print("DISTRESS INDICATOR ANALYSIS")
    print("="*60)

    if "sold_post" not in df.columns:
        return pd.DataFrame()

    # Create distress indicators (quartiles)
    distress_results = []

    # Low value = bottom quartile
    if "log_assessed_value" in df.columns:
        q25 = df["log_assessed_value"].quantile(0.25)
        df["low_value"] = df["log_assessed_value"] < q25

    # Old building = top quartile of age
    if "building_age" in df.columns:
        q75_age = df["building_age"].quantile(0.75)
        df["old_building"] = df["building_age"] > q75_age

    # Small lot = bottom quartile
    if "log_acres" in df.columns:
        q25_acres = df["log_acres"].quantile(0.25)
        df["small_lot"] = df["log_acres"] < q25_acres

    # Test each distress indicator
    distress_vars = ["low_value", "old_building", "small_lot"]
    available = [v for v in distress_vars if v in df.columns]

    inside = df[df["inside_inund"] == True]

    for var in available:
        distressed = inside[inside[var] == True]
        not_distressed = inside[inside[var] == False]

        sale_rate_distressed = distressed["sold_post"].mean()
        sale_rate_not_distressed = not_distressed["sold_post"].mean()

        diff = sale_rate_distressed - sale_rate_not_distressed

        # Chi-square test
        contingency = pd.crosstab(inside[var], inside["sold_post"])
        if contingency.shape == (2, 2):
            chi2, pval, _, _ = stats.chi2_contingency(contingency)
        else:
            chi2, pval = np.nan, np.nan

        sig = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.1 else ""

        print(f"\n{var}:")
        print(f"  Sale rate if {var}=True: {sale_rate_distressed*100:.2f}%")
        print(f"  Sale rate if {var}=False: {sale_rate_not_distressed*100:.2f}%")
        print(f"  Difference: {diff*100:+.2f} pp {sig}")

        distress_results.append({
            "indicator": var,
            "sale_rate_distressed": sale_rate_distressed,
            "sale_rate_not_distressed": sale_rate_not_distressed,
            "difference_pp": diff * 100,
            "chi2": chi2,
            "pval": pval,
            "n_distressed": len(distressed),
            "n_not_distressed": len(not_distressed)
        })

    return pd.DataFrame(distress_results)


def selection_difference_in_differences(df: pd.DataFrame) -> pd.DataFrame:
    """
    Test if selection on observables differs between inside/outside.
    DiD on the relationship between characteristics and selling.
    """
    print("\n" + "="*60)
    print("SELECTION DIFFERENCE-IN-DIFFERENCES")
    print("="*60)

    # This tests whether the RELATIONSHIP between characteristics
    # and selling changed differentially inside vs outside

    covariates = ["log_assessed_value", "building_age", "log_impsf"]
    available = [c for c in covariates if c in df.columns]

    if not available:
        return pd.DataFrame()

    results = []

    for cov in available:
        # Interaction model: sold_post ~ cov + inside + cov*inside
        model_df = df.dropna(subset=[cov, "sold_post", "inside_inund"]).copy()
        model_df["inside_inund"] = model_df["inside_inund"].astype(int)
        model_df[f"{cov}_x_inside"] = model_df[cov] * model_df["inside_inund"]

        X = sm.add_constant(model_df[[cov, "inside_inund", f"{cov}_x_inside"]])
        y = model_df["sold_post"].astype(int)

        try:
            model = sm.Logit(y, X)
            result = model.fit(disp=0, maxiter=100)

            interaction_coef = result.params[f"{cov}_x_inside"]
            interaction_pval = result.pvalues[f"{cov}_x_inside"]

            sig = "***" if interaction_pval < 0.01 else "**" if interaction_pval < 0.05 else "*" if interaction_pval < 0.1 else ""

            print(f"\n{cov} × inside interaction:")
            print(f"  Coefficient: {interaction_coef:.4f} {sig}")
            print(f"  P-value: {interaction_pval:.4f}")

            if interaction_coef != 0 and interaction_pval < 0.1:
                if interaction_coef > 0:
                    print(f"  => {cov} predicts selling MORE strongly inside")
                else:
                    print(f"  => {cov} predicts selling LESS strongly inside")

            results.append({
                "covariate": cov,
                "interaction_coef": interaction_coef,
                "interaction_pval": interaction_pval,
                "main_effect_cov": result.params[cov],
                "main_effect_inside": result.params["inside_inund"],
                "n_obs": len(model_df)
            })

        except Exception as e:
            print(f"Model for {cov} failed: {e}")

    return pd.DataFrame(results)


def generate_selection_summary(
    propensity: dict,
    comparison: pd.DataFrame,
    distress: pd.DataFrame,
    selection_did: pd.DataFrame
) -> pd.DataFrame:
    """Generate summary of selection analysis findings."""

    print("\n" + "="*60)
    print("SELECTION ANALYSIS SUMMARY")
    print("="*60)

    findings = []

    # 1. Overall propensity
    if propensity and "results" in propensity:
        inside_result = propensity["results"][propensity["results"]["variable"] == "inside_inund"]
        if len(inside_result) > 0:
            or_val = inside_result["odds_ratio"].values[0]
            pval = inside_result["pvalue"].values[0]
            findings.append({
                "category": "Propensity to Sell",
                "finding": f"Inside zone odds ratio = {or_val:.3f}",
                "significant": pval < 0.05,
                "interpretation": "Less likely to sell inside" if or_val < 1 else "More likely to sell inside"
            })

    # 2. Seller vs holder differences
    if len(comparison) > 0:
        inside_comp = comparison[comparison["zone"] == "inside"]
        sig_diffs = inside_comp[inside_comp["pval"] < 0.1]

        if len(sig_diffs) > 0:
            for _, row in sig_diffs.iterrows():
                direction = "higher" if row["diff_pct"] > 0 else "lower"
                findings.append({
                    "category": "Seller Characteristics",
                    "finding": f"Sellers have {direction} {row['covariate']} ({row['diff_pct']:+.1f}%)",
                    "significant": row["pval"] < 0.05,
                    "interpretation": "Selection on observables"
                })

    # 3. Distress analysis
    if len(distress) > 0:
        for _, row in distress.iterrows():
            if row["pval"] < 0.1:
                direction = "more" if row["difference_pp"] > 0 else "less"
                findings.append({
                    "category": "Distress Selection",
                    "finding": f"{row['indicator']} properties {direction} likely to sell",
                    "significant": row["pval"] < 0.05,
                    "interpretation": "Distress affects selling"
                })

    # 4. Differential selection
    if len(selection_did) > 0:
        sig_did = selection_did[selection_did["interaction_pval"] < 0.1]
        if len(sig_did) > 0:
            for _, row in sig_did.iterrows():
                findings.append({
                    "category": "Differential Selection",
                    "finding": f"{row['covariate']} selection differs inside vs outside",
                    "significant": row["interaction_pval"] < 0.05,
                    "interpretation": "Selection mechanism changed"
                })

    findings_df = pd.DataFrame(findings)

    print("\nKey Findings:")
    print("-" * 50)
    for _, row in findings_df.iterrows():
        sig_marker = "***" if row["significant"] else ""
        print(f"  [{row['category']}] {row['finding']} {sig_marker}")
        print(f"    => {row['interpretation']}")

    return findings_df


def main(caliper_m: int = 300):
    """Main function to run seller selection analysis."""

    print("="*70)
    print("SELLER SELECTION ANALYSIS")
    print(f"Caliper: ±{caliper_m}m from inundation boundary")
    print("="*70)

    # Ensure output directory
    DIAGNOSTICS_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    try:
        df = load_data(caliper_m)
        print(f"\nLoaded data: {len(df):,} observations")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    # Create seller indicators
    parcel_df = create_seller_indicator(df)

    # Ensure inside_inund exists
    if "inside_inund" not in parcel_df.columns:
        if "signed_dist_inund_m" in parcel_df.columns:
            parcel_df["inside_inund"] = parcel_df["signed_dist_inund_m"] < 0
        else:
            print("Cannot determine inside/outside status")
            return

    # Analysis 1: Propensity model
    propensity = estimate_propensity_model(parcel_df)
    if propensity and "results" in propensity:
        propensity["results"].to_csv(
            DIAGNOSTICS_DIR / "seller_propensity_model.csv", index=False
        )

    # Analysis 2: Sellers vs holders comparison
    comparison = compare_sellers_vs_holders(parcel_df)
    if len(comparison) > 0:
        comparison.to_csv(
            DIAGNOSTICS_DIR / "sellers_vs_holders.csv", index=False
        )

    # Analysis 3: Distress indicators
    distress = test_distress_indicators(parcel_df)
    if len(distress) > 0:
        distress.to_csv(
            DIAGNOSTICS_DIR / "distress_selection.csv", index=False
        )

    # Analysis 4: Selection DiD
    selection_did = selection_difference_in_differences(parcel_df)
    if len(selection_did) > 0:
        selection_did.to_csv(
            DIAGNOSTICS_DIR / "selection_did.csv", index=False
        )

    # Summary
    summary = generate_selection_summary(propensity, comparison, distress, selection_did)
    summary.to_csv(DIAGNOSTICS_DIR / "seller_selection_summary.csv", index=False)

    print(f"\nResults saved to {DIAGNOSTICS_DIR}/")
    print("  - seller_propensity_model.csv")
    print("  - sellers_vs_holders.csv")
    print("  - distress_selection.csv")
    print("  - selection_did.csv")
    print("  - seller_selection_summary.csv")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Seller selection analysis")
    parser.add_argument(
        "-c", "--caliper",
        type=int,
        default=300,
        help="Caliper window in meters (default: 300)"
    )

    args = parser.parse_args()
    main(caliper_m=args.caliper)
