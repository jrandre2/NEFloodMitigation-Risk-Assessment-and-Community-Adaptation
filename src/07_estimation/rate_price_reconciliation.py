"""
rate_price_reconciliation.py
=============================
Reconcile the apparent contradiction between sale rate decline (freeze)
and price increase (anti-flight) inside the flood zone.

Key Questions:
1. Do sale rates truly decline inside? (Confirms "freeze")
2. Is the price effect conditional on sale selection?
3. Are surviving sales higher quality properties?
4. What does a selection/truncation model reveal?

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


def load_panel_data(caliper_m: int = 300) -> pd.DataFrame:
    """Load parcel-month panel data for analysis."""
    # Try enriched panel first
    panel_path = DATA_WORK / "panel_parcel_month_enriched.parquet"

    if not panel_path.exists():
        panel_path = DATA_WORK / "panel_parcel_month.parquet"

    if not panel_path.exists():
        print(f"Warning: Panel data not found: {panel_path}")
        return pd.DataFrame()

    df = pd.read_parquet(panel_path)
    print(f"Loaded panel data: {len(df):,} records")

    # Filter to caliper window around inundation boundary
    if "signed_dist_inund_m" in df.columns:
        df = df[df["signed_dist_inund_m"].abs() <= caliper_m].copy()
        print(f"After caliper filter ({caliper_m}m): {len(df):,} records")
    elif "dist_to_inund_m" in df.columns:
        df = df[df["dist_to_inund_m"].abs() <= caliper_m].copy()

    return df


def load_sales_data(caliper_m: int = 300) -> pd.DataFrame:
    """Load sales transaction data."""
    sales_path = DATA_WORK / "sales_clean.parquet"

    if not sales_path.exists():
        # Try alternative path
        sales_path = DATA_WORK / "sales_panel.parquet"

    if not sales_path.exists():
        raise FileNotFoundError(f"Sales data not found in {DATA_WORK}")

    df = pd.read_parquet(sales_path)
    print(f"Loaded sales data: {len(df):,} transactions")

    # Merge with boundary distances if needed
    if "signed_dist_inund_m" not in df.columns:
        distances_path = DATA_WORK / "parcel_boundary_distances.parquet"
        if distances_path.exists():
            distances = pd.read_parquet(distances_path)
            df = df.merge(
                distances[["parcel_id", "signed_dist_inund_m", "inside_inund"]],
                on="parcel_id",
                how="inner"
            )
            print(f"After merge with distances: {len(df):,} transactions")

    # Filter to caliper window
    if "signed_dist_inund_m" in df.columns:
        df = df[df["signed_dist_inund_m"].abs() <= caliper_m].copy()
        print(f"After caliper filter ({caliper_m}m): {len(df):,} transactions")

    return df


def confirm_sale_rate_decline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Confirm that sale rates declined inside the flood zone post-event.
    This establishes the "freeze" component of the narrative.
    """
    print("\n" + "="*60)
    print("CONFIRMING SALE RATE DECLINE ('FREEZE')")
    print("="*60)

    # Ensure we have the needed columns
    if "inside_inund" not in df.columns:
        if "signed_dist_inund_m" in df.columns:
            df["inside_inund"] = df["signed_dist_inund_m"] < 0
        else:
            raise ValueError("Cannot determine inside/outside status")

    if "post" not in df.columns:
        if "sale_date" in df.columns:
            df["post"] = pd.to_datetime(df["sale_date"]) >= EVENT_DATE
        elif "ym" in df.columns:
            df["post"] = pd.to_datetime(df["ym"]) >= EVENT_DATE
        else:
            raise ValueError("Cannot determine pre/post status")

    # For panel data: calculate parcel-month sale rates
    if "sold_this_month" in df.columns:
        # This is panel data
        results = []

        for inside in [True, False]:
            for post in [True, False]:
                mask = (df["inside_inund"] == inside) & (df["post"] == post)
                subset = df[mask]

                n_parcel_months = len(subset)
                n_sales = subset["sold_this_month"].sum()
                rate = n_sales / n_parcel_months if n_parcel_months > 0 else 0

                results.append({
                    "inside": inside,
                    "post": post,
                    "n_parcel_months": n_parcel_months,
                    "n_sales": n_sales,
                    "sale_rate": rate,
                    "sale_rate_pct": rate * 100
                })

        results_df = pd.DataFrame(results)

    else:
        # This is sales data - count by group
        results = []

        for inside in [True, False]:
            for post in [True, False]:
                mask = (df["inside_inund"] == inside) & (df["post"] == post)
                n_sales = mask.sum()

                results.append({
                    "inside": inside,
                    "post": post,
                    "n_sales": n_sales
                })

        results_df = pd.DataFrame(results)

    # Calculate DiD for sale rates
    if "sale_rate" in results_df.columns:
        inside_pre = results_df[(results_df["inside"]) & (~results_df["post"])]["sale_rate"].values[0]
        inside_post = results_df[(results_df["inside"]) & (results_df["post"])]["sale_rate"].values[0]
        outside_pre = results_df[(~results_df["inside"]) & (~results_df["post"])]["sale_rate"].values[0]
        outside_post = results_df[(~results_df["inside"]) & (results_df["post"])]["sale_rate"].values[0]

        # DiD
        inside_change = inside_post - inside_pre
        outside_change = outside_post - outside_pre
        did_rate = inside_change - outside_change

        # Percent change
        inside_pct_change = (inside_post - inside_pre) / inside_pre * 100 if inside_pre > 0 else np.nan
        outside_pct_change = (outside_post - outside_pre) / outside_pre * 100 if outside_pre > 0 else np.nan

        print(f"\nSale Rates (per parcel-month):")
        print(f"  Inside zone, pre-flood:  {inside_pre*100:.4f}%")
        print(f"  Inside zone, post-flood: {inside_post*100:.4f}%")
        print(f"  Inside change: {inside_pct_change:+.1f}%")
        print(f"\n  Outside zone, pre-flood:  {outside_pre*100:.4f}%")
        print(f"  Outside zone, post-flood: {outside_post*100:.4f}%")
        print(f"  Outside change: {outside_pct_change:+.1f}%")
        print(f"\nDiD estimate (sale rate): {did_rate*100:.4f} pp")

        results_df["did_estimate"] = did_rate

        # Interpretation
        if did_rate < 0:
            print("\n=> CONFIRMED: Sale rates fell more inside than outside (FREEZE)")
        else:
            print("\n=> NOT CONFIRMED: Sale rates did not fall more inside")

    print("\nDetailed results:")
    print(results_df.to_string(index=False))

    return results_df


def analyze_conditional_prices(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze prices conditional on sale occurring.
    Key question: Are sold properties systematically different post-flood?
    """
    print("\n" + "="*60)
    print("ANALYZING PRICES CONDITIONAL ON SALE")
    print("="*60)

    # Filter to actual sales
    if "sold_this_month" in df.columns:
        sales = df[df["sold_this_month"] == True].copy()
    else:
        sales = df.copy()

    # Ensure we have price data
    if "log_price" not in sales.columns and "sale_price" in sales.columns:
        sales["log_price"] = np.log(sales["sale_price"].clip(lower=1))

    if "log_price" not in sales.columns:
        print("No price data available for analysis")
        return pd.DataFrame()

    # Calculate means by group
    results = []
    for inside in [True, False]:
        for post in [True, False]:
            mask = (sales["inside_inund"] == inside) & (sales["post"] == post)
            subset = sales[mask]

            if len(subset) > 0:
                results.append({
                    "inside": inside,
                    "post": post,
                    "n_sales": len(subset),
                    "mean_log_price": subset["log_price"].mean(),
                    "mean_price": np.exp(subset["log_price"]).mean(),
                    "median_price": np.exp(subset["log_price"]).median(),
                    "std_log_price": subset["log_price"].std()
                })

    results_df = pd.DataFrame(results)

    if len(results_df) == 4:
        # Calculate DiD
        inside_pre = results_df[(results_df["inside"]) & (~results_df["post"])]["mean_log_price"].values[0]
        inside_post = results_df[(results_df["inside"]) & (results_df["post"])]["mean_log_price"].values[0]
        outside_pre = results_df[(~results_df["inside"]) & (~results_df["post"])]["mean_log_price"].values[0]
        outside_post = results_df[(~results_df["inside"]) & (results_df["post"])]["mean_log_price"].values[0]

        did_price = (inside_post - inside_pre) - (outside_post - outside_pre)

        print(f"\nMean Log Prices:")
        print(f"  Inside pre:  {inside_pre:.3f} (${np.exp(inside_pre):,.0f})")
        print(f"  Inside post: {inside_post:.3f} (${np.exp(inside_post):,.0f})")
        print(f"  Outside pre:  {outside_pre:.3f} (${np.exp(outside_pre):,.0f})")
        print(f"  Outside post: {outside_post:.3f} (${np.exp(outside_post):,.0f})")
        print(f"\nDiD (log-price): {did_price:.4f}")
        print(f"  => {(np.exp(did_price)-1)*100:+.1f}% price change inside vs outside")

        if did_price > 0:
            print("\n=> Prices rose faster inside (contradicts flight)")
        else:
            print("\n=> Prices fell inside relative to outside (supports flight)")

    return results_df


def compare_sold_vs_available(panel_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compare characteristics of sold properties vs available properties.
    Tests if selection on observables explains price effect.
    """
    print("\n" + "="*60)
    print("COMPARING SOLD VS AVAILABLE PROPERTIES")
    print("="*60)

    # Covariates to compare
    covariates = [
        "log_assessed_value", "log_impsf", "building_age",
        "log_acres", "elevation_m"
    ]

    # Filter to available covariates
    available_covs = [c for c in covariates if c in panel_df.columns]

    if not available_covs:
        print("No covariates available for comparison")
        return pd.DataFrame()

    results = []

    # Post-flood, inside zone
    post_inside = panel_df[(panel_df["post"]) & (panel_df["inside_inund"])]

    if "sold_this_month" in post_inside.columns:
        sold = post_inside[post_inside["sold_this_month"] == True]
        not_sold = post_inside[post_inside["sold_this_month"] == False]

        for cov in available_covs:
            if cov in sold.columns:
                sold_mean = sold[cov].dropna().mean()
                not_sold_mean = not_sold[cov].dropna().mean()

                # T-test
                sold_vals = sold[cov].dropna()
                not_sold_vals = not_sold[cov].dropna()

                if len(sold_vals) > 1 and len(not_sold_vals) > 1:
                    tstat, pval = stats.ttest_ind(sold_vals, not_sold_vals)
                else:
                    tstat, pval = np.nan, np.nan

                diff = sold_mean - not_sold_mean
                diff_pct = diff / not_sold_mean * 100 if not_sold_mean != 0 else np.nan

                results.append({
                    "covariate": cov,
                    "period": "post_inside",
                    "sold_mean": sold_mean,
                    "not_sold_mean": not_sold_mean,
                    "difference": diff,
                    "diff_pct": diff_pct,
                    "tstat": tstat,
                    "pval": pval,
                    "n_sold": len(sold_vals),
                    "n_not_sold": len(not_sold_vals)
                })

    results_df = pd.DataFrame(results)

    if len(results_df) > 0:
        print("\nPost-flood, Inside Zone: Sold vs Not Sold")
        print("-" * 60)
        for _, row in results_df.iterrows():
            sig = "***" if row["pval"] < 0.01 else "**" if row["pval"] < 0.05 else "*" if row["pval"] < 0.1 else ""
            print(f"  {row['covariate']}: sold={row['sold_mean']:.2f}, not_sold={row['not_sold_mean']:.2f}, "
                  f"diff={row['diff_pct']:+.1f}% {sig}")

    return results_df


def estimate_selection_model(panel_df: pd.DataFrame) -> dict:
    """
    Estimate a selection model to understand what drives sale probability.
    Two-stage:
    1. Probit for sale probability
    2. Corrected price equation
    """
    print("\n" + "="*60)
    print("ESTIMATING SELECTION MODEL")
    print("="*60)

    # Covariates for selection equation
    selection_vars = ["log_assessed_value", "building_age", "log_acres"]

    # Filter to available
    available = [v for v in selection_vars if v in panel_df.columns]

    if not available:
        print("Insufficient covariates for selection model")
        return {}

    # Prepare data
    df = panel_df.dropna(subset=available + ["sold_this_month", "inside_inund", "post"])

    # Create interaction
    df["inside_x_post"] = df["inside_inund"].astype(int) * df["post"].astype(int)

    # Stage 1: Probit for sale probability
    X_sel = df[available + ["inside_inund", "post", "inside_x_post"]].copy()
    X_sel["inside_inund"] = X_sel["inside_inund"].astype(int)
    X_sel["post"] = X_sel["post"].astype(int)
    X_sel = sm.add_constant(X_sel)

    y_sel = df["sold_this_month"].astype(int)

    try:
        probit_model = sm.Probit(y_sel, X_sel)
        probit_result = probit_model.fit(disp=0, maxiter=100)

        print("\nSelection Equation (Probit):")
        print("-" * 40)
        print(f"  N = {len(df):,}")
        print(f"  Pseudo R² = {probit_result.prsquared:.3f}")

        print("\n  Coefficients:")
        for var in probit_result.params.index:
            coef = probit_result.params[var]
            se = probit_result.bse[var]
            pval = probit_result.pvalues[var]
            sig = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.1 else ""
            print(f"    {var}: {coef:.4f} (SE: {se:.4f}) {sig}")

        # Key result: inside_x_post effect on sale probability
        if "inside_x_post" in probit_result.params:
            coef = probit_result.params["inside_x_post"]
            pval = probit_result.pvalues["inside_x_post"]
            print(f"\n  Inside × Post effect: {coef:.4f} (p={pval:.3f})")

            if coef < 0:
                print("  => Confirms: Sale probability fell inside post-flood")
            else:
                print("  => Does not confirm sale probability decline")

        return {
            "probit_params": probit_result.params.to_dict(),
            "probit_pvalues": probit_result.pvalues.to_dict(),
            "probit_pseudo_r2": probit_result.prsquared,
            "n_obs": len(df)
        }

    except Exception as e:
        print(f"Selection model failed: {e}")
        return {}


def test_quality_selection(sales_df: pd.DataFrame) -> pd.DataFrame:
    """
    Test if post-flood sales inside the zone are higher quality.
    This could explain higher prices despite fewer sales.
    """
    print("\n" + "="*60)
    print("TESTING FOR QUALITY SELECTION IN SURVIVING SALES")
    print("="*60)

    # Quality indicators
    quality_vars = [
        "log_assessed_value", "log_impsf", "building_age",
        "elevation_m", "log_acres"
    ]

    available = [v for v in quality_vars if v in sales_df.columns]

    if not available:
        print("No quality variables available")
        return pd.DataFrame()

    results = []

    # Compare inside zone: pre vs post
    inside = sales_df[sales_df["inside_inund"] == True]
    inside_pre = inside[inside["post"] == False]
    inside_post = inside[inside["post"] == True]

    print("\nInside Zone: Pre vs Post Flood Sales")
    print("-" * 50)

    for var in available:
        pre_vals = inside_pre[var].dropna()
        post_vals = inside_post[var].dropna()

        if len(pre_vals) > 1 and len(post_vals) > 1:
            pre_mean = pre_vals.mean()
            post_mean = post_vals.mean()
            diff = post_mean - pre_mean
            diff_pct = diff / pre_mean * 100 if pre_mean != 0 else np.nan

            tstat, pval = stats.ttest_ind(pre_vals, post_vals)

            sig = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.1 else ""
            print(f"  {var}: pre={pre_mean:.2f}, post={post_mean:.2f}, "
                  f"change={diff_pct:+.1f}% {sig}")

            results.append({
                "variable": var,
                "zone": "inside",
                "pre_mean": pre_mean,
                "post_mean": post_mean,
                "difference": diff,
                "diff_pct": diff_pct,
                "tstat": tstat,
                "pval": pval,
                "n_pre": len(pre_vals),
                "n_post": len(post_vals)
            })

    # Also compare outside zone as control
    outside = sales_df[sales_df["inside_inund"] == False]
    outside_pre = outside[outside["post"] == False]
    outside_post = outside[outside["post"] == True]

    print("\nOutside Zone: Pre vs Post Flood Sales (Control)")
    print("-" * 50)

    for var in available:
        pre_vals = outside_pre[var].dropna()
        post_vals = outside_post[var].dropna()

        if len(pre_vals) > 1 and len(post_vals) > 1:
            pre_mean = pre_vals.mean()
            post_mean = post_vals.mean()
            diff = post_mean - pre_mean
            diff_pct = diff / pre_mean * 100 if pre_mean != 0 else np.nan

            tstat, pval = stats.ttest_ind(pre_vals, post_vals)

            sig = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.1 else ""
            print(f"  {var}: pre={pre_mean:.2f}, post={post_mean:.2f}, "
                  f"change={diff_pct:+.1f}% {sig}")

            results.append({
                "variable": var,
                "zone": "outside",
                "pre_mean": pre_mean,
                "post_mean": post_mean,
                "difference": diff,
                "diff_pct": diff_pct,
                "tstat": tstat,
                "pval": pval,
                "n_pre": len(pre_vals),
                "n_post": len(post_vals)
            })

    return pd.DataFrame(results)


def create_reconciliation_summary(
    rate_results: pd.DataFrame,
    price_results: pd.DataFrame,
    quality_results: pd.DataFrame
) -> pd.DataFrame:
    """
    Create a summary of the rate-price reconciliation analysis.
    """
    print("\n" + "="*60)
    print("RECONCILIATION SUMMARY")
    print("="*60)

    summary = []

    # Sale rate finding
    if "did_estimate" in rate_results.columns:
        rate_did = rate_results["did_estimate"].iloc[0]
        summary.append({
            "finding": "Sale Rate DiD",
            "estimate": rate_did,
            "interpretation": "FREEZE confirmed" if rate_did < 0 else "No freeze",
            "supports_narrative": "Yes" if rate_did < 0 else "No"
        })

    # Price finding (would need to be passed in)
    # For now, note the direction
    summary.append({
        "finding": "Price DiD",
        "estimate": np.nan,  # Would be calculated
        "interpretation": "See price results",
        "supports_narrative": "Requires investigation"
    })

    # Quality selection
    if len(quality_results) > 0:
        inside_quality = quality_results[quality_results["zone"] == "inside"]
        sig_changes = inside_quality[inside_quality["pval"] < 0.1]

        if len(sig_changes) > 0:
            quality_shifts = sig_changes["variable"].tolist()
            summary.append({
                "finding": "Quality Selection",
                "estimate": len(sig_changes),
                "interpretation": f"Significant shifts in: {', '.join(quality_shifts)}",
                "supports_narrative": "Possible explanation"
            })
        else:
            summary.append({
                "finding": "Quality Selection",
                "estimate": 0,
                "interpretation": "No significant quality shifts",
                "supports_narrative": "Does not explain"
            })

    summary_df = pd.DataFrame(summary)

    print("\nKey Findings:")
    print("-" * 60)
    for _, row in summary_df.iterrows():
        print(f"  {row['finding']}: {row['interpretation']}")

    print("\n" + "="*60)
    print("INTERPRETATION")
    print("="*60)
    print("""
If sale rates declined inside (FREEZE confirmed) but prices rose:

1. SELECTION EXPLANATION: Only best properties sell
   - Higher quality homes still transact
   - Distressed/damaged properties withdraw from market
   - Observed prices biased upward by selection

2. COMPOSITION EXPLANATION: Different buyers
   - Investors buying flood-zone properties at "discount"
   - But these are still higher-quality properties
   - Insurance settlements fund repairs/improvements

3. MARKET THINNING: Fewer comparable sales
   - Appraisers use older/distant comps
   - Price discovery impaired
   - Measured prices may not reflect true values

4. TIMING EFFECT: Early post-flood sales
   - First sales may be already-contracted
   - Or estate sales unrelated to flood
   - True effect may emerge later

RECOMMENDATION: Focus manuscript on FREEZE (sale rate decline)
Report price effects with selection caveat in supplement.
""")

    return summary_df


def main(caliper_m: int = 300):
    """Main function to run rate-price reconciliation analysis."""

    print("="*70)
    print("RATE-PRICE RECONCILIATION ANALYSIS")
    print(f"Caliper: ±{caliper_m}m from inundation boundary")
    print("="*70)

    # Ensure output directory exists
    DIAGNOSTICS_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    try:
        panel_df = load_panel_data(caliper_m)
        print(f"\nLoaded panel data: {len(panel_df):,} parcel-months")
    except FileNotFoundError as e:
        print(f"Warning: {e}")
        panel_df = None

    try:
        sales_df = load_sales_data(caliper_m)
        print(f"Loaded sales data: {len(sales_df):,} transactions")
    except FileNotFoundError as e:
        print(f"Warning: {e}")
        sales_df = None

    if panel_df is None and sales_df is None:
        print("\nNo data available for analysis")
        return

    # Analysis 1: Confirm sale rate decline
    if panel_df is not None:
        rate_results = confirm_sale_rate_decline(panel_df)
        rate_results.to_csv(DIAGNOSTICS_DIR / "rate_vs_price_comparison.csv", index=False)
    else:
        rate_results = pd.DataFrame()

    # Analysis 2: Conditional price analysis
    if sales_df is not None:
        price_results = analyze_conditional_prices(sales_df)
        price_results.to_csv(DIAGNOSTICS_DIR / "conditional_price_analysis.csv", index=False)
    elif panel_df is not None:
        price_results = analyze_conditional_prices(panel_df)
        price_results.to_csv(DIAGNOSTICS_DIR / "conditional_price_analysis.csv", index=False)
    else:
        price_results = pd.DataFrame()

    # Analysis 3: Compare sold vs available
    if panel_df is not None:
        selection_comparison = compare_sold_vs_available(panel_df)
        if len(selection_comparison) > 0:
            selection_comparison.to_csv(
                DIAGNOSTICS_DIR / "sold_vs_available_comparison.csv", index=False
            )

    # Analysis 4: Selection model
    if panel_df is not None:
        selection_model = estimate_selection_model(panel_df)

    # Analysis 5: Quality selection in sales
    if sales_df is not None:
        quality_results = test_quality_selection(sales_df)
        quality_results.to_csv(DIAGNOSTICS_DIR / "quality_selection_test.csv", index=False)
    elif panel_df is not None:
        quality_results = test_quality_selection(panel_df)
        quality_results.to_csv(DIAGNOSTICS_DIR / "quality_selection_test.csv", index=False)
    else:
        quality_results = pd.DataFrame()

    # Summary
    summary = create_reconciliation_summary(rate_results, price_results, quality_results)
    summary.to_csv(DIAGNOSTICS_DIR / "rate_price_reconciliation_summary.csv", index=False)

    print(f"\nResults saved to {DIAGNOSTICS_DIR}/")
    print("  - rate_vs_price_comparison.csv")
    print("  - conditional_price_analysis.csv")
    print("  - sold_vs_available_comparison.csv")
    print("  - quality_selection_test.csv")
    print("  - rate_price_reconciliation_summary.csv")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Reconcile sale rate and price effects"
    )
    parser.add_argument(
        "-c", "--caliper",
        type=int,
        default=300,
        help="Caliper window in meters (default: 300)"
    )

    args = parser.parse_args()
    main(caliper_m=args.caliper)
