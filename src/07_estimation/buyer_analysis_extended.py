"""
buyer_analysis_extended.py
==========================
Extended analysis of buyer composition changes post-flood.

Key Questions:
1. How did buyer types change inside the flood zone?
2. Are investors (LLCs, portfolios) buying more/less?
3. Did financing patterns change (cash vs mortgage)?
4. What are buyer proximity patterns?

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
RESULTS_DIR = PROJECT_ROOT / "results" / "integration_run"

# Event date
EVENT_DATE = pd.Timestamp("2019-03-01")


def load_sales_with_buyers(caliper_m: int = 300) -> pd.DataFrame:
    """Load sales data with buyer classification."""

    # Try integration run results first (has owner classification)
    classified_path = RESULTS_DIR / "sfr_regression_data.csv"
    if classified_path.exists():
        df = pd.read_csv(classified_path)
        print(f"Loaded from integration results: {len(df):,} records")
    else:
        # Fallback to sales_clean
        sales_path = DATA_WORK / "sales_clean.parquet"
        if sales_path.exists():
            df = pd.read_parquet(sales_path)
            print(f"Loaded from sales_clean: {len(df):,} records")
        else:
            raise FileNotFoundError("No sales data found")

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
            print(f"After merge with distances: {len(df):,} records")

    print(f"Loaded data: {len(df):,} sales")

    # Filter to caliper window
    if "signed_dist_inund_m" in df.columns:
        df = df[df["signed_dist_inund_m"].abs() <= caliper_m]
        print(f"After caliper filter ({caliper_m}m): {len(df):,} sales")
    elif "dist_to_inund_m" in df.columns:
        df = df[df["dist_to_inund_m"].abs() <= caliper_m]

    # Ensure date column
    if "sale_date" in df.columns:
        df["sale_date"] = pd.to_datetime(df["sale_date"])
        df["post"] = df["sale_date"] >= EVENT_DATE
    elif "post" not in df.columns:
        raise ValueError("Cannot determine pre/post status")

    # Ensure inside/outside indicator
    if "inside_inund" not in df.columns:
        if "signed_dist_inund_m" in df.columns:
            df["inside_inund"] = df["signed_dist_inund_m"] < 0

    return df


def analyze_owner_form_changes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze changes in buyer organizational form.
    """
    print("\n" + "="*60)
    print("BUYER ORGANIZATIONAL FORM ANALYSIS")
    print("="*60)

    # Check for owner form column
    owner_cols = ["owner_form", "buyer_form", "buyer_type", "owner_type"]
    owner_col = None
    for col in owner_cols:
        if col in df.columns:
            owner_col = col
            break

    if owner_col is None:
        print("No owner form column found")
        print(f"Available columns: {list(df.columns)[:20]}...")
        return pd.DataFrame()

    print(f"\nUsing column: {owner_col}")
    print(f"Unique values: {df[owner_col].unique()}")

    # Create summary by period and zone
    results = []

    for inside in [True, False]:
        for post in [True, False]:
            mask = (df["inside_inund"] == inside) & (df["post"] == post)
            subset = df[mask]

            if len(subset) == 0:
                continue

            # Count by owner form
            form_counts = subset[owner_col].value_counts()
            form_pcts = subset[owner_col].value_counts(normalize=True) * 100

            for form, count in form_counts.items():
                results.append({
                    "inside": inside,
                    "post": post,
                    "owner_form": form,
                    "count": count,
                    "pct": form_pcts.get(form, 0),
                    "zone": "inside" if inside else "outside",
                    "period": "post" if post else "pre"
                })

    results_df = pd.DataFrame(results)

    if len(results_df) == 0:
        return results_df

    # Calculate DiD for each form
    print("\nOwner Form DiD Analysis:")
    print("-" * 50)

    forms = results_df["owner_form"].unique()
    did_results = []

    for form in forms:
        form_data = results_df[results_df["owner_form"] == form]

        try:
            inside_pre = form_data[(form_data["inside"]) & (~form_data["post"])]["pct"].values
            inside_post = form_data[(form_data["inside"]) & (form_data["post"])]["pct"].values
            outside_pre = form_data[(~form_data["inside"]) & (~form_data["post"])]["pct"].values
            outside_post = form_data[(~form_data["inside"]) & (form_data["post"])]["pct"].values

            if len(inside_pre) > 0 and len(inside_post) > 0 and len(outside_pre) > 0 and len(outside_post) > 0:
                inside_change = inside_post[0] - inside_pre[0]
                outside_change = outside_post[0] - outside_pre[0]
                did = inside_change - outside_change

                print(f"\n{form}:")
                print(f"  Inside: {inside_pre[0]:.1f}% → {inside_post[0]:.1f}% ({inside_change:+.1f}pp)")
                print(f"  Outside: {outside_pre[0]:.1f}% → {outside_post[0]:.1f}% ({outside_change:+.1f}pp)")
                print(f"  DiD: {did:+.1f}pp")

                did_results.append({
                    "owner_form": form,
                    "inside_pre": inside_pre[0],
                    "inside_post": inside_post[0],
                    "outside_pre": outside_pre[0],
                    "outside_post": outside_post[0],
                    "inside_change": inside_change,
                    "outside_change": outside_change,
                    "did_estimate": did
                })
        except Exception as e:
            print(f"Could not calculate DiD for {form}: {e}")

    did_df = pd.DataFrame(did_results)

    return did_df


def analyze_portfolio_buyers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze changes in portfolio/institutional buyer activity.
    """
    print("\n" + "="*60)
    print("PORTFOLIO BUYER ANALYSIS")
    print("="*60)

    # Check for portfolio indicators
    portfolio_cols = ["is_multi_parcel", "portfolio_size", "is_portfolio", "n_parcels"]
    portfolio_col = None
    for col in portfolio_cols:
        if col in df.columns:
            portfolio_col = col
            break

    if portfolio_col is None:
        print("No portfolio indicator found")
        # Try to infer from LLC status
        if "owner_form" in df.columns:
            df["is_institutional"] = df["owner_form"].isin(["LLC", "Corporation", "Corp"])
            portfolio_col = "is_institutional"
            print("Using LLC/Corporation as proxy for institutional")
        else:
            return pd.DataFrame()

    print(f"Using column: {portfolio_col}")

    results = []

    # Calculate portfolio share by group
    for inside in [True, False]:
        for post in [True, False]:
            mask = (df["inside_inund"] == inside) & (df["post"] == post)
            subset = df[mask]

            if len(subset) == 0:
                continue

            if portfolio_col in ["is_multi_parcel", "is_portfolio", "is_institutional"]:
                # Binary indicator
                portfolio_share = subset[portfolio_col].mean() * 100
                portfolio_count = subset[portfolio_col].sum()
            else:
                # Numeric (portfolio_size or n_parcels)
                portfolio_share = (subset[portfolio_col] > 1).mean() * 100
                portfolio_count = (subset[portfolio_col] > 1).sum()

            results.append({
                "inside": inside,
                "post": post,
                "portfolio_share_pct": portfolio_share,
                "portfolio_count": portfolio_count,
                "total_sales": len(subset),
                "zone": "inside" if inside else "outside",
                "period": "post" if post else "pre"
            })

    results_df = pd.DataFrame(results)

    # Calculate DiD
    if len(results_df) == 4:
        inside_pre = results_df[(results_df["inside"]) & (~results_df["post"])]["portfolio_share_pct"].values[0]
        inside_post = results_df[(results_df["inside"]) & (results_df["post"])]["portfolio_share_pct"].values[0]
        outside_pre = results_df[(~results_df["inside"]) & (~results_df["post"])]["portfolio_share_pct"].values[0]
        outside_post = results_df[(~results_df["inside"]) & (results_df["post"])]["portfolio_share_pct"].values[0]

        inside_change = inside_post - inside_pre
        outside_change = outside_post - outside_pre
        did = inside_change - outside_change

        print(f"\nPortfolio Buyer Share:")
        print(f"  Inside: {inside_pre:.1f}% → {inside_post:.1f}% ({inside_change:+.1f}pp)")
        print(f"  Outside: {outside_pre:.1f}% → {outside_post:.1f}% ({outside_change:+.1f}pp)")
        print(f"  DiD: {did:+.1f}pp")

        if did > 0:
            print("\n  => Institutional buyers INCREASED inside relative to outside")
        else:
            print("\n  => Institutional buyers DECREASED inside relative to outside")

    return results_df


def analyze_buyer_proximity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze changes in buyer proximity patterns.
    """
    print("\n" + "="*60)
    print("BUYER PROXIMITY ANALYSIS")
    print("="*60)

    # Check for proximity columns
    proximity_cols = ["owner_dist_km", "buyer_dist_km", "log_owner_dist_km"]
    prox_col = None
    for col in proximity_cols:
        if col in df.columns:
            prox_col = col
            break

    if prox_col is None:
        # Check for locality indicators
        local_cols = ["is_local_owner", "is_local", "is_in_county"]
        for col in local_cols:
            if col in df.columns:
                prox_col = col
                break

    if prox_col is None:
        print("No proximity data found")
        return pd.DataFrame()

    print(f"Using column: {prox_col}")

    results = []

    for inside in [True, False]:
        for post in [True, False]:
            mask = (df["inside_inund"] == inside) & (df["post"] == post)
            subset = df[mask]

            if len(subset) == 0:
                continue

            if prox_col in ["is_local_owner", "is_local", "is_in_county"]:
                # Binary local indicator
                local_share = subset[prox_col].mean() * 100
                local_count = subset[prox_col].sum()
                mean_dist = np.nan
            else:
                # Distance measure
                local_share = np.nan
                local_count = np.nan
                mean_dist = subset[prox_col].mean()

            results.append({
                "inside": inside,
                "post": post,
                "local_share_pct": local_share,
                "local_count": local_count,
                "mean_distance": mean_dist,
                "total_sales": len(subset),
                "zone": "inside" if inside else "outside",
                "period": "post" if post else "pre"
            })

    results_df = pd.DataFrame(results)

    # Report findings
    if len(results_df) == 4:
        metric = "local_share_pct" if not results_df["local_share_pct"].isna().all() else "mean_distance"

        inside_pre = results_df[(results_df["inside"]) & (~results_df["post"])][metric].values[0]
        inside_post = results_df[(results_df["inside"]) & (results_df["post"])][metric].values[0]
        outside_pre = results_df[(~results_df["inside"]) & (~results_df["post"])][metric].values[0]
        outside_post = results_df[(~results_df["inside"]) & (results_df["post"])][metric].values[0]

        inside_change = inside_post - inside_pre
        outside_change = outside_post - outside_pre
        did = inside_change - outside_change

        print(f"\n{metric}:")
        print(f"  Inside: {inside_pre:.1f} → {inside_post:.1f} ({inside_change:+.1f})")
        print(f"  Outside: {outside_pre:.1f} → {outside_post:.1f} ({outside_change:+.1f})")
        print(f"  DiD: {did:+.1f}")

    return results_df


def analyze_price_by_buyer_type(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze whether different buyer types pay different prices.
    """
    print("\n" + "="*60)
    print("PRICE BY BUYER TYPE ANALYSIS")
    print("="*60)

    # Need price and owner form
    if "log_price" not in df.columns:
        if "sale_price" in df.columns:
            df["log_price"] = np.log(df["sale_price"].clip(lower=1))
        else:
            print("No price data")
            return pd.DataFrame()

    owner_cols = ["owner_form", "buyer_form", "buyer_type"]
    owner_col = None
    for col in owner_cols:
        if col in df.columns:
            owner_col = col
            break

    if owner_col is None:
        print("No owner form column")
        return pd.DataFrame()

    results = []

    # Post-flood, inside zone
    post_inside = df[(df["post"]) & (df["inside_inund"])]

    print("\nPrice by Buyer Type (Post-flood, Inside Zone):")
    print("-" * 50)

    for form in post_inside[owner_col].unique():
        form_prices = post_inside[post_inside[owner_col] == form]["log_price"].dropna()
        other_prices = post_inside[post_inside[owner_col] != form]["log_price"].dropna()

        if len(form_prices) > 5:
            mean_price = np.exp(form_prices.mean())
            median_price = np.exp(form_prices.median())

            if len(other_prices) > 5:
                tstat, pval = stats.ttest_ind(form_prices, other_prices)
                diff = form_prices.mean() - other_prices.mean()
            else:
                tstat, pval, diff = np.nan, np.nan, np.nan

            sig = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.1 else ""

            print(f"  {form}: ${mean_price:,.0f} (n={len(form_prices)}) "
                  f"diff={diff*100:+.1f}% {sig}")

            results.append({
                "buyer_type": form,
                "mean_log_price": form_prices.mean(),
                "mean_price": mean_price,
                "median_price": median_price,
                "n_sales": len(form_prices),
                "diff_vs_other_pct": diff * 100 if not np.isnan(diff) else np.nan,
                "pval": pval
            })

    return pd.DataFrame(results)


def estimate_buyer_did_regression(df: pd.DataFrame) -> dict:
    """
    Regression-based DiD for buyer composition.
    """
    print("\n" + "="*60)
    print("BUYER COMPOSITION DiD REGRESSION")
    print("="*60)

    # Create LLC indicator if available
    if "owner_form" in df.columns:
        df["is_LLC"] = (df["owner_form"] == "LLC").astype(int)
    elif "is_LLC" not in df.columns:
        print("Cannot create LLC indicator")
        return {}

    # Prepare data
    model_df = df.dropna(subset=["is_LLC", "inside_inund", "post"]).copy()
    model_df["inside_x_post"] = model_df["inside_inund"].astype(int) * model_df["post"].astype(int)
    model_df["inside_inund"] = model_df["inside_inund"].astype(int)
    model_df["post"] = model_df["post"].astype(int)

    # DiD regression
    X = sm.add_constant(model_df[["inside_inund", "post", "inside_x_post"]])
    y = model_df["is_LLC"]

    try:
        model = sm.OLS(y, X)
        result = model.fit(cov_type="HC1")

        print(f"\nLinear Probability Model: P(LLC Buyer)")
        print("-" * 50)
        print(f"N = {len(model_df):,}")
        print(f"R² = {result.rsquared:.4f}")

        for var in result.params.index:
            coef = result.params[var]
            se = result.bse[var]
            pval = result.pvalues[var]
            sig = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.1 else ""
            print(f"  {var}: {coef:.4f} (SE: {se:.4f}) {sig}")

        # Key result
        did_coef = result.params["inside_x_post"]
        did_pval = result.pvalues["inside_x_post"]

        print(f"\n*** DiD ESTIMATE ***")
        print(f"Inside × Post: {did_coef:.4f} (p={did_pval:.4f})")
        print(f"  => LLC buyer share changed by {did_coef*100:.1f}pp inside vs outside")

        return {
            "model": "OLS",
            "did_coef": did_coef,
            "did_se": result.bse["inside_x_post"],
            "did_pval": did_pval,
            "n_obs": len(model_df),
            "r_squared": result.rsquared
        }

    except Exception as e:
        print(f"Regression failed: {e}")
        return {}


def generate_buyer_summary(
    form_did: pd.DataFrame,
    portfolio: pd.DataFrame,
    proximity: pd.DataFrame,
    price_by_type: pd.DataFrame,
    did_regression: dict
) -> pd.DataFrame:
    """Generate summary of buyer analysis."""

    print("\n" + "="*60)
    print("BUYER ANALYSIS SUMMARY")
    print("="*60)

    findings = []

    # Owner form changes
    if len(form_did) > 0:
        # Look for significant changes
        for _, row in form_did.iterrows():
            if abs(row["did_estimate"]) > 2:  # 2pp threshold
                findings.append({
                    "category": "Owner Form",
                    "finding": f"{row['owner_form']} share DiD = {row['did_estimate']:+.1f}pp",
                    "direction": "increased" if row["did_estimate"] > 0 else "decreased",
                    "implication": "Buyer composition shifted"
                })

    # Portfolio buyers
    if len(portfolio) > 0:
        # Calculate DiD from the data
        pass  # Already printed in function

    # Regression result
    if did_regression:
        coef = did_regression.get("did_coef", 0)
        pval = did_regression.get("did_pval", 1)
        findings.append({
            "category": "LLC Buyers (Regression)",
            "finding": f"DiD = {coef*100:+.1f}pp (p={pval:.3f})",
            "direction": "increased" if coef > 0 else "decreased",
            "implication": "Statistically significant" if pval < 0.1 else "Not significant"
        })

    findings_df = pd.DataFrame(findings)

    print("\nKey Findings:")
    print("-" * 50)
    for _, row in findings_df.iterrows():
        print(f"  [{row['category']}] {row['finding']}")
        print(f"    => {row['implication']}")

    print("\n" + "="*60)
    print("INTERPRETATION FOR COUNTERINTUITIVE PRICE EFFECT")
    print("="*60)
    print("""
If institutional/portfolio buyers increased inside post-flood:
1. Investors may be buying "distressed" properties at discount
2. But these are still higher quality/value properties
3. Selection effect: only quality properties transact
4. Reported prices biased upward by buyer type

If local buyers decreased:
1. Local individuals may be risk-averse post-flood
2. Distant/institutional buyers less affected by local risk perception
3. Different reservation prices by buyer type
""")

    return findings_df


def main(caliper_m: int = 300):
    """Main function to run extended buyer analysis."""

    print("="*70)
    print("EXTENDED BUYER ANALYSIS")
    print(f"Caliper: ±{caliper_m}m from inundation boundary")
    print("="*70)

    # Ensure output directory
    DIAGNOSTICS_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    try:
        df = load_sales_with_buyers(caliper_m)
        print(f"\nLoaded data: {len(df):,} sales")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    # Ensure inside indicator
    if "inside_inund" not in df.columns:
        if "signed_dist_inund_m" in df.columns:
            df["inside_inund"] = df["signed_dist_inund_m"] < 0
        else:
            print("Cannot determine inside/outside status")
            return

    # Analysis 1: Owner form changes
    form_did = analyze_owner_form_changes(df)
    if len(form_did) > 0:
        form_did.to_csv(DIAGNOSTICS_DIR / "buyer_form_did.csv", index=False)

    # Analysis 2: Portfolio buyers
    portfolio = analyze_portfolio_buyers(df)
    if len(portfolio) > 0:
        portfolio.to_csv(DIAGNOSTICS_DIR / "portfolio_buyers.csv", index=False)

    # Analysis 3: Buyer proximity
    proximity = analyze_buyer_proximity(df)
    if len(proximity) > 0:
        proximity.to_csv(DIAGNOSTICS_DIR / "buyer_proximity.csv", index=False)

    # Analysis 4: Price by buyer type
    price_by_type = analyze_price_by_buyer_type(df)
    if len(price_by_type) > 0:
        price_by_type.to_csv(DIAGNOSTICS_DIR / "price_by_buyer_type.csv", index=False)

    # Analysis 5: DiD regression
    did_regression = estimate_buyer_did_regression(df)

    # Summary
    summary = generate_buyer_summary(form_did, portfolio, proximity, price_by_type, did_regression)
    summary.to_csv(DIAGNOSTICS_DIR / "buyer_analysis_summary.csv", index=False)

    print(f"\nResults saved to {DIAGNOSTICS_DIR}/")
    print("  - buyer_form_did.csv")
    print("  - portfolio_buyers.csv")
    print("  - buyer_proximity.csv")
    print("  - price_by_buyer_type.csv")
    print("  - buyer_analysis_summary.csv")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extended buyer analysis")
    parser.add_argument(
        "-c", "--caliper",
        type=int,
        default=300,
        help="Caliper window in meters (default: 300)"
    )

    args = parser.parse_args()
    main(caliper_m=args.caliper)
