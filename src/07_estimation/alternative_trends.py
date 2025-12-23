"""
alternative_trends.py
=====================
Test robustness to alternative trend specifications.

The counterintuitive positive price effect may be sensitive to how
trends are modeled. This script tests multiple specifications.

Specifications:
1. No trends (baseline)
2. Linear group trends (current)
3. Quadratic group trends
4. Neighborhood-specific trends
5. Matched sample approach
6. Pre-period growth matching

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


def load_data(caliper_m: int = 300, start_year: int = 2015, end_year: int = 2022) -> pd.DataFrame:
    """Load sales data with time variables."""

    sales_path = DATA_WORK / "sales_clean.parquet"
    if not sales_path.exists():
        sales_path = DATA_WORK / "sales_panel.parquet"

    if not sales_path.exists():
        raise FileNotFoundError("No sales data found")

    df = pd.read_parquet(sales_path)
    print(f"Loaded data: {len(df):,} sales")

    # Merge with boundary distances
    distances_path = DATA_WORK / "parcel_boundary_distances.parquet"
    if distances_path.exists():
        distances = pd.read_parquet(distances_path)
        df = df.merge(
            distances[["parcel_id", "signed_dist_inund_m", "inside_inund"]],
            on="parcel_id",
            how="inner"
        )
        print(f"After merge with distances: {len(df):,} sales")

    # Create log_price from price_nominal
    if "log_price" not in df.columns and "price_nominal" in df.columns:
        df["log_price"] = np.log(df["price_nominal"].clip(lower=1))

    # Ensure date column
    if "sale_date" in df.columns:
        df["sale_date"] = pd.to_datetime(df["sale_date"])
        df["year"] = df["sale_date"].dt.year
        df["month"] = df["sale_date"].dt.month
        df["ym"] = df["sale_date"].dt.to_period("M")
    elif "ym" in df.columns:
        df["ym"] = pd.to_datetime(df["ym"]).dt.to_period("M")
        df["year"] = df["ym"].dt.year
        df["month"] = df["ym"].dt.month

    # Filter years
    df = df[(df["year"] >= start_year) & (df["year"] <= end_year)]

    # Filter to caliper
    if "signed_dist_inund_m" in df.columns:
        df = df[df["signed_dist_inund_m"].abs() <= caliper_m]
        print(f"After caliper filter ({caliper_m}m): {len(df):,} sales")

    # Create time variables
    df["post"] = df["sale_date"] >= EVENT_DATE if "sale_date" in df.columns else False
    df["time"] = (df["ym"] - pd.Period("2015-01", "M")).apply(lambda x: x.n)

    # Inside indicator
    if "inside_inund" not in df.columns:
        if "signed_dist_inund_m" in df.columns:
            df["inside_inund"] = df["signed_dist_inund_m"] < 0

    return df


def estimate_no_trends(df: pd.DataFrame) -> dict:
    """
    Specification 1: Simple DiD without trend controls.
    """
    print("\n" + "="*60)
    print("SPECIFICATION 1: NO TRENDS")
    print("="*60)

    if "log_price" not in df.columns:
        if "sale_price" in df.columns:
            df["log_price"] = np.log(df["sale_price"].clip(lower=1))
        else:
            return {"error": "No price data"}

    model_df = df.dropna(subset=["log_price", "inside_inund", "post"]).copy()
    model_df["inside_x_post"] = model_df["inside_inund"].astype(int) * model_df["post"].astype(int)
    model_df["inside_inund"] = model_df["inside_inund"].astype(int)
    model_df["post"] = model_df["post"].astype(int)

    X = sm.add_constant(model_df[["inside_inund", "post", "inside_x_post"]])
    y = model_df["log_price"]

    try:
        result = sm.OLS(y, X).fit(cov_type="HC1")

        coef = result.params["inside_x_post"]
        se = result.bse["inside_x_post"]
        pval = result.pvalues["inside_x_post"]

        print(f"DiD coefficient: {coef:.4f} (SE: {se:.4f})")
        print(f"P-value: {pval:.4f}")
        print(f"=> {(np.exp(coef)-1)*100:+.1f}% price change inside vs outside")

        return {
            "specification": "no_trends",
            "did_coef": coef,
            "did_se": se,
            "did_pval": pval,
            "r_squared": result.rsquared,
            "n_obs": len(model_df)
        }

    except Exception as e:
        return {"error": str(e)}


def estimate_linear_trends(df: pd.DataFrame) -> dict:
    """
    Specification 2: DiD with linear group-specific trends.
    """
    print("\n" + "="*60)
    print("SPECIFICATION 2: LINEAR GROUP TRENDS")
    print("="*60)

    model_df = df.dropna(subset=["log_price", "inside_inund", "post", "time"]).copy()
    model_df["inside_inund"] = model_df["inside_inund"].astype(int)
    model_df["post"] = model_df["post"].astype(int)
    model_df["inside_x_post"] = model_df["inside_inund"] * model_df["post"]
    model_df["inside_x_time"] = model_df["inside_inund"] * model_df["time"]
    model_df["outside_x_time"] = (1 - model_df["inside_inund"]) * model_df["time"]

    X = sm.add_constant(model_df[["inside_inund", "post", "inside_x_post",
                                   "inside_x_time", "outside_x_time"]])
    y = model_df["log_price"]

    try:
        result = sm.OLS(y, X).fit(cov_type="HC1")

        coef = result.params["inside_x_post"]
        se = result.bse["inside_x_post"]
        pval = result.pvalues["inside_x_post"]

        print(f"DiD coefficient: {coef:.4f} (SE: {se:.4f})")
        print(f"P-value: {pval:.4f}")
        print(f"Inside trend: {result.params['inside_x_time']:.4f}")
        print(f"Outside trend: {result.params['outside_x_time']:.4f}")

        return {
            "specification": "linear_trends",
            "did_coef": coef,
            "did_se": se,
            "did_pval": pval,
            "inside_trend": result.params["inside_x_time"],
            "outside_trend": result.params["outside_x_time"],
            "r_squared": result.rsquared,
            "n_obs": len(model_df)
        }

    except Exception as e:
        return {"error": str(e)}


def estimate_quadratic_trends(df: pd.DataFrame) -> dict:
    """
    Specification 3: DiD with quadratic group-specific trends.
    """
    print("\n" + "="*60)
    print("SPECIFICATION 3: QUADRATIC GROUP TRENDS")
    print("="*60)

    model_df = df.dropna(subset=["log_price", "inside_inund", "post", "time"]).copy()
    model_df["inside_inund"] = model_df["inside_inund"].astype(int)
    model_df["post"] = model_df["post"].astype(int)
    model_df["inside_x_post"] = model_df["inside_inund"] * model_df["post"]
    model_df["time_sq"] = model_df["time"] ** 2

    # Linear terms
    model_df["inside_x_time"] = model_df["inside_inund"] * model_df["time"]
    model_df["outside_x_time"] = (1 - model_df["inside_inund"]) * model_df["time"]

    # Quadratic terms
    model_df["inside_x_time_sq"] = model_df["inside_inund"] * model_df["time_sq"]
    model_df["outside_x_time_sq"] = (1 - model_df["inside_inund"]) * model_df["time_sq"]

    X = sm.add_constant(model_df[["inside_inund", "post", "inside_x_post",
                                   "inside_x_time", "outside_x_time",
                                   "inside_x_time_sq", "outside_x_time_sq"]])
    y = model_df["log_price"]

    try:
        result = sm.OLS(y, X).fit(cov_type="HC1")

        coef = result.params["inside_x_post"]
        se = result.bse["inside_x_post"]
        pval = result.pvalues["inside_x_post"]

        print(f"DiD coefficient: {coef:.4f} (SE: {se:.4f})")
        print(f"P-value: {pval:.4f}")

        return {
            "specification": "quadratic_trends",
            "did_coef": coef,
            "did_se": se,
            "did_pval": pval,
            "r_squared": result.rsquared,
            "n_obs": len(model_df)
        }

    except Exception as e:
        return {"error": str(e)}


def estimate_neighborhood_trends(df: pd.DataFrame) -> dict:
    """
    Specification 4: DiD with neighborhood-specific trends.
    """
    print("\n" + "="*60)
    print("SPECIFICATION 4: NEIGHBORHOOD-SPECIFIC TRENDS")
    print("="*60)

    # Check for neighborhood variable
    hood_cols = ["neighborhood", "neighborhood_id", "census_tract", "zip"]
    hood_col = None
    for col in hood_cols:
        if col in df.columns:
            hood_col = col
            break

    if hood_col is None:
        print("No neighborhood variable found")
        return {"error": "No neighborhood variable"}

    model_df = df.dropna(subset=["log_price", "inside_inund", "post", "time", hood_col]).copy()

    # Need at least some variation
    if model_df[hood_col].nunique() < 3:
        return {"error": "Insufficient neighborhood variation"}

    model_df["inside_inund"] = model_df["inside_inund"].astype(int)
    model_df["post"] = model_df["post"].astype(int)
    model_df["inside_x_post"] = model_df["inside_inund"] * model_df["post"]

    # Create neighborhood-time interactions
    hood_dummies = pd.get_dummies(model_df[hood_col], prefix="hood", drop_first=True)
    for col in hood_dummies.columns:
        model_df[f"{col}_time"] = hood_dummies[col] * model_df["time"]

    trend_cols = [c for c in model_df.columns if c.endswith("_time") and c.startswith("hood")]

    X_cols = ["inside_inund", "post", "inside_x_post"] + trend_cols
    X = sm.add_constant(model_df[X_cols])
    y = model_df["log_price"]

    try:
        result = sm.OLS(y, X).fit(cov_type="HC1")

        coef = result.params["inside_x_post"]
        se = result.bse["inside_x_post"]
        pval = result.pvalues["inside_x_post"]

        print(f"DiD coefficient: {coef:.4f} (SE: {se:.4f})")
        print(f"P-value: {pval:.4f}")
        print(f"Number of neighborhood trends: {len(trend_cols)}")

        return {
            "specification": "neighborhood_trends",
            "did_coef": coef,
            "did_se": se,
            "did_pval": pval,
            "n_neighborhoods": len(trend_cols) + 1,
            "r_squared": result.rsquared,
            "n_obs": len(model_df)
        }

    except Exception as e:
        return {"error": str(e)}


def estimate_covariate_adjusted(df: pd.DataFrame) -> dict:
    """
    Specification 5: DiD with covariate controls.
    """
    print("\n" + "="*60)
    print("SPECIFICATION 5: COVARIATE ADJUSTED")
    print("="*60)

    covariates = ["log_assessed_value", "building_age", "log_impsf", "log_acres"]
    available = [c for c in covariates if c in df.columns]

    if not available:
        return {"error": "No covariates available"}

    model_df = df.dropna(subset=["log_price", "inside_inund", "post"] + available).copy()
    model_df["inside_inund"] = model_df["inside_inund"].astype(int)
    model_df["post"] = model_df["post"].astype(int)
    model_df["inside_x_post"] = model_df["inside_inund"] * model_df["post"]

    X = sm.add_constant(model_df[["inside_inund", "post", "inside_x_post"] + available])
    y = model_df["log_price"]

    try:
        result = sm.OLS(y, X).fit(cov_type="HC1")

        coef = result.params["inside_x_post"]
        se = result.bse["inside_x_post"]
        pval = result.pvalues["inside_x_post"]

        print(f"DiD coefficient: {coef:.4f} (SE: {se:.4f})")
        print(f"P-value: {pval:.4f}")
        print(f"Covariates: {', '.join(available)}")

        return {
            "specification": "covariate_adjusted",
            "did_coef": coef,
            "did_se": se,
            "did_pval": pval,
            "covariates": available,
            "r_squared": result.rsquared,
            "n_obs": len(model_df)
        }

    except Exception as e:
        return {"error": str(e)}


def estimate_matched_sample(df: pd.DataFrame) -> dict:
    """
    Specification 6: Matched sample DiD.
    Match inside to outside properties on pre-event characteristics.
    """
    print("\n" + "="*60)
    print("SPECIFICATION 6: MATCHED SAMPLE")
    print("="*60)

    # Simple matching on assessed value quartile
    if "log_assessed_value" not in df.columns:
        return {"error": "No assessed value for matching"}

    # Pre-period data for matching
    pre_df = df[df["post"] == False].copy()

    # Create value quartiles
    pre_df["value_quartile"] = pd.qcut(pre_df["log_assessed_value"], 4, labels=False, duplicates="drop")

    # Match: for each inside property, find outside properties in same quartile
    inside_pre = pre_df[pre_df["inside_inund"] == True]
    outside_pre = pre_df[pre_df["inside_inund"] == False]

    # Get matched parcels
    matched_parcels = set()

    for quartile in inside_pre["value_quartile"].unique():
        inside_in_q = inside_pre[inside_pre["value_quartile"] == quartile]
        outside_in_q = outside_pre[outside_pre["value_quartile"] == quartile]

        # Add all inside parcels
        if "parcel_id" in inside_in_q.columns:
            matched_parcels.update(inside_in_q["parcel_id"].unique())

            # Add matched outside parcels (up to same number)
            n_inside = len(inside_in_q["parcel_id"].unique())
            outside_ids = outside_in_q["parcel_id"].unique()[:n_inside*2]  # 2:1 matching
            matched_parcels.update(outside_ids)

    if len(matched_parcels) == 0:
        return {"error": "Matching failed"}

    # Filter full data to matched parcels
    if "parcel_id" in df.columns:
        matched_df = df[df["parcel_id"].isin(matched_parcels)].copy()
    else:
        return {"error": "No parcel_id for matching"}

    print(f"Matched sample: {len(matched_df):,} observations")
    print(f"Unique parcels: {len(matched_parcels):,}")

    # Run DiD on matched sample
    matched_df["inside_inund"] = matched_df["inside_inund"].astype(int)
    matched_df["post"] = matched_df["post"].astype(int)
    matched_df["inside_x_post"] = matched_df["inside_inund"] * matched_df["post"]

    X = sm.add_constant(matched_df[["inside_inund", "post", "inside_x_post"]])
    y = matched_df["log_price"]

    try:
        result = sm.OLS(y, X).fit(cov_type="HC1")

        coef = result.params["inside_x_post"]
        se = result.bse["inside_x_post"]
        pval = result.pvalues["inside_x_post"]

        print(f"DiD coefficient: {coef:.4f} (SE: {se:.4f})")
        print(f"P-value: {pval:.4f}")

        return {
            "specification": "matched_sample",
            "did_coef": coef,
            "did_se": se,
            "did_pval": pval,
            "n_matched_parcels": len(matched_parcels),
            "r_squared": result.rsquared,
            "n_obs": len(matched_df)
        }

    except Exception as e:
        return {"error": str(e)}


def compare_specifications(results: list) -> pd.DataFrame:
    """Compare results across specifications."""

    print("\n" + "="*60)
    print("SPECIFICATION COMPARISON")
    print("="*60)

    comparison = []

    for r in results:
        if "error" in r:
            continue

        comparison.append({
            "specification": r.get("specification", "unknown"),
            "did_coef": r.get("did_coef", np.nan),
            "did_se": r.get("did_se", np.nan),
            "did_pval": r.get("did_pval", np.nan),
            "n_obs": r.get("n_obs", 0),
            "r_squared": r.get("r_squared", np.nan)
        })

    comparison_df = pd.DataFrame(comparison)

    if len(comparison_df) > 0:
        comparison_df["pct_effect"] = (np.exp(comparison_df["did_coef"]) - 1) * 100
        comparison_df["significant"] = comparison_df["did_pval"] < 0.05

        print("\nResults by Specification:")
        print("-" * 70)
        for _, row in comparison_df.iterrows():
            sig = "***" if row["did_pval"] < 0.01 else "**" if row["did_pval"] < 0.05 else "*" if row["did_pval"] < 0.1 else ""
            print(f"  {row['specification']:25s}: {row['pct_effect']:+6.1f}% "
                  f"(SE: {row['did_se']:.3f}, N={row['n_obs']:,}) {sig}")

        # Summary statistics
        print("\n" + "-" * 70)
        print(f"Mean DiD coefficient: {comparison_df['did_coef'].mean():.4f}")
        print(f"Coefficient range: [{comparison_df['did_coef'].min():.4f}, {comparison_df['did_coef'].max():.4f}]")
        print(f"Consistently positive: {(comparison_df['did_coef'] > 0).all()}")
        print(f"Consistently significant: {comparison_df['significant'].all()}")

    return comparison_df


def main(caliper_m: int = 300, start_year: int = 2015, end_year: int = 2022):
    """Main function to run alternative trend specifications."""

    print("="*70)
    print("ALTERNATIVE TREND SPECIFICATIONS")
    print(f"Caliper: ±{caliper_m}m | Period: {start_year}-{end_year}")
    print("="*70)

    # Ensure output directory
    DIAGNOSTICS_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    try:
        df = load_data(caliper_m, start_year, end_year)
        print(f"\nLoaded data: {len(df):,} sales")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    # Run all specifications
    results = []

    # Spec 1: No trends
    results.append(estimate_no_trends(df.copy()))

    # Spec 2: Linear trends
    results.append(estimate_linear_trends(df.copy()))

    # Spec 3: Quadratic trends
    results.append(estimate_quadratic_trends(df.copy()))

    # Spec 4: Neighborhood trends
    results.append(estimate_neighborhood_trends(df.copy()))

    # Spec 5: Covariate adjusted
    results.append(estimate_covariate_adjusted(df.copy()))

    # Spec 6: Matched sample
    results.append(estimate_matched_sample(df.copy()))

    # Compare
    comparison = compare_specifications(results)
    comparison.to_csv(DIAGNOSTICS_DIR / "trend_specification_comparison.csv", index=False)

    # Summary interpretation
    print("\n" + "="*60)
    print("INTERPRETATION")
    print("="*60)

    if len(comparison) > 0:
        all_positive = (comparison["did_coef"] > 0).all()
        any_negative = (comparison["did_coef"] < 0).any()
        range_coef = comparison["did_coef"].max() - comparison["did_coef"].min()

        if all_positive:
            print("""
FINDING: Positive price effect is ROBUST across specifications.

The counterintuitive result (prices rising faster inside) persists
regardless of how trends are modeled. This suggests the effect is:

1. NOT a specification artifact
2. Likely driven by selection or composition
3. Requires investigation of WHO is selling/buying

Next steps:
- Focus on seller_selection_analysis.py results
- Focus on buyer_analysis_extended.py results
- Consider rate_price_reconciliation findings
""")
        elif any_negative:
            print(f"""
FINDING: Price effect is SENSITIVE to specification.

Some specifications show negative effect (flight), others positive.
Coefficient range: {range_coef:.4f}

This suggests:
1. Trend modeling matters significantly
2. Need to justify specification choice
3. Consider reporting range of estimates
""")

    print(f"\nResults saved to {DIAGNOSTICS_DIR}/trend_specification_comparison.csv")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Alternative trend specifications")
    parser.add_argument("-c", "--caliper", type=int, default=300,
                        help="Caliper window in meters (default: 300)")
    parser.add_argument("--start-year", type=int, default=2015,
                        help="Start year (default: 2015)")
    parser.add_argument("--end-year", type=int, default=2022,
                        help="End year (default: 2022)")

    args = parser.parse_args()
    main(caliper_m=args.caliper, start_year=args.start_year, end_year=args.end_year)
