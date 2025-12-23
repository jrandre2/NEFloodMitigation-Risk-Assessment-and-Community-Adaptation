#!/usr/bin/env python3
"""
Run covariate balance tests at the SFHA boundary.

For each covariate, estimate: Y = α + β₁·inside_sfha + ε
within the RD caliper window. β₁ ≈ 0 indicates balance.

Input:  data_work/parcel_covariates_full.parquet
Output: data_work/diagnostics/covariate_balance.csv
        figures/fig_covariate_balance.png

This script tests whether parcels inside vs. outside the SFHA boundary
are comparable on pre-treatment covariates (year built, assessed value,
lot size, building size). This is a key RD validity check.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Ensure we're in project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.chdir(PROJECT_ROOT)

try:
    import pandas as pd
    import numpy as np
    import statsmodels.api as sm
except ImportError as e:
    print(f'ERROR: Missing required package: {e}')
    print('Install with: pip install pandas numpy statsmodels')
    sys.exit(1)

# Configuration
IN_FILE = Path('data_work/parcel_covariates_full.parquet')
OUT_DIR = Path('data_work/diagnostics')
FIG_DIR = Path('figures')

# Covariates to test with labels
COVARIATES = [
    ('year_built', 'Year Built'),
    ('building_age', 'Building Age (years)'),
    ('Total_Assessed_Value', 'Assessed Value ($)'),
    ('log_assessed_value', 'Log Assessed Value'),
    ('GIS_Acres', 'Lot Size (acres)'),
    ('log_acres', 'Log Lot Size'),
    ('ImpSF', 'Building SF'),
    ('log_impsf', 'Log Building SF'),
    ('is_improved', 'Is Improved (0/1)'),
]

# Calipers to test
CALIPERS = [100, 200, 300]


def balance_test(df: pd.DataFrame, covariate: str, inside_col: str = 'inside_sfha') -> dict | None:
    """
    Run balance test for a single covariate.

    Estimates: covariate = α + β·inside + ε
    with HC3 robust standard errors.

    Parameters
    ----------
    df : pd.DataFrame
        Data with covariate and inside indicator
    covariate : str
        Column name to test
    inside_col : str
        Name of inside indicator column

    Returns
    -------
    dict or None
        Test results or None if insufficient data
    """
    # Prepare data
    subset = df[[covariate, inside_col]].dropna()

    if len(subset) < 100:
        return None

    # Check for variation
    if subset[inside_col].nunique() < 2:
        return None

    X = sm.add_constant(subset[inside_col])
    y = subset[covariate]

    try:
        model = sm.OLS(y, X).fit(cov_type='HC3')
    except Exception as e:
        print(f'    ERROR fitting model: {e}')
        return None

    inside_mask = subset[inside_col] == 1
    inside_mean = y[inside_mask].mean()
    outside_mean = y[~inside_mask].mean()

    return {
        'coef': model.params[inside_col],
        'se': model.bse[inside_col],
        'tstat': model.tvalues[inside_col],
        'pval': model.pvalues[inside_col],
        'n': len(subset),
        'n_inside': inside_mask.sum(),
        'n_outside': (~inside_mask).sum(),
        'inside_mean': inside_mean,
        'outside_mean': outside_mean,
        'diff_pct': 100 * (inside_mean - outside_mean) / outside_mean if outside_mean != 0 else np.nan,
    }


def main():
    """Run covariate balance tests at multiple calipers."""
    print('=' * 60)
    print('COVARIATE BALANCE TESTS')
    print('=' * 60)

    # Load data
    if not IN_FILE.exists():
        raise SystemExit(f'ERROR: Input file not found: {IN_FILE}\n'
                        f'Run integrate_assessor.py first.')

    print(f'\nLoading {IN_FILE}...')
    df = pd.read_parquet(IN_FILE)
    print(f'Loaded {len(df):,} parcels')

    # Check required columns
    if 'signed_dist_sfha_m' not in df.columns:
        raise SystemExit('ERROR: signed_dist_sfha_m column not found')

    if 'inside_sfha' not in df.columns:
        # Create from signed distance
        df['inside_sfha'] = (df['signed_dist_sfha_m'] < 0).astype(int)
        print('Created inside_sfha from signed_dist_sfha_m')

    results = []

    for caliper in CALIPERS:
        print(f'\n{"="*40}')
        print(f'Caliper: ±{caliper}m')
        print(f'{"="*40}')

        # Filter to caliper window
        window = df[np.abs(df['signed_dist_sfha_m']) <= caliper].copy()
        n_inside = (window['inside_sfha'] == 1).sum()
        n_outside = (window['inside_sfha'] == 0).sum()
        print(f'Parcels in window: {len(window):,} ({n_inside:,} inside, {n_outside:,} outside)')

        if len(window) < 200:
            print('WARNING: Too few parcels for reliable balance tests')
            continue

        for var, label in COVARIATES:
            if var not in window.columns:
                print(f'  SKIP {label}: column not found')
                continue

            # Check coverage
            coverage = window[var].notna().mean()
            if coverage < 0.1:
                print(f'  SKIP {label}: only {100*coverage:.1f}% coverage')
                continue

            result = balance_test(window, var)
            if result is None:
                print(f'  SKIP {label}: insufficient data for test')
                continue

            results.append({
                'caliper': caliper,
                'covariate': var,
                'label': label,
                **result,
                'balanced': result['pval'] > 0.05,
            })

            # Print result
            status = '✓' if result['pval'] > 0.05 else '✗'
            print(f'  {status} {label}:')
            print(f'      Inside: {result["inside_mean"]:.2f} (n={result["n_inside"]:,})')
            print(f'      Outside: {result["outside_mean"]:.2f} (n={result["n_outside"]:,})')
            print(f'      Diff: {result["coef"]:.3f} (SE={result["se"]:.3f}, p={result["pval"]:.3f})')

    if not results:
        print('\nERROR: No balance tests completed')
        return None

    # Create results dataframe
    results_df = pd.DataFrame(results)

    # Summary
    print('\n' + '=' * 60)
    print('SUMMARY')
    print('=' * 60)

    for caliper in CALIPERS:
        subset = results_df[results_df['caliper'] == caliper]
        if len(subset) > 0:
            n_balanced = subset['balanced'].sum()
            n_total = len(subset)
            print(f'±{caliper}m: {n_balanced}/{n_total} covariates balanced (p > 0.05)')

    # Save results
    OUT_DIR.mkdir(exist_ok=True)
    out_file = OUT_DIR / 'covariate_balance.csv'
    results_df.to_csv(out_file, index=False)
    print(f'\nWrote {out_file}')

    # Create figure (if matplotlib available)
    try:
        import matplotlib.pyplot as plt

        FIG_DIR.mkdir(exist_ok=True)

        # Create balance plot for 300m caliper
        df_300 = results_df[results_df['caliper'] == 300].copy()

        if len(df_300) > 0:
            fig, ax = plt.subplots(figsize=(10, 6))

            # Sort by p-value
            df_300 = df_300.sort_values('pval')

            # Plot coefficients with confidence intervals
            y_pos = range(len(df_300))
            ax.errorbar(
                df_300['coef'],
                y_pos,
                xerr=1.96 * df_300['se'],
                fmt='o',
                capsize=3,
                color='steelblue',
                markersize=8
            )

            # Add vertical line at 0
            ax.axvline(x=0, color='black', linestyle='--', linewidth=1)

            # Labels
            ax.set_yticks(y_pos)
            ax.set_yticklabels(df_300['label'])
            ax.set_xlabel('Inside - Outside Difference (with 95% CI)')
            ax.set_title('Covariate Balance at SFHA Boundary (±300m)')

            # Add p-value annotations
            for i, (_, row) in enumerate(df_300.iterrows()):
                p_text = f'p={row["pval"]:.3f}'
                color = 'green' if row['balanced'] else 'red'
                ax.annotate(p_text, xy=(row['coef'], i), xytext=(5, 0),
                           textcoords='offset points', fontsize=8, color=color)

            plt.tight_layout()
            fig_path = FIG_DIR / 'fig_covariate_balance.png'
            plt.savefig(fig_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f'Wrote {fig_path}')

    except ImportError:
        print('\nNote: matplotlib not available, skipping figure')

    print('\n' + '=' * 60)
    print('BALANCE TESTS COMPLETE')
    print('=' * 60)

    return results_df


if __name__ == '__main__':
    main()
