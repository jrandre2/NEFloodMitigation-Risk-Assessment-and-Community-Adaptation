#!/usr/bin/env python3
"""
Module: decompose_price_effect.py
Purpose: Decompose the counterintuitive positive price effect by property characteristics.

This module investigates whether the positive price effect (prices rising faster inside
the flood zone) is driven by compositional changes in the types of properties sold.

Key Questions
-------------
1. Are newer properties disproportionately sold inside post-flood?
2. Do different property types show different effects?
3. Are larger/more expensive properties driving the result?
4. Is there evidence of new construction or renovations?

Input Files
-----------
- data_work/sales_clean.parquet
- data_work/parcel_boundary_distances.parquet
- data_work/parcel_covariates_full.parquet

Output Files
------------
- data_work/diagnostics/price_effect_by_building_age.csv
- data_work/diagnostics/price_effect_by_property_chars.csv
- data_work/diagnostics/composition_shift_analysis.csv
- figures/fig_price_decomposition.png

Functions
---------
load_data : Load and merge sales, distances, and covariate data
analyze_composition_shift : Test if property composition changed post-flood
estimate_did_by_subgroup : Estimate DiD separately for subgroups
plot_decomposition : Visualize decomposition results
main : Execute decomposition analysis
"""
from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path
from typing import Optional, Dict, List, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

SALES = Path('data_work/sales_clean.parquet')
DISTANCES = Path('data_work/parcel_boundary_distances.parquet')
COVARIATES = Path('data_work/parcel_covariates_full.parquet')
OUT_DIR = Path('data_work/diagnostics')
FIG_DIR = Path('figures')


def ensure_env():
    """Verify virtual environment is activated."""
    if not os.getenv('VIRTUAL_ENV') or not os.getenv('VIRTUAL_ENV').endswith('/.venv'):
        print('ERROR: Please activate project .venv before running.', file=sys.stderr)
        sys.exit(1)


def load_data(caliper: int = 300,
              start_year: int = 2010,
              end_year: int = 2022) -> pd.DataFrame:
    """
    Load and merge sales with boundary distances and property characteristics.
    """
    print('Loading data...')

    # Load sales
    sales = pd.read_parquet(SALES)
    sales['sale_date'] = pd.to_datetime(sales['sale_date'])
    sales = sales[(sales['sale_date'].dt.year >= start_year) &
                  (sales['sale_date'].dt.year <= end_year)]

    # Load distances
    distances = pd.read_parquet(DISTANCES)

    # Load covariates
    if COVARIATES.exists():
        covariates = pd.read_parquet(COVARIATES)
        print(f'  Loaded {len(covariates):,} parcel covariates')
    else:
        print('  WARNING: No covariate file found, using limited features')
        covariates = None

    # Merge
    df = sales.merge(
        distances[['parcel_id', 'signed_dist_inund_m', 'inside_inund']],
        on='parcel_id',
        how='inner'
    )

    if covariates is not None:
        # Select relevant columns from covariates
        cov_cols = ['parcel_id']
        for col in ['year_built', 'Total_Assessed_Value', 'GIS_Acres', 'ImpSF',
                    'building_age', 'LandSF', 'property_class']:
            if col in covariates.columns:
                cov_cols.append(col)

        df = df.merge(covariates[cov_cols], on='parcel_id', how='left')

    # Filter to caliper window
    df = df[df['signed_dist_inund_m'].abs() <= caliper].copy()

    # Time variables
    df['year'] = df['sale_date'].dt.year
    df['post'] = (df['sale_date'] >= '2019-03-01').astype(int)

    # Create building age at sale if we have year_built
    if 'year_built' in df.columns:
        df['age_at_sale'] = df['year'] - df['year_built']
        df['new_construction'] = (df['age_at_sale'] <= 5).astype(int)
        df['old_building'] = (df['age_at_sale'] > 30).astype(int)

    print(f'  Loaded {len(df):,} sales within ±{caliper}m')

    return df


def analyze_composition_shift(df: pd.DataFrame) -> pd.DataFrame:
    """
    Test if property composition changed after the flood inside vs outside.
    """
    print('\nAnalyzing composition shift...')

    results = []

    # Define characteristics to test
    chars_to_test = []

    if 'age_at_sale' in df.columns:
        chars_to_test.append(('age_at_sale', 'Mean Age at Sale'))
        chars_to_test.append(('new_construction', 'New Construction (<=5 yrs)'))

    if 'price_nominal' in df.columns:
        df['log_price'] = np.log(df['price_nominal'].replace(0, np.nan))
        chars_to_test.append(('log_price', 'Log Sale Price'))

    if 'Total_Assessed_Value' in df.columns:
        df['log_assessed'] = np.log(df['Total_Assessed_Value'].replace(0, np.nan))
        chars_to_test.append(('log_assessed', 'Log Assessed Value'))

    if 'GIS_Acres' in df.columns:
        df['log_acres'] = np.log(df['GIS_Acres'].replace(0, np.nan))
        chars_to_test.append(('log_acres', 'Log Lot Size'))

    if 'ImpSF' in df.columns:
        df['log_impsf'] = np.log(df['ImpSF'].replace(0, np.nan))
        chars_to_test.append(('log_impsf', 'Log Building SF'))

    for var, label in chars_to_test:
        if var not in df.columns:
            continue

        # Calculate means by group
        means = df.groupby(['inside_inund', 'post'])[var].agg(['mean', 'std', 'count'])
        means = means.reset_index()

        # DiD calculation
        try:
            inside_pre = means[(means['inside_inund'] == 1) & (means['post'] == 0)]['mean'].values[0]
            inside_post = means[(means['inside_inund'] == 1) & (means['post'] == 1)]['mean'].values[0]
            outside_pre = means[(means['inside_inund'] == 0) & (means['post'] == 0)]['mean'].values[0]
            outside_post = means[(means['inside_inund'] == 0) & (means['post'] == 1)]['mean'].values[0]

            did = (inside_post - inside_pre) - (outside_post - outside_pre)

            # Regression for SE
            d = df[[var, 'inside_inund', 'post']].dropna()
            d['inside_x_post'] = d['inside_inund'] * d['post']
            X = sm.add_constant(d[['inside_inund', 'post', 'inside_x_post']])
            y = d[var]

            try:
                model = sm.OLS(y, X).fit(cov_type='HC3')
                se = model.bse.get('inside_x_post', np.nan)
                pval = model.pvalues.get('inside_x_post', np.nan)
            except Exception:
                se, pval = np.nan, np.nan

            results.append({
                'variable': var,
                'label': label,
                'inside_pre': inside_pre,
                'inside_post': inside_post,
                'outside_pre': outside_pre,
                'outside_post': outside_post,
                'did': did,
                'se': se,
                'pval': pval,
                'n_obs': len(d)
            })

            print(f'  {label}: DiD = {did:.4f} (p={pval:.4f})')

        except (IndexError, KeyError) as e:
            print(f'  {label}: Could not compute - {e}')

    return pd.DataFrame(results)


def estimate_did_by_subgroup(df: pd.DataFrame,
                              grouping_var: str,
                              outcome: str = 'log_price') -> pd.DataFrame:
    """
    Estimate DiD separately for subgroups defined by grouping_var.
    """
    print(f'\nEstimating DiD by {grouping_var}...')

    if grouping_var not in df.columns:
        print(f'  WARNING: {grouping_var} not in data')
        return pd.DataFrame()

    if outcome not in df.columns:
        df['log_price'] = np.log(df['price_nominal'].replace(0, np.nan))

    results = []

    for group_val in df[grouping_var].dropna().unique():
        subset = df[df[grouping_var] == group_val].copy()
        subset = subset.dropna(subset=[outcome, 'inside_inund', 'post'])

        if len(subset) < 50:
            continue

        subset['inside_x_post'] = subset['inside_inund'] * subset['post']

        X = sm.add_constant(subset[['inside_inund', 'post', 'inside_x_post']])
        y = subset[outcome]

        try:
            model = sm.OLS(y, X).fit(cov_type='HC3')
            coef = model.params.get('inside_x_post', np.nan)
            se = model.bse.get('inside_x_post', np.nan)
            pval = model.pvalues.get('inside_x_post', np.nan)

            results.append({
                'grouping_var': grouping_var,
                'group_value': group_val,
                'outcome': outcome,
                'coef': coef,
                'se': se,
                'ci_lo': coef - 1.96 * se,
                'ci_hi': coef + 1.96 * se,
                'pval': pval,
                'n_obs': len(subset)
            })

            print(f'  {grouping_var}={group_val}: coef={coef:.4f} (p={pval:.4f}, n={len(subset)})')

        except Exception as e:
            print(f'  {grouping_var}={group_val}: Error - {e}')

    return pd.DataFrame(results)


def create_subgroups(df: pd.DataFrame) -> pd.DataFrame:
    """Create subgroup indicators for analysis."""

    # Building age groups
    if 'age_at_sale' in df.columns:
        df['age_group'] = pd.cut(
            df['age_at_sale'],
            bins=[-np.inf, 5, 20, 50, np.inf],
            labels=['New (0-5)', 'Young (6-20)', 'Mature (21-50)', 'Old (50+)']
        )

    # Price terciles
    if 'price_nominal' in df.columns:
        df['price_tercile'] = pd.qcut(
            df['price_nominal'],
            q=3,
            labels=['Low', 'Medium', 'High']
        )

    # Lot size terciles
    if 'GIS_Acres' in df.columns:
        df['lot_tercile'] = pd.qcut(
            df['GIS_Acres'].clip(lower=0.01),
            q=3,
            labels=['Small', 'Medium', 'Large']
        )

    return df


def plot_decomposition(composition_results: pd.DataFrame,
                       subgroup_results: pd.DataFrame,
                       out_path: Path) -> None:
    """Generate decomposition visualization."""

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. Composition shift DiD
    ax = axes[0, 0]
    if len(composition_results) > 0:
        comp = composition_results.dropna(subset=['did', 'se'])
        if len(comp) > 0:
            x = range(len(comp))
            ax.barh(x, comp['did'], xerr=1.96*comp['se'], capsize=3, alpha=0.7)
            ax.set_yticks(x)
            ax.set_yticklabels(comp['label'], fontsize=9)
            ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
            ax.set_xlabel('DiD (Inside Post vs Outside Post)')
            ax.set_title('Composition Shift: Property Characteristics')

    # 2. DiD by age group
    ax = axes[0, 1]
    age_results = subgroup_results[subgroup_results['grouping_var'] == 'age_group']
    if len(age_results) > 0:
        x = range(len(age_results))
        ax.bar(x, age_results['coef'], yerr=1.96*age_results['se'], capsize=3, alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(age_results['group_value'], rotation=45, ha='right')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.set_ylabel('DiD Coefficient (Log Price)')
        ax.set_title('Price Effect by Building Age')

    # 3. DiD by price tercile
    ax = axes[1, 0]
    price_results = subgroup_results[subgroup_results['grouping_var'] == 'price_tercile']
    if len(price_results) > 0:
        x = range(len(price_results))
        ax.bar(x, price_results['coef'], yerr=1.96*price_results['se'], capsize=3,
               alpha=0.7, color='coral')
        ax.set_xticks(x)
        ax.set_xticklabels(price_results['group_value'])
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.set_ylabel('DiD Coefficient (Log Price)')
        ax.set_title('Price Effect by Price Tercile')

    # 4. Sample composition over time
    ax = axes[1, 1]
    ax.text(0.5, 0.5, 'Additional analysis\nto be added',
            ha='center', va='center', transform=ax.transAxes, fontsize=12)
    ax.set_title('Sample Composition Over Time')

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Wrote {out_path}')


def main(caliper_m: int = 300,
         start_year: int = 2010,
         end_year: int = 2022):
    """Execute price effect decomposition analysis."""

    ensure_env()

    print('\n' + '='*70)
    print('PRICE EFFECT DECOMPOSITION ANALYSIS')
    print('='*70)

    # Load data
    df = load_data(caliper_m, start_year, end_year)

    # Create subgroups
    df = create_subgroups(df)

    # 1. Composition shift analysis
    composition_results = analyze_composition_shift(df)

    # 2. DiD by subgroups
    all_subgroup_results = []

    for grouping in ['age_group', 'price_tercile', 'lot_tercile']:
        if grouping in df.columns:
            results = estimate_did_by_subgroup(df, grouping, 'log_price')
            if len(results) > 0:
                all_subgroup_results.append(results)

    subgroup_results = pd.concat(all_subgroup_results, ignore_index=True) if all_subgroup_results else pd.DataFrame()

    # 3. New construction analysis
    print('\n' + '-'*50)
    print('NEW CONSTRUCTION ANALYSIS')
    print('-'*50)

    if 'new_construction' in df.columns:
        new_const = df.groupby(['inside_inund', 'post'])['new_construction'].mean()
        print('\nShare of sales that are new construction:')
        print(new_const.unstack())

        # DiD on new construction share
        df_nc = df.dropna(subset=['new_construction'])
        df_nc['inside_x_post'] = df_nc['inside_inund'] * df_nc['post']
        X = sm.add_constant(df_nc[['inside_inund', 'post', 'inside_x_post']])
        y = df_nc['new_construction']

        try:
            model = sm.OLS(y, X).fit(cov_type='HC3')
            print(f'\nNew Construction DiD: {model.params["inside_x_post"]:.4f} '
                  f'(SE: {model.bse["inside_x_post"]:.4f}, p={model.pvalues["inside_x_post"]:.4f})')
        except Exception as e:
            print(f'Could not estimate new construction DiD: {e}')

    # Save results
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if len(composition_results) > 0:
        comp_path = OUT_DIR / 'composition_shift_analysis.csv'
        composition_results.to_csv(comp_path, index=False)
        print(f'\nWrote {comp_path}')

    if len(subgroup_results) > 0:
        subgroup_path = OUT_DIR / 'price_effect_by_subgroup.csv'
        subgroup_results.to_csv(subgroup_path, index=False)
        print(f'Wrote {subgroup_path}')

    # Generate figure
    fig_path = FIG_DIR / 'fig_price_decomposition.png'
    plot_decomposition(composition_results, subgroup_results, fig_path)

    # Summary
    print('\n' + '='*70)
    print('DECOMPOSITION SUMMARY')
    print('='*70)

    if len(composition_results) > 0:
        sig_shifts = composition_results[composition_results['pval'] < 0.05]
        if len(sig_shifts) > 0:
            print('\nSignificant composition shifts detected:')
            for _, row in sig_shifts.iterrows():
                direction = 'increased' if row['did'] > 0 else 'decreased'
                print(f"  - {row['label']} {direction} inside post-flood (DiD={row['did']:.3f})")
        else:
            print('\nNo significant composition shifts at 5% level.')

    if len(subgroup_results) > 0:
        print('\nDiD coefficients by subgroup:')
        for _, row in subgroup_results.iterrows():
            sig = '*' if row['pval'] < 0.05 else ''
            print(f"  {row['grouping_var']}={row['group_value']}: {row['coef']:.3f}{sig} (n={row['n_obs']})")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Decompose price effect by property characteristics.'
    )
    parser.add_argument(
        '--caliper', '-c',
        type=int,
        default=300,
        help='Caliper window in meters. Default: 300'
    )
    parser.add_argument(
        '--start-year',
        type=int,
        default=2010,
        help='Start year for analysis. Default: 2010'
    )
    parser.add_argument(
        '--end-year',
        type=int,
        default=2022,
        help='End year for analysis. Default: 2022'
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(
        caliper=args.caliper,
        start_year=args.start_year,
        end_year=args.end_year
    )
