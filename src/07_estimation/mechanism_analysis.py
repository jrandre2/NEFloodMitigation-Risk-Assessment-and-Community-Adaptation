#!/usr/bin/env python3
"""
Module: mechanism_analysis.py
Purpose: Investigate channels through which flood affects housing market.

This module tests potential mechanisms for flood effects:
1. Insurance channel - Compare SFHA (mandatory) vs inundation (optional)
2. Credit constraints - Heterogeneity by property value
3. Information/salience - Distance from worst damage
4. Buyer composition - Changes in buyer types post-flood

Input Files
-----------
- data_work/panel_parcel_month.parquet
- data_work/parcel_covariates_full.parquet

Output Files
------------
- data_work/diagnostics/mechanism_insurance.csv
- data_work/diagnostics/mechanism_credit.csv
- data_work/diagnostics/mechanism_information.csv
- data_work/diagnostics/heterogeneity_by_chars.csv
- data_work/diagnostics/mechanism_summary.csv
- figures/fig_mechanism_heterogeneity.png
"""
from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path
from typing import Optional, List

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

PANEL = Path('data_work/panel_parcel_month.parquet')
COVARIATES = Path('data_work/parcel_covariates_full.parquet')
OUT_DIR = Path('data_work/diagnostics')
FIG_DIR = Path('figures')


def ensure_env():
    if not os.getenv('VIRTUAL_ENV') or not os.getenv('VIRTUAL_ENV').endswith('/.venv'):
        print('ERROR: Please activate project .venv', file=sys.stderr)
        sys.exit(1)


def months_since_event(ym: str) -> int:
    y, m = ym.split('-')
    return (int(y) - 2019) * 12 + (int(m) - 3)


def estimate_did(d: pd.DataFrame, inside_col: str) -> dict:
    """Helper to estimate DiD."""
    d = d.copy()
    d['event_m'] = d['ym'].apply(months_since_event)
    d['post'] = (d['event_m'] >= 0).astype(int)
    d['inside'] = d[inside_col].astype(int)
    d['inside_x_post'] = d['inside'] * d['post']

    X = sm.add_constant(d[['inside', 'post', 'inside_x_post']].astype(float))
    y = d['sold_this_month'].astype(float)

    try:
        model = sm.OLS(y, X).fit(cov_type='HC3')
        return {
            'coef': model.params.get('inside_x_post', np.nan),
            'se': model.bse.get('inside_x_post', np.nan),
            'pval': model.pvalues.get('inside_x_post', np.nan),
            'n_obs': len(d)
        }
    except Exception:
        return {'coef': np.nan, 'se': np.nan, 'pval': np.nan, 'n_obs': 0}


def insurance_channel_analysis(panel: pd.DataFrame, caliper: int = 300) -> dict:
    """
    Compare SFHA (mandatory insurance) vs inundation (optional insurance).

    If insurance mandate drives effects, SFHA should show stronger effects.
    """
    results = {}

    # SFHA boundary
    dist_sfha = 'signed_dist_sfha_m'
    if dist_sfha in panel.columns:
        d_sfha = panel[np.isfinite(panel[dist_sfha]) & (panel[dist_sfha].abs() <= caliper)].copy()
        sfha_result = estimate_did(d_sfha, 'inside_sfha')
        results['sfha_coef'] = sfha_result['coef']
        results['sfha_se'] = sfha_result['se']
        results['sfha_n'] = sfha_result['n_obs']

    # Inundation boundary
    dist_inund = 'signed_dist_inund_m'
    if dist_inund in panel.columns:
        d_inund = panel[np.isfinite(panel[dist_inund]) & (panel[dist_inund].abs() <= caliper)].copy()
        inund_result = estimate_did(d_inund, 'inside_inund')
        results['inund_coef'] = inund_result['coef']
        results['inund_se'] = inund_result['se']
        results['inund_n'] = inund_result['n_obs']

    # Difference
    if 'sfha_coef' in results and 'inund_coef' in results:
        diff = results['sfha_coef'] - results['inund_coef']
        se_diff = np.sqrt(results['sfha_se']**2 + results['inund_se']**2)
        results['diff_sfha_minus_inund'] = diff
        results['diff_se'] = se_diff
        results['insurance_effect_pct'] = diff / results['inund_coef'] * 100 if results['inund_coef'] != 0 else np.nan

    return results


def credit_constraint_analysis(
    panel: pd.DataFrame,
    boundary: str = 'inund',
    caliper: int = 300
) -> pd.DataFrame:
    """
    Test heterogeneity by property value (proxy for credit constraints).

    Higher value properties may face tighter credit constraints post-flood.
    """
    dist_col = f'signed_dist_{boundary}_m'
    inside_col = f'inside_{boundary}'

    if dist_col not in panel.columns:
        return pd.DataFrame()

    d = panel.copy()
    d = d[np.isfinite(d[dist_col])]
    d = d[d[dist_col].abs() <= caliper].copy()

    # Need assessed value
    if 'Total_Assessed_Value' not in d.columns and 'log_assessed_value' not in d.columns:
        return pd.DataFrame()

    if 'log_assessed_value' not in d.columns:
        d['log_assessed_value'] = np.log1p(d['Total_Assessed_Value'])

    # Terciles
    terciles = d['log_assessed_value'].quantile([0.33, 0.67])
    d['value_tercile'] = pd.cut(
        d['log_assessed_value'],
        bins=[-np.inf, terciles[0.33], terciles[0.67], np.inf],
        labels=['Low', 'Medium', 'High']
    )

    results = []
    for tercile in ['Low', 'Medium', 'High']:
        subset = d[d['value_tercile'] == tercile]
        if len(subset) < 100:
            continue
        res = estimate_did(subset, inside_col)
        results.append({
            'boundary': boundary,
            'caliper_m': caliper,
            'value_tercile': tercile,
            **res
        })

    return pd.DataFrame(results)


def heterogeneity_by_property_chars(
    panel: pd.DataFrame,
    boundary: str = 'inund',
    caliper: int = 300,
    chars: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Estimate heterogeneous effects by property characteristics.
    """
    if chars is None:
        chars = ['building_age', 'log_assessed_value', 'log_acres']

    dist_col = f'signed_dist_{boundary}_m'
    inside_col = f'inside_{boundary}'

    if dist_col not in panel.columns:
        return pd.DataFrame()

    d = panel.copy()
    d = d[np.isfinite(d[dist_col])]
    d = d[d[dist_col].abs() <= caliper].copy()

    results = []
    for char in chars:
        if char not in d.columns:
            continue

        # Median split
        median = d[char].median()
        for label, condition in [('Below median', d[char] < median), ('Above median', d[char] >= median)]:
            subset = d[condition]
            if len(subset) < 100:
                continue
            res = estimate_did(subset, inside_col)
            results.append({
                'boundary': boundary,
                'caliper_m': caliper,
                'characteristic': char,
                'subgroup': label,
                **res
            })

    return pd.DataFrame(results)


def buyer_composition_mechanism(
    panel: pd.DataFrame,
    boundary: str = 'inund',
    caliper: int = 300
) -> dict:
    """
    Test whether buyer composition changes drive effects.
    """
    dist_col = f'signed_dist_{boundary}_m'
    inside_col = f'inside_{boundary}'

    if dist_col not in panel.columns:
        return {}

    d = panel.copy()
    d = d[np.isfinite(d[dist_col])]
    d = d[d[dist_col].abs() <= caliper].copy()
    d = d[d['sold_this_month'] == 1].copy()

    d['event_m'] = d['ym'].apply(months_since_event)
    d['post'] = (d['event_m'] >= 0).astype(int)
    d['inside'] = (d[inside_col] == 1).astype(int)

    results = {}

    # LLC share
    if 'is_llc' in d.columns or 'owner_form' in d.columns:
        if 'is_llc' not in d.columns:
            d['is_llc'] = (d['owner_form'] == 'LLC').astype(int)

        for group in ['inside_pre', 'inside_post', 'outside_pre', 'outside_post']:
            inside_val = 1 if 'inside' in group else 0
            post_val = 1 if 'post' in group else 0
            subset = d[(d['inside'] == inside_val) & (d['post'] == post_val)]
            if len(subset) > 0:
                results[f'llc_share_{group}'] = subset['is_llc'].mean()

    # Portfolio buyer share
    if 'is_multi_parcel' in d.columns:
        for group in ['inside_pre', 'inside_post', 'outside_pre', 'outside_post']:
            inside_val = 1 if 'inside' in group else 0
            post_val = 1 if 'post' in group else 0
            subset = d[(d['inside'] == inside_val) & (d['post'] == post_val)]
            if len(subset) > 0:
                results[f'portfolio_share_{group}'] = subset['is_multi_parcel'].mean()

    return results


def mechanism_summary_table(all_results: dict) -> pd.DataFrame:
    """Create consolidated mechanism evidence table."""
    rows = []

    if 'insurance' in all_results:
        ins = all_results['insurance']
        rows.append({
            'mechanism': 'Insurance Mandate',
            'test': 'SFHA vs Inundation DiD',
            'finding': f"SFHA-Inund diff: {ins.get('diff_sfha_minus_inund', np.nan):.4f}",
            'supports_mechanism': ins.get('diff_sfha_minus_inund', 0) < 0
        })

    if 'credit' in all_results and len(all_results['credit']) > 0:
        credit = all_results['credit']
        low = credit[credit['value_tercile'] == 'Low']['coef'].values
        high = credit[credit['value_tercile'] == 'High']['coef'].values
        if len(low) > 0 and len(high) > 0:
            rows.append({
                'mechanism': 'Credit Constraints',
                'test': 'High vs Low value heterogeneity',
                'finding': f"High: {high[0]:.4f}, Low: {low[0]:.4f}",
                'supports_mechanism': abs(high[0]) > abs(low[0])
            })

    return pd.DataFrame(rows)


def plot_heterogeneity(het_results: pd.DataFrame, out_path: Path) -> None:
    """Plot heterogeneous effects."""
    if het_results.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    chars = het_results['characteristic'].unique()
    y_pos = 0
    y_ticks = []
    y_labels = []

    for char in chars:
        subset = het_results[het_results['characteristic'] == char]
        for _, row in subset.iterrows():
            ax.errorbar(
                [row['coef']], [y_pos],
                xerr=[[row['coef'] - (row['coef'] - 1.96 * row['se'])],
                      [(row['coef'] + 1.96 * row['se']) - row['coef']]],
                fmt='o', capsize=4, markersize=8
            )
            y_ticks.append(y_pos)
            y_labels.append(f"{char}: {row['subgroup']}")
            y_pos += 1
        y_pos += 0.5

    ax.axvline(x=0, color='gray', linestyle='--')
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels)
    ax.set_xlabel('DiD Coefficient')
    ax.set_title('Heterogeneous Treatment Effects')
    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Wrote {out_path}')


def main(boundary: str = 'inund', caliper: int = 300):
    """Run mechanism analysis."""
    ensure_env()

    if not PANEL.exists():
        raise SystemExit(f'Missing {PANEL}')

    panel = pd.read_parquet(PANEL)

    # Merge covariates if available
    if COVARIATES.exists():
        cov = pd.read_parquet(COVARIATES)
        merge_cols = [c for c in cov.columns if c != 'parcel_id' and c not in panel.columns]
        if merge_cols:
            panel = panel.merge(cov[['parcel_id'] + merge_cols], on='parcel_id', how='left')

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    print('=' * 60)
    print('MECHANISM ANALYSIS')
    print('=' * 60)

    all_results = {}

    # 1. Insurance channel
    print('\n1. Insurance Channel (SFHA vs Inundation)')
    print('-' * 40)
    ins = insurance_channel_analysis(panel, caliper)
    all_results['insurance'] = ins
    if ins:
        for k, v in ins.items():
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
        pd.DataFrame([ins]).to_csv(OUT_DIR / 'mechanism_insurance.csv', index=False)

    # 2. Credit constraints
    print('\n2. Credit Constraints (by property value)')
    print('-' * 40)
    credit = credit_constraint_analysis(panel, boundary, caliper)
    all_results['credit'] = credit
    if not credit.empty:
        print(credit[['value_tercile', 'coef', 'se', 'n_obs']].to_string(index=False))
        credit.to_csv(OUT_DIR / 'mechanism_credit.csv', index=False)

    # 3. Heterogeneity by characteristics
    print('\n3. Heterogeneity by Property Characteristics')
    print('-' * 40)
    het = heterogeneity_by_property_chars(panel, boundary, caliper)
    all_results['heterogeneity'] = het
    if not het.empty:
        print(het[['characteristic', 'subgroup', 'coef', 'se']].to_string(index=False))
        het.to_csv(OUT_DIR / 'heterogeneity_by_chars.csv', index=False)
        plot_heterogeneity(het, FIG_DIR / 'fig_mechanism_heterogeneity.png')

    # 4. Buyer composition
    print('\n4. Buyer Composition')
    print('-' * 40)
    buyer = buyer_composition_mechanism(panel, boundary, caliper)
    all_results['buyer'] = buyer
    if buyer:
        for k, v in buyer.items():
            print(f"  {k}: {v:.3f}")
        pd.DataFrame([buyer]).to_csv(OUT_DIR / 'mechanism_buyer_composition.csv', index=False)

    # Summary
    print('\n5. Mechanism Summary')
    print('-' * 40)
    summary = mechanism_summary_table(all_results)
    if not summary.empty:
        print(summary.to_string(index=False))
        summary.to_csv(OUT_DIR / 'mechanism_summary.csv', index=False)

    print('\n' + '=' * 60)
    print('MECHANISM ANALYSIS COMPLETE')
    print('=' * 60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Mechanism analysis')
    parser.add_argument('--boundary', '-b', default='inund')
    parser.add_argument('--caliper', '-c', type=int, default=300)
    args = parser.parse_args()
    main(args.boundary, args.caliper)
