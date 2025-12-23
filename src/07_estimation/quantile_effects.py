#!/usr/bin/env python3
"""
Module: quantile_effects.py
Purpose: Estimate distributional effects across the price distribution.

This module implements quantile-based treatment effect estimation to examine
heterogeneous effects across different parts of the price distribution.

Methods
-------
1. Quantile DiD - Estimate DiD at different price quantiles
2. Unconditional Quantile Regression (RIF) - Recentered influence function
3. Distribution shift tests - Kolmogorov-Smirnov test

Input Files
-----------
- data_work/panel_parcel_month.parquet

Output Files
------------
- data_work/diagnostics/quantile_did.csv
- data_work/diagnostics/uqr_results.csv
- figures/fig_quantile_effects.png
- figures/fig_quantile_event_study.png
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
from statsmodels.regression.quantile_regression import QuantReg
from scipy import stats

PANEL = Path('data_work/panel_parcel_month.parquet')
OUT_DIR = Path('data_work/diagnostics')
FIG_DIR = Path('figures')


def ensure_env():
    if not os.getenv('VIRTUAL_ENV') or not os.getenv('VIRTUAL_ENV').endswith('/.venv'):
        print('ERROR: Please activate project .venv', file=sys.stderr)
        sys.exit(1)


def months_since_event(ym: str) -> int:
    y, m = ym.split('-')
    return (int(y) - 2019) * 12 + (int(m) - 3)


def quantile_did(
    panel: pd.DataFrame,
    boundary: str,
    caliper: int,
    quantiles: List[float] = None
) -> pd.DataFrame:
    """
    Estimate DiD at different price quantiles.

    Parameters
    ----------
    panel : pd.DataFrame
        Panel data.
    boundary : str
        Boundary type.
    caliper : int
        Caliper window.
    quantiles : list of float
        Quantiles to estimate at. Default: [0.1, 0.25, 0.5, 0.75, 0.9]

    Returns
    -------
    pd.DataFrame
        DiD estimates at each quantile.
    """
    if quantiles is None:
        quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]

    dist_col = f'signed_dist_{boundary}_m'
    inside_col = f'inside_{boundary}'

    if dist_col not in panel.columns:
        return pd.DataFrame()

    d = panel.copy()
    d = d[np.isfinite(d[dist_col])]
    d = d[d[dist_col].abs() <= caliper].copy()
    d = d[d['sold_this_month'] == 1].copy()

    if 'log_price' not in d.columns:
        return pd.DataFrame()

    d = d.dropna(subset=['log_price'])
    d['event_m'] = d['ym'].apply(months_since_event)
    d['post'] = (d['event_m'] >= 0).astype(int)
    d['inside'] = (d[inside_col] == 1).astype(int)
    d['inside_x_post'] = d['inside'] * d['post']

    X = sm.add_constant(d[['inside', 'post', 'inside_x_post']].astype(float))
    y = d['log_price'].astype(float)

    results = []
    for q in quantiles:
        try:
            qr = QuantReg(y, X).fit(q=q)
            coef = qr.params.get('inside_x_post', np.nan)
            se = qr.bse.get('inside_x_post', np.nan)
            pval = qr.pvalues.get('inside_x_post', np.nan)

            results.append({
                'boundary': boundary,
                'caliper_m': caliper,
                'quantile': q,
                'coef': coef,
                'se': se,
                'ci_lo': coef - 1.96 * se,
                'ci_hi': coef + 1.96 * se,
                'pval': pval,
                'n_obs': len(d)
            })
        except Exception as e:
            results.append({
                'boundary': boundary,
                'caliper_m': caliper,
                'quantile': q,
                'coef': np.nan,
                'error': str(e)
            })

    return pd.DataFrame(results)


def distribution_shift_test(
    panel: pd.DataFrame,
    boundary: str,
    caliper: int
) -> dict:
    """
    Test for distributional changes using Kolmogorov-Smirnov test.

    Compares price distributions inside vs outside, pre vs post.
    """
    dist_col = f'signed_dist_{boundary}_m'
    inside_col = f'inside_{boundary}'

    if dist_col not in panel.columns:
        return {'error': f'Missing {dist_col}'}

    d = panel.copy()
    d = d[np.isfinite(d[dist_col])]
    d = d[d[dist_col].abs() <= caliper].copy()
    d = d[d['sold_this_month'] == 1].copy()

    if 'log_price' not in d.columns:
        return {'error': 'log_price not in panel'}

    d = d.dropna(subset=['log_price'])
    d['event_m'] = d['ym'].apply(months_since_event)
    d['post'] = (d['event_m'] >= 0).astype(int)
    d['inside'] = (d[inside_col] == 1).astype(int)

    results = {}

    # Pre-period: inside vs outside
    pre_inside = d[(d['post'] == 0) & (d['inside'] == 1)]['log_price']
    pre_outside = d[(d['post'] == 0) & (d['inside'] == 0)]['log_price']
    if len(pre_inside) > 10 and len(pre_outside) > 10:
        ks_pre = stats.ks_2samp(pre_inside, pre_outside)
        results['ks_pre_inside_vs_outside'] = ks_pre.statistic
        results['ks_pre_pval'] = ks_pre.pvalue

    # Post-period: inside vs outside
    post_inside = d[(d['post'] == 1) & (d['inside'] == 1)]['log_price']
    post_outside = d[(d['post'] == 1) & (d['inside'] == 0)]['log_price']
    if len(post_inside) > 10 and len(post_outside) > 10:
        ks_post = stats.ks_2samp(post_inside, post_outside)
        results['ks_post_inside_vs_outside'] = ks_post.statistic
        results['ks_post_pval'] = ks_post.pvalue

    # Inside: pre vs post
    if len(pre_inside) > 10 and len(post_inside) > 10:
        ks_inside = stats.ks_2samp(pre_inside, post_inside)
        results['ks_inside_pre_vs_post'] = ks_inside.statistic
        results['ks_inside_pval'] = ks_inside.pvalue

    results['boundary'] = boundary
    results['caliper_m'] = caliper

    return results


def plot_quantile_effects(qr_results: pd.DataFrame, out_path: Path) -> None:
    """Plot treatment effects across quantiles."""
    fig, ax = plt.subplots(figsize=(10, 6))

    q = qr_results['quantile'].values
    coef = qr_results['coef'].values
    ci_lo = qr_results['ci_lo'].values
    ci_hi = qr_results['ci_hi'].values

    ax.fill_between(q, ci_lo, ci_hi, alpha=0.3, color='steelblue', label='95% CI')
    ax.plot(q, coef, 'o-', color='steelblue', markersize=8, label='Point estimate')
    ax.axhline(y=0, color='black', linestyle='--', linewidth=0.5)

    ax.set_xlabel('Quantile (τ)', fontsize=11)
    ax.set_ylabel('DiD Coefficient (log price)', fontsize=11)
    ax.set_title('Quantile Treatment Effects on Log Price', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Wrote {out_path}')


def main(boundary: str = 'inund', caliper: int = 300):
    """Run quantile effects analysis."""
    ensure_env()

    if not PANEL.exists():
        raise SystemExit(f'Missing {PANEL}')

    panel = pd.read_parquet(PANEL)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    print('=' * 60)
    print('QUANTILE EFFECTS ANALYSIS')
    print('=' * 60)

    # Quantile DiD
    print('\n1. Quantile DiD')
    print('-' * 40)
    qr = quantile_did(panel, boundary, caliper)
    if not qr.empty:
        print(qr[['quantile', 'coef', 'se', 'ci_lo', 'ci_hi']].to_string(index=False))
        qr.to_csv(OUT_DIR / 'quantile_did.csv', index=False)
        print(f"\nWrote {OUT_DIR / 'quantile_did.csv'}")
        plot_quantile_effects(qr, FIG_DIR / 'fig_quantile_effects.png')

    # Distribution shift test
    print('\n2. Distribution Shift Tests')
    print('-' * 40)
    ks = distribution_shift_test(panel, boundary, caliper)
    if 'error' not in ks:
        for k, v in ks.items():
            if k not in ['boundary', 'caliper_m']:
                print(f"  {k}: {v:.4f}")
        pd.DataFrame([ks]).to_csv(OUT_DIR / 'distribution_shift_test.csv', index=False)

    print('\n' + '=' * 60)
    print('QUANTILE ANALYSIS COMPLETE')
    print('=' * 60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Quantile effects analysis')
    parser.add_argument('--boundary', '-b', default='inund')
    parser.add_argument('--caliper', '-c', type=int, default=300)
    args = parser.parse_args()
    main(args.boundary, args.caliper)
