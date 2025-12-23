#!/usr/bin/env python3
"""
Module: placebo_tests.py
Purpose: Falsification tests to validate causal identification.

This module implements comprehensive placebo and falsification tests
to strengthen causal identification in the boundary RD-in-panel design.

Tests
-----
1. Placebo Event Dates: Test DiD at fake event dates
2. Placebo Boundaries: Shift boundary location and re-estimate
3. Permutation Inference: Randomization-based p-values
4. Leave-One-Out: Stability to dropping individual months
5. Triple Difference: Use SFHA as additional control for inundation

Input Files
-----------
- data_work/panel_parcel_month.parquet

Output Files
------------
- data_work/diagnostics/placebo_event_dates.csv
- data_work/diagnostics/placebo_boundaries.csv
- data_work/diagnostics/permutation_pvalues.csv
- data_work/diagnostics/leave_one_out.csv
- data_work/diagnostics/triple_diff.csv
- figures/fig_placebo_event_dates.png
- figures/fig_permutation_distribution.png
- figures/fig_placebo_boundaries.png

Notes
-----
These tests are designed to detect spurious effects that could arise
from model misspecification, data artifacts, or confounding.
"""
from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path
from typing import Optional, List, Dict
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from tqdm import tqdm

PANEL = Path('data_work/panel_parcel_month.parquet')
OUT_DIR = Path('data_work/diagnostics')
FIG_DIR = Path('figures')


def ensure_env():
    """Verify virtual environment is activated."""
    if not os.getenv('VIRTUAL_ENV') or not os.getenv('VIRTUAL_ENV').endswith('/.venv'):
        print('ERROR: Please activate project .venv', file=sys.stderr)
        sys.exit(1)


def months_since_event(ym: str, event_ym: str = '2019-03') -> int:
    """
    Convert year-month string to event time relative to specified event.

    Parameters
    ----------
    ym : str
        Year-month in 'YYYY-MM' format.
    event_ym : str
        Event year-month (default: '2019-03').

    Returns
    -------
    int
        Months since event (negative for pre-event).
    """
    y, m = ym.split('-')
    ey, em = event_ym.split('-')
    y, m = int(y), int(m)
    ey, em = int(ey), int(em)
    return (y - ey) * 12 + (m - em)


def estimate_did(
    panel: pd.DataFrame,
    boundary: str,
    caliper: int,
    event_ym: str = '2019-03'
) -> Dict:
    """
    Estimate simple DiD at boundary for given event date.

    Parameters
    ----------
    panel : pd.DataFrame
        Panel data.
    boundary : str
        Boundary type: 'inund' or 'sfha'.
    caliper : int
        Caliper window in meters.
    event_ym : str
        Event year-month for defining post period.

    Returns
    -------
    dict
        DiD estimate, SE, CI, and sample size.
    """
    dist_col = f'signed_dist_{boundary}_m'
    inside_col = f'inside_{boundary}'

    if dist_col not in panel.columns or inside_col not in panel.columns:
        return {'coef': np.nan, 'se': np.nan, 'n_obs': 0}

    d = panel.copy()
    d = d[np.isfinite(d[dist_col])]
    d = d[d[dist_col].abs() <= caliper].copy()

    d['event_m'] = d['ym'].apply(lambda x: months_since_event(x, event_ym))
    d['post'] = (d['event_m'] >= 0).astype(int)
    d['inside'] = (d[inside_col] == 1).astype(int)

    X = pd.DataFrame({
        'const': 1.0,
        'inside': d['inside'].astype(float),
        'post': d['post'].astype(float),
        'inside_x_post': (d['inside'] * d['post']).astype(float)
    })
    y = d['sold_this_month'].astype(float)

    try:
        model = sm.OLS(y, X).fit(cov_type='HC3')
        coef = model.params.get('inside_x_post', np.nan)
        se = model.bse.get('inside_x_post', np.nan)
        pval = model.pvalues.get('inside_x_post', np.nan)
    except Exception:
        coef, se, pval = np.nan, np.nan, np.nan

    return {
        'coef': coef,
        'se': se,
        'ci_lo': coef - 1.96 * se if np.isfinite(se) else np.nan,
        'ci_hi': coef + 1.96 * se if np.isfinite(se) else np.nan,
        'pval': pval,
        'n_obs': len(d)
    }


def placebo_event_dates(
    panel: pd.DataFrame,
    boundary: str = 'inund',
    caliper: int = 300,
    fake_dates: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Test DiD at fake event dates to check for spurious effects.

    Parameters
    ----------
    panel : pd.DataFrame
        Panel data.
    boundary : str
        Boundary type.
    caliper : int
        Caliper window.
    fake_dates : list of str, optional
        Fake event dates. Default: one year before/after true event.

    Returns
    -------
    pd.DataFrame
        DiD estimates at each placebo date.
    """
    if fake_dates is None:
        fake_dates = ['2017-03', '2018-03', '2020-03', '2021-03']

    # Include true event for comparison
    all_dates = ['2019-03'] + fake_dates

    results = []
    for event_ym in all_dates:
        result = estimate_did(panel, boundary, caliper, event_ym)
        results.append({
            'event_date': event_ym,
            'is_true_event': event_ym == '2019-03',
            'boundary': boundary,
            'caliper_m': caliper,
            **result
        })

    return pd.DataFrame(results)


def placebo_boundaries(
    panel: pd.DataFrame,
    boundary: str = 'inund',
    caliper: int = 300,
    shift_meters: Optional[List[int]] = None
) -> pd.DataFrame:
    """
    Test DiD with shifted boundary locations.

    Shifts the boundary inward/outward by specified amounts and
    re-estimates effects to test boundary specificity.

    Parameters
    ----------
    panel : pd.DataFrame
        Panel data.
    boundary : str
        Boundary type.
    caliper : int
        Caliper window.
    shift_meters : list of int, optional
        Amounts to shift boundary (negative = inward).
        Default: [-100, -50, 0, 50, 100, 150, 200].

    Returns
    -------
    pd.DataFrame
        DiD estimates at each shifted boundary.
    """
    if shift_meters is None:
        shift_meters = [-100, -50, 0, 50, 100, 150, 200]

    dist_col = f'signed_dist_{boundary}_m'
    inside_col = f'inside_{boundary}'

    if dist_col not in panel.columns:
        print(f'WARNING: {dist_col} not found')
        return pd.DataFrame()

    results = []
    for shift in shift_meters:
        # Create shifted inside indicator
        d = panel.copy()
        d = d[np.isfinite(d[dist_col])]

        # Shift: positive shift moves boundary outward (more parcels inside)
        # Negative shift moves boundary inward (fewer parcels inside)
        d['inside_shifted'] = (d[dist_col] <= shift).astype(int)

        # Filter to caliper around SHIFTED boundary
        d['dist_to_shifted'] = d[dist_col] - shift
        d = d[d['dist_to_shifted'].abs() <= caliper].copy()

        d['event_m'] = d['ym'].apply(months_since_event)
        d['post'] = (d['event_m'] >= 0).astype(int)

        X = pd.DataFrame({
            'const': 1.0,
            'inside': d['inside_shifted'].astype(float),
            'post': d['post'].astype(float),
            'inside_x_post': (d['inside_shifted'] * d['post']).astype(float)
        })
        y = d['sold_this_month'].astype(float)

        try:
            model = sm.OLS(y, X).fit(cov_type='HC3')
            coef = model.params.get('inside_x_post', np.nan)
            se = model.bse.get('inside_x_post', np.nan)
            pval = model.pvalues.get('inside_x_post', np.nan)
        except Exception:
            coef, se, pval = np.nan, np.nan, np.nan

        results.append({
            'shift_m': shift,
            'is_true_boundary': shift == 0,
            'boundary': boundary,
            'caliper_m': caliper,
            'coef': coef,
            'se': se,
            'ci_lo': coef - 1.96 * se if np.isfinite(se) else np.nan,
            'ci_hi': coef + 1.96 * se if np.isfinite(se) else np.nan,
            'pval': pval,
            'n_obs': len(d)
        })

    return pd.DataFrame(results)


def permutation_inference(
    panel: pd.DataFrame,
    boundary: str = 'inund',
    caliper: int = 300,
    n_permutations: int = 1000,
    seed: int = 42
) -> Dict:
    """
    Randomization inference via treatment permutation.

    Permutes treatment assignment across parcels and re-estimates
    DiD to construct a null distribution for p-value computation.

    Parameters
    ----------
    panel : pd.DataFrame
        Panel data.
    boundary : str
        Boundary type.
    caliper : int
        Caliper window.
    n_permutations : int
        Number of permutations.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict
        True estimate, permutation distribution, and exact p-value.
    """
    np.random.seed(seed)

    dist_col = f'signed_dist_{boundary}_m'
    inside_col = f'inside_{boundary}'

    if dist_col not in panel.columns:
        return {'true_coef': np.nan, 'pval_perm': np.nan}

    # Prepare data
    d = panel.copy()
    d = d[np.isfinite(d[dist_col])]
    d = d[d[dist_col].abs() <= caliper].copy()
    d['event_m'] = d['ym'].apply(months_since_event)
    d['post'] = (d['event_m'] >= 0).astype(int)
    d['inside'] = (d[inside_col] == 1).astype(int)

    # Get parcel IDs
    parcel_ids = d['parcel_id'].unique() if 'parcel_id' in d.columns else d.index.unique()
    parcel_treatment = d.groupby('parcel_id' if 'parcel_id' in d.columns else d.index.name)['inside'].first()

    # True estimate
    true_result = estimate_did(panel, boundary, caliper)
    true_coef = true_result['coef']

    # Permutation distribution
    perm_coefs = []
    for _ in tqdm(range(n_permutations), desc='Permutation inference'):
        # Shuffle treatment assignment
        shuffled_treatment = parcel_treatment.sample(frac=1, replace=False)
        shuffled_treatment.index = parcel_treatment.index

        # Map back to panel
        d_perm = d.copy()
        if 'parcel_id' in d_perm.columns:
            d_perm['inside_perm'] = d_perm['parcel_id'].map(shuffled_treatment)
        else:
            d_perm['inside_perm'] = d_perm.index.map(shuffled_treatment)

        # Estimate
        X = pd.DataFrame({
            'const': 1.0,
            'inside': d_perm['inside_perm'].astype(float),
            'post': d_perm['post'].astype(float),
            'inside_x_post': (d_perm['inside_perm'] * d_perm['post']).astype(float)
        })
        y = d_perm['sold_this_month'].astype(float)

        try:
            model = sm.OLS(y, X).fit()
            perm_coefs.append(model.params.get('inside_x_post', np.nan))
        except Exception:
            perm_coefs.append(np.nan)

    perm_coefs = np.array([c for c in perm_coefs if np.isfinite(c)])

    # Two-sided p-value
    if len(perm_coefs) > 0 and np.isfinite(true_coef):
        pval_perm = np.mean(np.abs(perm_coefs) >= np.abs(true_coef))
    else:
        pval_perm = np.nan

    return {
        'boundary': boundary,
        'caliper_m': caliper,
        'true_coef': true_coef,
        'perm_mean': np.mean(perm_coefs),
        'perm_std': np.std(perm_coefs),
        'pval_perm': pval_perm,
        'n_permutations': len(perm_coefs),
        'perm_distribution': perm_coefs
    }


def leave_one_out_sensitivity(
    panel: pd.DataFrame,
    boundary: str = 'inund',
    caliper: int = 300
) -> pd.DataFrame:
    """
    Leave-one-month-out sensitivity analysis.

    Drops each month one at a time and re-estimates to assess
    sensitivity to individual time periods.

    Parameters
    ----------
    panel : pd.DataFrame
        Panel data.
    boundary : str
        Boundary type.
    caliper : int
        Caliper window.

    Returns
    -------
    pd.DataFrame
        DiD estimates with each month dropped.
    """
    dist_col = f'signed_dist_{boundary}_m'
    if dist_col not in panel.columns:
        return pd.DataFrame()

    d = panel.copy()
    d = d[np.isfinite(d[dist_col])]
    d = d[d[dist_col].abs() <= caliper].copy()

    months = sorted(d['ym'].unique())

    results = []

    # Full sample estimate
    full_result = estimate_did(panel, boundary, caliper)
    results.append({
        'dropped_month': 'None (full sample)',
        'boundary': boundary,
        'caliper_m': caliper,
        **full_result
    })

    # Leave one out
    for ym in months:
        panel_loo = panel[panel['ym'] != ym]
        loo_result = estimate_did(panel_loo, boundary, caliper)
        results.append({
            'dropped_month': ym,
            'boundary': boundary,
            'caliper_m': caliper,
            **loo_result
        })

    return pd.DataFrame(results)


def triple_difference(
    panel: pd.DataFrame,
    caliper: int = 300
) -> Dict:
    """
    Triple difference using SFHA as additional control for inundation.

    Estimates: (Inside_inund × Post) - (Inside_sfha × Post)
    This controls for any SFHA-specific trends.

    Parameters
    ----------
    panel : pd.DataFrame
        Panel data.
    caliper : int
        Caliper window.

    Returns
    -------
    dict
        Triple-diff estimate and components.
    """
    # Need both boundaries
    inund_dist = 'signed_dist_inund_m'
    sfha_dist = 'signed_dist_sfha_m'

    if inund_dist not in panel.columns or sfha_dist not in panel.columns:
        return {'triple_diff': np.nan}

    # Sample: within caliper of BOTH boundaries
    d = panel.copy()
    d = d[np.isfinite(d[inund_dist]) & np.isfinite(d[sfha_dist])]
    d = d[(d[inund_dist].abs() <= caliper) & (d[sfha_dist].abs() <= caliper)].copy()

    d['event_m'] = d['ym'].apply(months_since_event)
    d['post'] = (d['event_m'] >= 0).astype(int)
    d['inside_inund'] = (d['inside_inund'] == 1).astype(int)
    d['inside_sfha'] = (d['inside_sfha'] == 1).astype(int)

    # Estimate both DiDs
    # Inundation DiD
    X_inund = pd.DataFrame({
        'const': 1.0,
        'inside': d['inside_inund'].astype(float),
        'post': d['post'].astype(float),
        'inside_x_post': (d['inside_inund'] * d['post']).astype(float)
    })
    y = d['sold_this_month'].astype(float)

    try:
        model_inund = sm.OLS(y, X_inund).fit(cov_type='HC3')
        did_inund = model_inund.params.get('inside_x_post', np.nan)
        se_inund = model_inund.bse.get('inside_x_post', np.nan)
    except Exception:
        did_inund, se_inund = np.nan, np.nan

    # SFHA DiD (same sample)
    X_sfha = pd.DataFrame({
        'const': 1.0,
        'inside': d['inside_sfha'].astype(float),
        'post': d['post'].astype(float),
        'inside_x_post': (d['inside_sfha'] * d['post']).astype(float)
    })

    try:
        model_sfha = sm.OLS(y, X_sfha).fit(cov_type='HC3')
        did_sfha = model_sfha.params.get('inside_x_post', np.nan)
        se_sfha = model_sfha.bse.get('inside_x_post', np.nan)
    except Exception:
        did_sfha, se_sfha = np.nan, np.nan

    # Triple diff: difference of DiDs
    triple_diff = did_inund - did_sfha
    # Approximate SE (assuming independence)
    se_triple = np.sqrt(se_inund**2 + se_sfha**2) if np.isfinite(se_inund) and np.isfinite(se_sfha) else np.nan

    return {
        'caliper_m': caliper,
        'did_inund': did_inund,
        'se_inund': se_inund,
        'did_sfha': did_sfha,
        'se_sfha': se_sfha,
        'triple_diff': triple_diff,
        'se_triple': se_triple,
        'ci_lo_triple': triple_diff - 1.96 * se_triple if np.isfinite(se_triple) else np.nan,
        'ci_hi_triple': triple_diff + 1.96 * se_triple if np.isfinite(se_triple) else np.nan,
        'n_obs': len(d)
    }


def plot_placebo_events(results: pd.DataFrame, out_path: Path) -> None:
    """Plot placebo event date results."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Sort by date
    results = results.sort_values('event_date')

    # Color by true vs placebo
    colors = ['red' if r else 'steelblue' for r in results['is_true_event']]

    ax.errorbar(
        range(len(results)),
        results['coef'],
        yerr=[results['coef'] - results['ci_lo'], results['ci_hi'] - results['coef']],
        fmt='o',
        capsize=4,
        markersize=8,
        color='gray'
    )

    # Highlight points
    for i, (_, row) in enumerate(results.iterrows()):
        color = 'red' if row['is_true_event'] else 'steelblue'
        ax.scatter([i], [row['coef']], color=color, s=100, zorder=5)

    ax.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
    ax.set_xticks(range(len(results)))
    ax.set_xticklabels(results['event_date'], rotation=45)
    ax.set_xlabel('Event Date')
    ax.set_ylabel('DiD Coefficient')
    ax.set_title('Placebo Event Dates (Red = True Event)')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Wrote {out_path}')


def plot_permutation_dist(perm_result: Dict, out_path: Path) -> None:
    """Plot permutation distribution with true estimate."""
    fig, ax = plt.subplots(figsize=(10, 6))

    perm_coefs = perm_result['perm_distribution']
    true_coef = perm_result['true_coef']

    ax.hist(perm_coefs, bins=50, density=True, alpha=0.7, color='steelblue', label='Permutation null')
    ax.axvline(x=true_coef, color='red', linewidth=2, label=f'True estimate: {true_coef:.4f}')
    ax.axvline(x=-abs(true_coef), color='red', linewidth=1, linestyle='--', alpha=0.5)
    ax.axvline(x=abs(true_coef), color='red', linewidth=1, linestyle='--', alpha=0.5)

    ax.set_xlabel('DiD Coefficient')
    ax.set_ylabel('Density')
    ax.set_title(f"Permutation Inference: p = {perm_result['pval_perm']:.3f}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Wrote {out_path}')


def plot_placebo_boundaries(results: pd.DataFrame, out_path: Path) -> None:
    """Plot boundary shift sensitivity."""
    fig, ax = plt.subplots(figsize=(10, 6))

    results = results.sort_values('shift_m')

    ax.fill_between(
        results['shift_m'],
        results['ci_lo'],
        results['ci_hi'],
        alpha=0.3,
        color='steelblue'
    )
    ax.plot(results['shift_m'], results['coef'], 'o-', color='steelblue', markersize=6)

    # Highlight true boundary
    true_row = results[results['is_true_boundary']]
    if not true_row.empty:
        ax.scatter(true_row['shift_m'], true_row['coef'], color='red', s=150, zorder=5,
                   label='True boundary')

    ax.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
    ax.axvline(x=0, color='gray', linestyle=':', linewidth=1, alpha=0.5)

    ax.set_xlabel('Boundary Shift (meters; negative = inward)')
    ax.set_ylabel('DiD Coefficient')
    ax.set_title('Placebo Boundaries: Sensitivity to Boundary Location')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Wrote {out_path}')


def main(
    boundary: str = 'inund',
    caliper: int = 300,
    n_permutations: int = 500,
    run_all: bool = True
):
    """
    Run all placebo and falsification tests.

    Parameters
    ----------
    boundary : str
        Boundary type for testing.
    caliper : int
        Caliper window.
    n_permutations : int
        Number of permutations for randomization inference.
    run_all : bool
        If True, run all tests.
    """
    ensure_env()

    if not PANEL.exists():
        raise SystemExit(f'Missing {PANEL}')

    panel = pd.read_parquet(PANEL)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    print('=' * 60)
    print('PLACEBO AND FALSIFICATION TESTS')
    print('=' * 60)
    print(f'Boundary: {boundary}')
    print(f'Caliper: ±{caliper}m')
    print()

    # 1. Placebo Event Dates
    print('1. Testing placebo event dates...')
    pe_results = placebo_event_dates(panel, boundary, caliper)
    pe_path = OUT_DIR / 'placebo_event_dates.csv'
    pe_results.to_csv(pe_path, index=False)
    print(f'   Wrote {pe_path}')
    plot_placebo_events(pe_results, FIG_DIR / 'fig_placebo_event_dates.png')

    # Summary
    true_coef = pe_results[pe_results['is_true_event']]['coef'].values[0]
    placebo_coefs = pe_results[~pe_results['is_true_event']]['coef']
    print(f'   True event coef: {true_coef:.6f}')
    print(f'   Placebo mean: {placebo_coefs.mean():.6f} (range: {placebo_coefs.min():.6f} to {placebo_coefs.max():.6f})')
    print()

    # 2. Placebo Boundaries
    print('2. Testing placebo boundaries...')
    pb_results = placebo_boundaries(panel, boundary, caliper)
    pb_path = OUT_DIR / 'placebo_boundaries.csv'
    pb_results.to_csv(pb_path, index=False)
    print(f'   Wrote {pb_path}')
    plot_placebo_boundaries(pb_results, FIG_DIR / 'fig_placebo_boundaries.png')
    print()

    # 3. Permutation Inference
    print(f'3. Running permutation inference ({n_permutations} permutations)...')
    warnings.filterwarnings('ignore')
    perm_result = permutation_inference(panel, boundary, caliper, n_permutations)
    warnings.filterwarnings('default')

    perm_path = OUT_DIR / 'permutation_pvalues.csv'
    perm_summary = {k: v for k, v in perm_result.items() if k != 'perm_distribution'}
    pd.DataFrame([perm_summary]).to_csv(perm_path, index=False)
    print(f'   Wrote {perm_path}')

    if 'perm_distribution' in perm_result and len(perm_result['perm_distribution']) > 0:
        plot_permutation_dist(perm_result, FIG_DIR / 'fig_permutation_distribution.png')

    print(f"   Exact p-value: {perm_result['pval_perm']:.4f}")
    print()

    # 4. Leave-One-Out
    print('4. Running leave-one-out sensitivity...')
    loo_results = leave_one_out_sensitivity(panel, boundary, caliper)
    loo_path = OUT_DIR / 'leave_one_out.csv'
    loo_results.to_csv(loo_path, index=False)
    print(f'   Wrote {loo_path}')

    # Summary
    full_coef = loo_results[loo_results['dropped_month'] == 'None (full sample)']['coef'].values[0]
    loo_coefs = loo_results[loo_results['dropped_month'] != 'None (full sample)']['coef']
    print(f'   Full sample coef: {full_coef:.6f}')
    print(f'   LOO range: {loo_coefs.min():.6f} to {loo_coefs.max():.6f}')
    print(f'   Max absolute deviation: {(loo_coefs - full_coef).abs().max():.6f}')
    print()

    # 5. Triple Difference
    print('5. Computing triple difference (inundation - SFHA)...')
    triple_result = triple_difference(panel, caliper)
    triple_path = OUT_DIR / 'triple_diff.csv'
    pd.DataFrame([triple_result]).to_csv(triple_path, index=False)
    print(f'   Wrote {triple_path}')
    print(f"   Inund DiD: {triple_result['did_inund']:.6f}")
    print(f"   SFHA DiD: {triple_result['did_sfha']:.6f}")
    print(f"   Triple diff: {triple_result['triple_diff']:.6f} (SE: {triple_result['se_triple']:.6f})")

    print()
    print('=' * 60)
    print('FALSIFICATION TESTS COMPLETE')
    print('=' * 60)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Placebo and falsification tests for RD identification.'
    )
    parser.add_argument(
        '--boundary', '-b',
        choices=['inund', 'sfha'],
        default='inund',
        help='Boundary type. Default: inund'
    )
    parser.add_argument(
        '--caliper', '-c',
        type=int,
        default=300,
        help='Caliper window in meters. Default: 300'
    )
    parser.add_argument(
        '--n-permutations',
        type=int,
        default=500,
        help='Number of permutations. Default: 500'
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(
        boundary=args.boundary,
        caliper=args.caliper,
        n_permutations=args.n_permutations
    )
