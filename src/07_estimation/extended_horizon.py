#!/usr/bin/env python3
"""
Module: extended_horizon.py
Purpose: Extend analysis beyond the standard ±24 month window.

This module implements persistence analysis to test whether treatment
effects persist, grow, or fade over extended time horizons.

Input Files
-----------
- data_work/panel_parcel_month.parquet

Output Files
------------
- data_work/diagnostics/dynamic_effects_extended.csv
- data_work/diagnostics/persistence_analysis.csv
- figures/fig_extended_event_study.png
- figures/fig_cumulative_effects.png
"""
from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path
from typing import Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

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


def dynamic_treatment_effects(
    panel: pd.DataFrame,
    boundary: str,
    caliper: int,
    window: Tuple[int, int] = (-36, 36)
) -> pd.DataFrame:
    """
    Estimate dynamic treatment effects with extended horizon.
    """
    dist_col = f'signed_dist_{boundary}_m'
    inside_col = f'inside_{boundary}'

    if dist_col not in panel.columns:
        return pd.DataFrame()

    d = panel.copy()
    d = d[np.isfinite(d[dist_col])]
    d = d[d[dist_col].abs() <= caliper].copy()

    d['event_m'] = d['ym'].apply(months_since_event)
    d = d[(d['event_m'] >= window[0]) & (d['event_m'] <= window[1])].copy()
    d['inside'] = (d[inside_col] == 1).astype(int)

    event_times = sorted([t for t in d['event_m'].unique() if t != -1])
    results = []

    for t in event_times:
        d_t = d[d['event_m'].isin([t, -1])].copy()
        d_t['period_t'] = (d_t['event_m'] == t).astype(int)

        X = pd.DataFrame({
            'const': 1.0,
            'inside': d_t['inside'].astype(float),
            'period_t': d_t['period_t'].astype(float),
            'inside_x_period': (d_t['inside'] * d_t['period_t']).astype(float)
        })
        y = d_t['sold_this_month'].astype(float)

        try:
            model = sm.OLS(y, X).fit(cov_type='HC3')
            coef = model.params.get('inside_x_period', np.nan)
            se = model.bse.get('inside_x_period', np.nan)
        except Exception:
            coef, se = np.nan, np.nan

        results.append({
            'boundary': boundary,
            'caliper_m': caliper,
            'event_m': t,
            'coef': coef,
            'se': se,
            'ci_lo': coef - 1.96 * se if np.isfinite(se) else np.nan,
            'ci_hi': coef + 1.96 * se if np.isfinite(se) else np.nan,
            'n_obs': len(d_t)
        })

    results.append({
        'boundary': boundary, 'caliper_m': caliper, 'event_m': -1,
        'coef': 0.0, 'se': 0.0, 'ci_lo': 0.0, 'ci_hi': 0.0,
        'n_obs': len(d[d['event_m'] == -1])
    })

    return pd.DataFrame(results).sort_values('event_m')


def persistence_analysis(
    panel: pd.DataFrame,
    boundary: str,
    caliper: int,
    periods: list = None
) -> pd.DataFrame:
    """
    Test effect persistence across different time windows.
    """
    if periods is None:
        periods = [(0, 6), (6, 12), (12, 18), (18, 24), (24, 36)]

    dist_col = f'signed_dist_{boundary}_m'
    inside_col = f'inside_{boundary}'

    if dist_col not in panel.columns:
        return pd.DataFrame()

    d = panel.copy()
    d = d[np.isfinite(d[dist_col])]
    d = d[d[dist_col].abs() <= caliper].copy()
    d['event_m'] = d['ym'].apply(months_since_event)
    d['inside'] = (d[inside_col] == 1).astype(int)

    results = []
    for start, end in periods:
        d_period = d[(d['event_m'] >= start) & (d['event_m'] < end)].copy()
        if len(d_period) < 100:
            continue

        X = sm.add_constant(d_period[['inside']].astype(float))
        y = d_period['sold_this_month'].astype(float)

        try:
            model = sm.OLS(y, X).fit(cov_type='HC3')
            coef = model.params.get('inside', np.nan)
            se = model.bse.get('inside', np.nan)
        except Exception:
            coef, se = np.nan, np.nan

        results.append({
            'boundary': boundary,
            'caliper_m': caliper,
            'period': f'{start}-{end}m',
            'start_month': start,
            'end_month': end,
            'coef': coef,
            'se': se,
            'ci_lo': coef - 1.96 * se if np.isfinite(se) else np.nan,
            'ci_hi': coef + 1.96 * se if np.isfinite(se) else np.nan,
            'n_obs': len(d_period)
        })

    return pd.DataFrame(results)


def plot_extended_event_study(es: pd.DataFrame, out_path: Path) -> None:
    """Plot extended event study."""
    fig, ax = plt.subplots(figsize=(12, 6))

    es = es.sort_values('event_m')
    ax.fill_between(es['event_m'], es['ci_lo'], es['ci_hi'],
                    alpha=0.3, color='steelblue')
    ax.plot(es['event_m'], es['coef'], 'o-', color='steelblue', markersize=3)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.axvline(x=0, color='red', linestyle='--', alpha=0.7)
    ax.axvspan(es['event_m'].min(), 0, alpha=0.1, color='gray')

    ax.set_xlabel('Months Relative to Flood', fontsize=11)
    ax.set_ylabel('Treatment Effect', fontsize=11)
    ax.set_title('Extended Event Study (±36 months)', fontsize=12)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Wrote {out_path}')


def main(boundary: str = 'inund', caliper: int = 300):
    """Run extended horizon analysis."""
    ensure_env()

    if not PANEL.exists():
        raise SystemExit(f'Missing {PANEL}')

    panel = pd.read_parquet(PANEL)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    print('=' * 60)
    print('EXTENDED HORIZON ANALYSIS')
    print('=' * 60)

    # Check available time range
    panel['event_m'] = panel['ym'].apply(months_since_event)
    print(f"Available range: [{panel['event_m'].min()}, {panel['event_m'].max()}]")

    # Extended event study
    print('\n1. Extended Event Study')
    print('-' * 40)
    es = dynamic_treatment_effects(panel, boundary, caliper, window=(-36, 36))
    if not es.empty:
        es.to_csv(OUT_DIR / 'dynamic_effects_extended.csv', index=False)
        print(f"Wrote {OUT_DIR / 'dynamic_effects_extended.csv'}")
        plot_extended_event_study(es, FIG_DIR / 'fig_extended_event_study.png')

    # Persistence analysis
    print('\n2. Persistence Analysis')
    print('-' * 40)
    persist = persistence_analysis(panel, boundary, caliper)
    if not persist.empty:
        print(persist[['period', 'coef', 'se', 'n_obs']].to_string(index=False))
        persist.to_csv(OUT_DIR / 'persistence_analysis.csv', index=False)
        print(f"\nWrote {OUT_DIR / 'persistence_analysis.csv'}")

    print('\n' + '=' * 60)
    print('EXTENDED HORIZON COMPLETE')
    print('=' * 60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extended horizon analysis')
    parser.add_argument('--boundary', '-b', default='inund')
    parser.add_argument('--caliper', '-c', type=int, default=300)
    args = parser.parse_args()
    main(args.boundary, args.caliper)
