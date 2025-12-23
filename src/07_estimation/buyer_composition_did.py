#!/usr/bin/env python3
"""
Module: buyer_composition_did.py
Purpose: Analyze buyer composition shifts at SFHA boundary post-flood.

Tests whether post-flood purchases inside the SFHA reallocate toward:
1. LLCs vs. individuals
2. Multi-parcel (portfolio) owners vs. single-parcel buyers
3. More distant buyers (greater owner-parcel distance)

Input Files
-----------
- data_work/panel_parcel_month.parquet

Output Files
------------
- data_work/diagnostics/buyer_composition_did.csv
- data_work/diagnostics/buyer_distance_did.csv
- figures/fig_buyer_composition_event_study.png
"""
from __future__ import annotations
import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

PANEL = Path('data_work/panel_parcel_month.parquet')
OUT_DIR = Path('data_work/diagnostics')
FIG_DIR = Path('figures')

def ensure_env():
    """Verify virtual environment is activated."""
    if not os.getenv('VIRTUAL_ENV') or not os.getenv('VIRTUAL_ENV').endswith('/.venv'):
        print('ERROR: Please activate project .venv before running.', file=sys.stderr)
        sys.exit(1)

def months_since_event(ym: str) -> int:
    """Convert year-month string to event time relative to March 2019."""
    y, m = ym.split('-')
    return (int(y) - 2019) * 12 + (int(m) - 3)


def buyer_composition_did(panel: pd.DataFrame, boundary: str = 'sfha',
                          caliper: int = 300) -> pd.DataFrame:
    """
    Estimate DiD for buyer composition at boundary.

    Tests whether buyer types shift inside the SFHA post-flood.

    Parameters
    ----------
    panel : pd.DataFrame
        Panel with owner_form_snapshot and owner_scale_snapshot.
    boundary : str
        'sfha' or 'inund'.
    caliper : int
        Caliper width in meters.

    Returns
    -------
    pd.DataFrame
        DiD estimates for each buyer type.
    """
    dist_col = f'signed_dist_{boundary}_m'
    inside_col = f'inside_{boundary}'

    if dist_col not in panel.columns or inside_col not in panel.columns:
        return pd.DataFrame()

    # Filter to sales only and within caliper
    d = panel.copy()
    d = d[d['sold_this_month'] == 1]
    d = d[np.isfinite(d[dist_col])]
    d = d[d[dist_col].abs() <= caliper].copy()

    if len(d) == 0:
        return pd.DataFrame()

    d['event_m'] = d['ym'].apply(months_since_event)
    d['post'] = (d['event_m'] >= 0).astype(int)
    d['inside'] = (d[inside_col] == 1).astype(int)

    results = []

    # Test 1: LLC vs Individual
    if 'owner_form_snapshot' in d.columns:
        d['is_llc'] = (d['owner_form_snapshot'] == 'LLC').astype(int)

        # Shares by group
        for (inside, post), grp in d.groupby(['inside', 'post']):
            llc_share = grp['is_llc'].mean()
            results.append({
                'outcome': 'LLC_share',
                'inside': inside,
                'post': post,
                'mean': llc_share,
                'n': len(grp)
            })

        # DiD regression
        X = pd.DataFrame({
            'const': 1.0,
            'inside': d['inside'].astype(float),
            'post': d['post'].astype(float),
            'inside_x_post': (d['inside'] * d['post']).astype(float)
        })
        y = d['is_llc'].astype(float)

        if len(y) > 10:
            model = sm.OLS(y, X).fit(cov_type='HC3')
            results.append({
                'outcome': 'LLC_share',
                'statistic': 'DiD',
                'estimate': model.params.get('inside_x_post', np.nan),
                'se': model.bse.get('inside_x_post', np.nan),
                'pvalue': model.pvalues.get('inside_x_post', np.nan),
                'n': len(d)
            })

    # Test 2: Multi-parcel (portfolio) vs single-parcel
    if 'owner_scale_snapshot' in d.columns:
        d['is_portfolio'] = (d['owner_scale_snapshot'] == 'multi').astype(int)

        for (inside, post), grp in d.groupby(['inside', 'post']):
            portfolio_share = grp['is_portfolio'].mean()
            results.append({
                'outcome': 'portfolio_share',
                'inside': inside,
                'post': post,
                'mean': portfolio_share,
                'n': len(grp)
            })

        X = pd.DataFrame({
            'const': 1.0,
            'inside': d['inside'].astype(float),
            'post': d['post'].astype(float),
            'inside_x_post': (d['inside'] * d['post']).astype(float)
        })
        y = d['is_portfolio'].astype(float)

        if len(y) > 10:
            model = sm.OLS(y, X).fit(cov_type='HC3')
            results.append({
                'outcome': 'portfolio_share',
                'statistic': 'DiD',
                'estimate': model.params.get('inside_x_post', np.nan),
                'se': model.bse.get('inside_x_post', np.nan),
                'pvalue': model.pvalues.get('inside_x_post', np.nan),
                'n': len(d)
            })

    return pd.DataFrame(results)


def buyer_composition_event_study(panel: pd.DataFrame, boundary: str = 'sfha',
                                   caliper: int = 300) -> dict:
    """
    Event study for buyer composition by month.

    Parameters
    ----------
    panel : pd.DataFrame
        Panel data.
    boundary : str
        'sfha' or 'inund'.
    caliper : int
        Caliper width.

    Returns
    -------
    dict
        Event study coefficients for LLC share and portfolio share.
    """
    dist_col = f'signed_dist_{boundary}_m'
    inside_col = f'inside_{boundary}'

    d = panel.copy()
    d = d[d['sold_this_month'] == 1]
    d = d[np.isfinite(d[dist_col])]
    d = d[d[dist_col].abs() <= caliper].copy()

    if len(d) == 0:
        return {}

    d['event_m'] = d['ym'].apply(months_since_event)
    d['inside'] = (d[inside_col] == 1).astype(int)

    results = {'llc': {}, 'portfolio': {}}

    # For each event month, compute inside share - outside share
    for t in sorted(d['event_m'].unique()):
        month_data = d[d['event_m'] == t]

        if 'owner_form_snapshot' in d.columns:
            inside_llc = month_data[month_data['inside'] == 1]['owner_form_snapshot'].eq('LLC').mean()
            outside_llc = month_data[month_data['inside'] == 0]['owner_form_snapshot'].eq('LLC').mean()
            if np.isfinite(inside_llc) and np.isfinite(outside_llc):
                results['llc'][t] = {
                    'diff': inside_llc - outside_llc,
                    'inside': inside_llc,
                    'outside': outside_llc,
                    'n_inside': (month_data['inside'] == 1).sum(),
                    'n_outside': (month_data['inside'] == 0).sum()
                }

        if 'owner_scale_snapshot' in d.columns:
            inside_port = month_data[month_data['inside'] == 1]['owner_scale_snapshot'].eq('multi').mean()
            outside_port = month_data[month_data['inside'] == 0]['owner_scale_snapshot'].eq('multi').mean()
            if np.isfinite(inside_port) and np.isfinite(outside_port):
                results['portfolio'][t] = {
                    'diff': inside_port - outside_port,
                    'inside': inside_port,
                    'outside': outside_port,
                    'n_inside': (month_data['inside'] == 1).sum(),
                    'n_outside': (month_data['inside'] == 0).sum()
                }

    return results


def plot_buyer_composition_event_study(results: dict, out_path: Path):
    """
    Plot event study for buyer composition.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # LLC share
    ax = axes[0]
    if results.get('llc'):
        times = sorted(results['llc'].keys())
        diffs = [results['llc'][t]['diff'] for t in times]

        ax.axhline(y=0, color='gray', linestyle='--', linewidth=1)
        ax.axvline(x=-0.5, color='red', linestyle='--', linewidth=1.5)

        colors = ['steelblue' if t < 0 else 'coral' for t in times]
        ax.scatter(times, diffs, c=colors, s=30, zorder=5)
        ax.plot(times, diffs, color='gray', alpha=0.5, linewidth=1)

        ax.set_xlabel('Event Time (months)', fontsize=10)
        ax.set_ylabel('LLC Share: Inside - Outside', fontsize=10)
        ax.set_title('LLC Buyer Share Differential', fontsize=11)

    # Portfolio share
    ax = axes[1]
    if results.get('portfolio'):
        times = sorted(results['portfolio'].keys())
        diffs = [results['portfolio'][t]['diff'] for t in times]

        ax.axhline(y=0, color='gray', linestyle='--', linewidth=1)
        ax.axvline(x=-0.5, color='red', linestyle='--', linewidth=1.5)

        colors = ['steelblue' if t < 0 else 'coral' for t in times]
        ax.scatter(times, diffs, c=colors, s=30, zorder=5)
        ax.plot(times, diffs, color='gray', alpha=0.5, linewidth=1)

        ax.set_xlabel('Event Time (months)', fontsize=10)
        ax.set_ylabel('Portfolio Share: Inside - Outside', fontsize=10)
        ax.set_title('Portfolio Buyer Share Differential', fontsize=11)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Wrote {out_path}')


def summarize_composition_by_group(panel: pd.DataFrame, boundary: str = 'sfha',
                                    caliper: int = 300) -> pd.DataFrame:
    """
    Summary table of buyer composition by inside/outside × pre/post.
    """
    dist_col = f'signed_dist_{boundary}_m'
    inside_col = f'inside_{boundary}'

    d = panel.copy()
    d = d[d['sold_this_month'] == 1]
    d = d[np.isfinite(d[dist_col])]
    d = d[d[dist_col].abs() <= caliper].copy()

    if len(d) == 0:
        return pd.DataFrame()

    d['event_m'] = d['ym'].apply(months_since_event)
    d['post'] = (d['event_m'] >= 0).astype(int)
    d['inside'] = (d[inside_col] == 1).astype(int)
    d['period'] = d['post'].map({0: 'Pre-flood', 1: 'Post-flood'})
    d['location'] = d['inside'].map({0: 'Outside SFHA', 1: 'Inside SFHA'})

    # Compute shares
    summary = d.groupby(['location', 'period']).agg({
        'owner_form_snapshot': lambda x: (x == 'LLC').mean() if 'owner_form_snapshot' in d.columns else np.nan,
        'owner_scale_snapshot': lambda x: (x == 'multi').mean() if 'owner_scale_snapshot' in d.columns else np.nan,
        'sold_this_month': 'sum'
    }).reset_index()

    summary.columns = ['Location', 'Period', 'LLC_Share', 'Portfolio_Share', 'N_Sales']

    return summary


def main():
    """Run buyer composition analysis."""
    ensure_env()

    OUT_DIR.mkdir(exist_ok=True, parents=True)
    FIG_DIR.mkdir(exist_ok=True, parents=True)

    if not PANEL.exists():
        raise SystemExit(f'Missing {PANEL}')

    panel = pd.read_parquet(PANEL)

    print('='*60)
    print('BUYER COMPOSITION ANALYSIS')
    print('='*60)

    # Check available columns
    print('\nAvailable owner columns:')
    owner_cols = [c for c in panel.columns if 'owner' in c.lower()]
    for c in owner_cols:
        print(f'  {c}: {panel[c].dtype}')
        if panel[c].dtype == 'object':
            print(f'    Values: {panel[c].value_counts().head().to_dict()}')

    # 1. DiD for buyer composition
    print('\n\n1. Buyer Composition DiD')
    print('-'*40)

    did_results = buyer_composition_did(panel, 'sfha', 300)
    if not did_results.empty:
        print(did_results.to_string(index=False))
        did_results.to_csv(OUT_DIR / 'buyer_composition_did.csv', index=False)
        print(f'\nWrote {OUT_DIR / "buyer_composition_did.csv"}')

    # 2. Summary table
    print('\n\n2. Composition Summary by Group')
    print('-'*40)

    summary = summarize_composition_by_group(panel, 'sfha', 300)
    if not summary.empty:
        print(summary.to_string(index=False))
        summary.to_csv(OUT_DIR / 'buyer_composition_summary.csv', index=False)
        print(f'\nWrote {OUT_DIR / "buyer_composition_summary.csv"}')

    # 3. Event study
    print('\n\n3. Event Study for Buyer Composition')
    print('-'*40)

    es_results = buyer_composition_event_study(panel, 'sfha', 300)
    if es_results:
        plot_buyer_composition_event_study(es_results,
                                           FIG_DIR / 'fig_buyer_composition_event_study.png')

        # Save event study data
        es_data = []
        for outcome in ['llc', 'portfolio']:
            if outcome in es_results:
                for t, vals in es_results[outcome].items():
                    es_data.append({
                        'outcome': outcome,
                        'event_time': t,
                        **vals
                    })
        if es_data:
            pd.DataFrame(es_data).to_csv(OUT_DIR / 'buyer_composition_event_study.csv', index=False)

    # 4. Interpretation
    print('\n\n' + '='*60)
    print('INTERPRETATION')
    print('='*60)

    if not did_results.empty:
        llc_did = did_results[(did_results['outcome'] == 'LLC_share') &
                              (did_results['statistic'] == 'DiD')]
        port_did = did_results[(did_results['outcome'] == 'portfolio_share') &
                               (did_results['statistic'] == 'DiD')]

        if not llc_did.empty:
            est = llc_did['estimate'].values[0]
            pval = llc_did['pvalue'].values[0]
            print(f'\nLLC Share DiD: {est:.4f} (p={pval:.3f})')
            if est > 0:
                print('  -> LLCs INCREASE share inside SFHA post-flood')
            else:
                print('  -> LLCs DECREASE share inside SFHA post-flood')

        if not port_did.empty:
            est = port_did['estimate'].values[0]
            pval = port_did['pvalue'].values[0]
            print(f'\nPortfolio Share DiD: {est:.4f} (p={pval:.3f})')
            if est > 0:
                print('  -> Portfolio buyers INCREASE share inside SFHA post-flood')
            else:
                print('  -> Portfolio buyers DECREASE share inside SFHA post-flood')


if __name__ == '__main__':
    main()
