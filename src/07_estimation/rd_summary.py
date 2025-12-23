#!/usr/bin/env python3
"""
Module: rd_summary.py
Purpose: Generate boundary RD summary statistics and DiD estimates.

This module estimates difference-in-differences effects at SFHA and
inundation boundaries for both sale rates and log prices, using linear
probability models with HC3 robust standard errors.

IMPORTANT: The inundation boundary is the PRIMARY specification because
it passes pre-trends tests (F=1.50, p=0.34). The SFHA boundary fails
pre-trends (F=3.33, p<0.001) and should be used for robustness only.

Input Files
-----------
- data_work/panel_parcel_month.parquet

Output Files
------------
- data_work/tab_rd_summary.csv
- data_work/tab_rd_price_summary.csv

Functions
---------
ensure_env : Verify virtual environment is activated
months_since_event : Convert year-month to event time
summarize_boundary : Compute DiD for sale rates at boundary
summarize_price : Compute DiD for log prices at boundary
main : Execute RD summary estimation

Notes
-----
Default boundary is 'inund' (inundation). Use --boundary=sfha for robustness.
Spatial standard errors via Conley (1999) available with --spatial-se flag.
"""
from __future__ import annotations
import argparse
import os, sys
from pathlib import Path
from typing import Optional, List
import pandas as pd
import numpy as np
import statsmodels.api as sm

PANEL = Path('data_work/panel_parcel_month.parquet')
OUT_DIR = Path('data_work')

def ensure_env():
    if not os.getenv('VIRTUAL_ENV') or not os.getenv('VIRTUAL_ENV').endswith('/.venv'):
        print('ERROR: Please activate project .venv (source .venv/bin/activate) before running.', file=sys.stderr)
        sys.exit(1)

def months_since_event(ym: str) -> int:
    """
    Convert year-month string to event time relative to March 2019.

    Parameters
    ----------
    ym : str
        Year-month in 'YYYY-MM' format.

    Returns
    -------
    int
        Months since March 2019 (negative for pre-event).
    """
    y, m = ym.split('-')
    y = int(y); m = int(m)
    return (y - 2019) * 12 + (m - 3)

def summarize_boundary(panel: pd.DataFrame, boundary: str, caliper: int) -> pd.DataFrame:
    """
    Compute DiD estimate for sale rates at boundary.

    Parameters
    ----------
    panel : pd.DataFrame
        Panel with signed distance and inside indicator columns.
    boundary : str
        Boundary type: 'inund' or 'sfha'.
    caliper : int
        Caliper window in meters (e.g., 150, 300).

    Returns
    -------
    pd.DataFrame
        Single-row DataFrame with DiD estimate and CI.
    """
    # boundary in {'inund','sfha'}
    dist_col = f'signed_dist_{boundary}_m'
    inside_col = f'inside_{boundary}'
    if dist_col not in panel.columns or inside_col not in panel.columns:
        return pd.DataFrame()
    d = panel.copy()
    d = d[np.isfinite(d[dist_col])]
    d = d[d[dist_col].abs() <= caliper].copy()
    # Post indicator
    d['event_m'] = d['ym'].apply(months_since_event)
    d['post'] = (d['event_m'] >= 0).astype(int)
    d['inside'] = (d[inside_col] == 1).astype(int)
    # Group means
    grp = d.groupby(['inside','post'])['sold_this_month'].mean().unstack()
    inside_pre = float(grp.loc[1,0]) if (1 in grp.index and 0 in grp.columns) else np.nan
    inside_post = float(grp.loc[1,1]) if (1 in grp.index and 1 in grp.columns) else np.nan
    outside_pre = float(grp.loc[0,0]) if (0 in grp.index and 0 in grp.columns) else np.nan
    outside_post = float(grp.loc[0,1]) if (0 in grp.index and 1 in grp.columns) else np.nan
    # DiD via linear probability model with HC3 robust SE
    X = pd.DataFrame({
        'const': 1.0,
        'inside': d['inside'].astype(float),
        'post': d['post'].astype(float),
        'inside_x_post': (d['inside'] * d['post']).astype(float)
    })
    y = d['sold_this_month'].astype(float)
    model = sm.OLS(y, X).fit(cov_type='HC3')
    b = model.params.get('inside_x_post', np.nan)
    se = model.bse.get('inside_x_post', np.nan)
    lo, hi = (b - 1.96*se, b + 1.96*se) if np.isfinite(se) else (np.nan, np.nan)
    out = pd.DataFrame([{ 
        'boundary': boundary,
        'caliper_m': caliper,
        'inside_pre_rate': inside_pre,
        'inside_post_rate': inside_post,
        'outside_pre_rate': outside_pre,
        'outside_post_rate': outside_post,
        'delta_inside': inside_post - inside_pre if np.isfinite(inside_pre) and np.isfinite(inside_post) else np.nan,
        'delta_outside': outside_post - outside_pre if np.isfinite(outside_pre) and np.isfinite(outside_post) else np.nan,
        'DiD_estimate': b,
        'DiD_ci_lo': lo,
        'DiD_ci_hi': hi,
        'n_obs': len(d)
    }])
    return out

def summarize_price(panel: pd.DataFrame, boundary: str, caliper: int) -> pd.DataFrame:
    """
    Compute DiD estimate for log prices at boundary.

    Parameters
    ----------
    panel : pd.DataFrame
        Panel with signed distance and inside indicator columns.
    boundary : str
        Boundary type: 'inund' or 'sfha'.
    caliper : int
        Caliper window in meters (e.g., 150, 300).

    Returns
    -------
    pd.DataFrame
        Single-row DataFrame with DiD estimate and CI.
    """
    dist_col = f'signed_dist_{boundary}_m'
    inside_col = f'inside_{boundary}'
    if dist_col not in panel.columns or inside_col not in panel.columns:
        return pd.DataFrame()
    d = panel.copy()
    d = d[np.isfinite(d[dist_col])]
    d = d[d[dist_col].abs() <= caliper]
    # Only months with a sale (log_price defined)
    d = d[d['sold_this_month'] == 1].copy()
    d['event_m'] = d['ym'].apply(months_since_event)
    d['post'] = (d['event_m'] >= 0).astype(int)
    d['inside'] = (d[inside_col] == 1).astype(int)
    # OLS: log_price ~ inside + post + inside:post
    X = pd.DataFrame({'const': 1.0, 'inside': d['inside'].astype(float), 'post': d['post'].astype(float)})
    X['inside_x_post'] = X['inside'] * X['post']
    y = d['log_price'].astype(float)
    model = sm.OLS(y, X).fit(cov_type='HC3')
    b = model.params.get('inside_x_post', np.nan)
    se = model.bse.get('inside_x_post', np.nan)
    lo, hi = (b - 1.96*se, b + 1.96*se) if np.isfinite(se) else (np.nan, np.nan)
    # Group medians for reference
    grp = d.groupby(['inside','post'])['log_price'].median().unstack()
    inside_pre = float(grp.loc[1,0]) if (1 in grp.index and 0 in grp.columns) else np.nan
    inside_post = float(grp.loc[1,1]) if (1 in grp.index and 1 in grp.columns) else np.nan
    outside_pre = float(grp.loc[0,0]) if (0 in grp.index and 0 in grp.columns) else np.nan
    outside_post = float(grp.loc[0,1]) if (0 in grp.index and 1 in grp.columns) else np.nan
    out = pd.DataFrame([{ 
        'boundary': boundary,
        'caliper_m': caliper,
        'inside_pre_med_log_price': inside_pre,
        'inside_post_med_log_price': inside_post,
        'outside_pre_med_log_price': outside_pre,
        'outside_post_med_log_price': outside_post,
        'DiD_estimate_log_price': b,
        'DiD_ci_lo': lo,
        'DiD_ci_hi': hi,
        'n_obs': len(d)
    }])
    return out

def main(
    boundaries: Optional[List[str]] = None,
    calipers: Optional[List[int]] = None,
    spatial_se: bool = False,
    conley_cutoff_km: float = 5.0
):
    """
    Execute RD summary estimation for sale rates and prices.

    Runs DiD estimation at specified boundaries and calipers,
    writes summary CSV files.

    Parameters
    ----------
    boundaries : list of str, optional
        Boundaries to test. Default is ['inund'] (primary spec).
        Use ['inund', 'sfha'] for full analysis with SFHA as robustness.
    calipers : list of int, optional
        Caliper windows in meters. Default is [150, 300].
    spatial_se : bool, optional
        If True, compute Conley spatial standard errors.
        Requires spatial_econometrics module. Default is False.
    conley_cutoff_km : float, optional
        Distance cutoff for Conley SEs in km. Default is 5.0.

    Raises
    ------
    SystemExit
        If panel file does not exist.
    """
    ensure_env()
    if not PANEL.exists():
        raise SystemExit(f'Missing {PANEL}')

    # Defaults: inundation is primary (passes pre-trends)
    if boundaries is None:
        boundaries = ['inund']  # Default to inund only
    if calipers is None:
        calipers = [150, 300]

    panel = pd.read_parquet(PANEL)

    # Sale rate DiD summaries
    outs = []
    for boundary in boundaries:
        for cal in calipers:
            df = summarize_boundary(panel, boundary, cal)
            if not df.empty:
                outs.append(df)
    if not outs:
        print('No RD summaries created')
    else:
        out = pd.concat(outs, ignore_index=True)
        out_path = OUT_DIR / 'tab_rd_summary.csv'
        out.to_csv(out_path, index=False)
        print('Wrote', out_path)

    # Price DiD summaries
    pouts = []
    for boundary in boundaries:
        for cal in calipers:
            dfp = summarize_price(panel, boundary, cal)
            if not dfp.empty:
                pouts.append(dfp)
    if pouts:
        pout = pd.concat(pouts, ignore_index=True)
        ppath = OUT_DIR / 'tab_rd_price_summary.csv'
        pout.to_csv(ppath, index=False)
        print('Wrote', ppath)

    # Spatial standard errors (if requested)
    if spatial_se:
        print(f'\nComputing Conley spatial SEs with {conley_cutoff_km}km cutoff...')
        try:
            from spatial_econometrics import conley_standard_errors
            for boundary in boundaries:
                for cal in calipers:
                    result = conley_standard_errors(panel, boundary, cal, conley_cutoff_km)
                    print(f'  {boundary} {cal}m: Conley SE = {result.get("se", "N/A"):.6f}')
        except ImportError:
            print('  WARNING: spatial_econometrics module not found.')
            print('  Run: python src/07_estimation/spatial_econometrics.py first.')


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='RD Summary: Boundary DiD estimation for sale rates and prices.'
    )
    parser.add_argument(
        '--boundary', '-b',
        nargs='+',
        choices=['inund', 'sfha'],
        default=['inund'],
        help='Boundaries to test. Default: inund (primary spec). '
             'Use "inund sfha" for full analysis.'
    )
    parser.add_argument(
        '--caliper', '-c',
        nargs='+',
        type=int,
        default=[150, 300],
        help='Caliper windows in meters. Default: 150 300'
    )
    parser.add_argument(
        '--spatial-se',
        action='store_true',
        help='Compute Conley spatial standard errors'
    )
    parser.add_argument(
        '--conley-cutoff',
        type=float,
        default=5.0,
        help='Distance cutoff for Conley SEs in km. Default: 5.0'
    )
    parser.add_argument(
        '--all-boundaries',
        action='store_true',
        help='Run both inund and sfha boundaries'
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    boundaries = args.boundary
    if args.all_boundaries:
        boundaries = ['inund', 'sfha']

    main(
        boundaries=boundaries,
        calipers=args.caliper,
        spatial_se=args.spatial_se,
        conley_cutoff_km=args.conley_cutoff
    )
