#!/usr/bin/env python3
"""
Module: rd_summary.py
Purpose: Generate boundary RD summary statistics and DiD estimates.

This module estimates difference-in-differences effects at SFHA and
inundation boundaries for both sale rates and log prices, using linear
probability models with HC3 robust standard errors.

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
"""
from __future__ import annotations
import os, sys
from pathlib import Path
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

def main():
    """
    Execute RD summary estimation for sale rates and prices.

    Runs DiD estimation for both inundation and SFHA boundaries at
    150m and 300m calipers, writes summary CSV files.

    Raises
    ------
    SystemExit
        If panel file does not exist.
    """
    ensure_env()
    if not PANEL.exists():
        raise SystemExit(f'Missing {PANEL}')
    panel = pd.read_parquet(PANEL)
    outs = []
    for boundary in ['inund','sfha']:
        for cal in [150, 300]:
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
    for boundary in ['inund','sfha']:
        for cal in [150, 300]:
            dfp = summarize_price(panel, boundary, cal)
            if not dfp.empty:
                pouts.append(dfp)
    if pouts:
        pout = pd.concat(pouts, ignore_index=True)
        ppath = OUT_DIR / 'tab_rd_price_summary.csv'
        pout.to_csv(ppath, index=False)
        print('Wrote', ppath)

if __name__ == '__main__':
    main()
