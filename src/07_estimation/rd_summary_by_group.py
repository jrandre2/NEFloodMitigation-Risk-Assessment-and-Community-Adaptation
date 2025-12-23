#!/usr/bin/env python3
"""
Module: rd_summary_by_group.py
Purpose: Generate RD summary statistics stratified by owner type and scale.

This module estimates difference-in-differences effects at SFHA and inundation
boundaries separately for each owner classification group (Individual, LLC,
Corporation, etc.) and portfolio scale (single vs. multi-parcel).

Input Files
-----------
- data_work/panel_parcel_month.parquet

Output Files
------------
- data_work/tab_rd_summary_by_group.csv
- data_work/tab_rd_price_summary_by_group.csv
- data_work/tab_rd_summary_by_owner_form.csv
- data_work/tab_rd_summary_by_owner_scale.csv

Functions
---------
ensure_env : Verify virtual environment is activated
months_since_event : Convert year-month to event time
did_sale_rate : Compute DiD for sale rates by group
did_log_price : Compute DiD for log prices by group
main : Execute group-level RD estimation
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
    y, m = ym.split('-')
    y = int(y); m = int(m)
    return (y - 2019) * 12 + (m - 3)

def did_sale_rate(panel: pd.DataFrame, boundary: str, caliper: int, group_col: str) -> pd.DataFrame:
    dist_col = f'signed_dist_{boundary}_m'
    inside_col = f'inside_{boundary}'
    if dist_col not in panel.columns or inside_col not in panel.columns:
        return pd.DataFrame()
    d = panel.copy()
    d = d[np.isfinite(d[dist_col])]
    d = d[d[dist_col].abs() <= caliper].copy()
    d = d[~d[group_col].isna()].copy()
    d['event_m'] = d['ym'].apply(months_since_event)
    d['post'] = (d['event_m'] >= 0).astype(int)
    d['inside'] = (d[inside_col] == 1).astype(int)
    outs = []
    for g, sub in d.groupby(group_col):
        if len(sub) < 50:
            continue
        X = pd.DataFrame({
            'const': 1.0,
            'inside': sub['inside'].astype(float),
            'post': sub['post'].astype(float),
        })
        X['inside_x_post'] = X['inside'] * X['post']
        y = sub['sold_this_month'].astype(float)
        try:
            model = sm.OLS(y, X).fit(cov_type='HC3')
            b = model.params.get('inside_x_post', np.nan)
            se = model.bse.get('inside_x_post', np.nan)
            lo, hi = (b - 1.96*se, b + 1.96*se) if np.isfinite(se) else (np.nan, np.nan)
        except Exception:
            b, lo, hi = (np.nan, np.nan, np.nan)
        grp = sub.groupby(['inside','post'])['sold_this_month'].mean().unstack()
        inside_pre = float(grp.loc[1,0]) if (1 in grp.index and 0 in grp.columns) else np.nan
        inside_post = float(grp.loc[1,1]) if (1 in grp.index and 1 in grp.columns) else np.nan
        outside_pre = float(grp.loc[0,0]) if (0 in grp.index and 0 in grp.columns) else np.nan
        outside_post = float(grp.loc[0,1]) if (0 in grp.index and 1 in grp.columns) else np.nan
        outs.append({
            'group_col': group_col,
            'group': g,
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
            'n_obs': len(sub)
        })
    return pd.DataFrame(outs)

def did_log_price(panel: pd.DataFrame, boundary: str, caliper: int, group_col: str) -> pd.DataFrame:
    dist_col = f'signed_dist_{boundary}_m'
    inside_col = f'inside_{boundary}'
    if dist_col not in panel.columns or inside_col not in panel.columns:
        return pd.DataFrame()
    d = panel.copy()
    d = d[np.isfinite(d[dist_col])]
    d = d[d[dist_col].abs() <= caliper]
    d = d[d['sold_this_month'] == 1].copy()  # months with sale only
    d = d[~d[group_col].isna()].copy()
    d['event_m'] = d['ym'].apply(months_since_event)
    d['post'] = (d['event_m'] >= 0).astype(int)
    d['inside'] = (d[inside_col] == 1).astype(int)
    outs = []
    for g, sub in d.groupby(group_col):
        if len(sub) < 20:
            continue
        X = pd.DataFrame({'const': 1.0, 'inside': sub['inside'].astype(float), 'post': sub['post'].astype(float)})
        X['inside_x_post'] = X['inside'] * X['post']
        y = sub['log_price'].astype(float)
        try:
            model = sm.OLS(y, X).fit(cov_type='HC3')
            b = model.params.get('inside_x_post', np.nan)
            se = model.bse.get('inside_x_post', np.nan)
            lo, hi = (b - 1.96*se, b + 1.96*se) if np.isfinite(se) else (np.nan, np.nan)
        except Exception:
            b, lo, hi = (np.nan, np.nan, np.nan)
        grp = sub.groupby(['inside','post'])['log_price'].median().unstack()
        inside_pre = float(grp.loc[1,0]) if (1 in grp.index and 0 in grp.columns) else np.nan
        inside_post = float(grp.loc[1,1]) if (1 in grp.index and 1 in grp.columns) else np.nan
        outside_pre = float(grp.loc[0,0]) if (0 in grp.index and 0 in grp.columns) else np.nan
        outside_post = float(grp.loc[0,1]) if (0 in grp.index and 1 in grp.columns) else np.nan
        outs.append({
            'group_col': group_col,
            'group': g,
            'boundary': boundary,
            'caliper_m': caliper,
            'inside_pre_med_log_price': inside_pre,
            'inside_post_med_log_price': inside_post,
            'outside_pre_med_log_price': outside_pre,
            'outside_post_med_log_price': outside_post,
            'DiD_estimate_log_price': b,
            'DiD_ci_lo': lo,
            'DiD_ci_hi': hi,
            'n_obs': len(sub)
        })
    return pd.DataFrame(outs)

def main():
    """
    Execute group-level RD estimation.

    Runs DiD estimation for both boundaries and calipers, stratified by
    owner form and scale columns. Writes consolidated and split CSV files.

    Raises
    ------
    SystemExit
        If panel file does not exist.
    """
    ensure_env()
    if not PANEL.exists():
        raise SystemExit(f'Missing {PANEL}')
    panel = pd.read_parquet(PANEL)
    group_cols = []
    for gc in ['owner_form_snapshot','owner_scale_snapshot']:
        if gc in panel.columns:
            group_cols.append(gc)
    if not group_cols:
        print('No group columns available on panel; skip.')
        return
    all_sr, all_pr = [], []
    for boundary in ['inund','sfha']:
        for cal in [150, 300]:
            for gc in group_cols:
                df_sr = did_sale_rate(panel, boundary, cal, gc)
                if not df_sr.empty:
                    all_sr.append(df_sr)
                df_pr = did_log_price(panel, boundary, cal, gc)
                if not df_pr.empty:
                    all_pr.append(df_pr)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if all_sr:
        sr = pd.concat(all_sr, ignore_index=True)
        path = OUT_DIR / 'tab_rd_summary_by_group.csv'
        sr.to_csv(path, index=False)
        # split convenience exports
        for gc in sorted(sr['group_col'].unique()):
            sub = sr[sr['group_col'] == gc]
            sub.to_csv(OUT_DIR / f'tab_rd_summary_by_{gc.replace("_snapshot","")}.csv', index=False)
        print('Wrote', path)
    if all_pr:
        pr = pd.concat(all_pr, ignore_index=True)
        pathp = OUT_DIR / 'tab_rd_price_summary_by_group.csv'
        pr.to_csv(pathp, index=False)
        for gc in sorted(pr['group_col'].unique()):
            sub = pr[pr['group_col'] == gc]
            sub.to_csv(OUT_DIR / f'tab_rd_price_summary_by_{gc.replace("_snapshot","")}.csv', index=False)
        print('Wrote', pathp)

if __name__ == '__main__':
    main()

