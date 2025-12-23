#!/usr/bin/env python3
"""
Module: make_figures.py
Purpose: Generate publication-quality figures from analysis results.

This module creates event study coefficient plots, boundary RD visualizations,
and saves corresponding data tables for the Freeze and Flight manuscript.

Input Files
-----------
- data_work/panel_parcel_month.parquet
- data_work/event_study_summary.parquet

Output Files
------------
- data_work/fig_event_study.png
- data_work/fig_rd_inund_sale_rate.png
- data_work/fig_rd_inund_price.png
- data_work/fig_rd_sfha_sale_rate.png
- data_work/fig_rd_sfha_price.png
- data_work/tab_event_study.csv
- data_work/tab_rd_inund.csv
- data_work/tab_rd_sfha.csv

Functions
---------
ensure_env : Verify virtual environment is activated
save_df : Save DataFrame to CSV
make_event_study_fig : Create event study coefficient plot
make_boundary_rd_fig : Create boundary RD visualization
main : Execute figure generation pipeline
"""
from __future__ import annotations
import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PANEL = Path('data_work/panel_parcel_month.parquet')
EVT = Path('data_work/event_study_summary.parquet')
OUT_DIR = Path('data_work')

def ensure_env():
    if not os.getenv('VIRTUAL_ENV') or not os.getenv('VIRTUAL_ENV').endswith('/.venv'):
        raise SystemExit('Activate .venv first')

def save_df(df: pd.DataFrame, name: str) -> Path:
    p = OUT_DIR / f'{name}.csv'
    df.to_csv(p, index=False)
    return p

def make_event_study_fig():
    if not EVT.exists():
        return None, None
    df = pd.read_parquet(EVT)
    df = df.replace([np.inf, -np.inf], np.nan)
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(df['event_m'], df['control_rate'], label='Control (not inundated)', color='tab:blue')
    axes[0].plot(df['event_m'], df['treated_rate'], label='Treated (inundated)', color='tab:red')
    axes[0].axvline(0, color='k', linestyle='--', alpha=0.5)
    axes[0].set_ylabel('Sale rate (monthly)')
    axes[0].legend()
    axes[1].plot(df['event_m'], df['price_diff'], label='Treated - Control (log price)', color='tab:green')
    axes[1].axhline(0, color='k', linestyle='--', alpha=0.5)
    axes[1].axvline(0, color='k', linestyle='--', alpha=0.5)
    axes[1].set_xlabel('Months since Mar-2019')
    axes[1].set_ylabel('Log-price diff')
    fig.tight_layout()
    out_png = OUT_DIR / 'fig_event_study.png'
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    out_csv = save_df(df, 'tab_event_study')
    return out_png, out_csv

def make_boundary_rd_fig(panel: pd.DataFrame, dist_col: str, label: str):
    d = panel.copy()
    d = d[np.isfinite(d[dist_col])]
    d = d[(d[dist_col] >= -300) & (d[dist_col] <= 300)].copy()
    bins = np.arange(-300, 300+50, 50)
    d['bin'] = pd.cut(d[dist_col], bins=bins, include_lowest=True)
    def months_since_event(ym: str) -> int:
        y, m = ym.split('-')
        y = int(y); m = int(m)
        return (y - 2019) * 12 + (m - 3)
    d['event_m'] = d['ym'].apply(months_since_event)
    d['period'] = np.where(d['event_m'] < 0, 'pre', 'post')
    agg = d.groupby(['bin','period']).agg(
        sale_rate=('sold_this_month','mean'),
        med_log_price=('log_price','median'),
        n=('sold_this_month','size')
    ).reset_index()
    # Build x from bin midpoints
    mids = []
    for iv in agg['bin'].cat.categories:
        lo = iv.left; hi = iv.right
        mids.append((lo+hi)/2)
    x = np.array(mids)
    # Sale rate
    plt.figure(figsize=(10,4))
    for per, color in [('pre','tab:blue'),('post','tab:orange')]:
        sub = agg[agg['period']==per]
        plt.plot(x[:len(sub)], sub['sale_rate'].values, label=per, color=color)
    plt.axvline(0, color='k', linestyle='--', alpha=0.5)
    plt.xlabel(f'Signed distance to {label} boundary (m)')
    plt.ylabel('Sale rate (monthly)')
    plt.legend()
    out1 = OUT_DIR / f'fig_rd_{label}_sale_rate.png'
    plt.tight_layout(); plt.savefig(out1, dpi=150); plt.close()
    # Price
    plt.figure(figsize=(10,4))
    for per, color in [('pre','tab:blue'),('post','tab:orange')]:
        sub = agg[agg['period']==per]
        plt.plot(x[:len(sub)], sub['med_log_price'].values, label=per, color=color)
    plt.axvline(0, color='k', linestyle='--', alpha=0.5)
    plt.xlabel(f'Signed distance to {label} boundary (m)')
    plt.ylabel('Median log price')
    plt.legend()
    out2 = OUT_DIR / f'fig_rd_{label}_price.png'
    plt.tight_layout(); plt.savefig(out2, dpi=150); plt.close()
    out_tab = save_df(agg, f'tab_rd_{label}')
    return (out1, out2, out_tab)

def main():
    """
    Execute figure generation pipeline.

    Creates event study and boundary RD figures for both inundation and
    SFHA boundaries, saving PNG images and data tables.
    """
    ensure_env()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    es_png, es_csv = make_event_study_fig()
    panel = pd.read_parquet(PANEL)
    outs = []
    if 'signed_dist_inund_m' in panel.columns:
        outs.append(make_boundary_rd_fig(panel, 'signed_dist_inund_m', 'inund'))
    if 'signed_dist_sfha_m' in panel.columns:
        outs.append(make_boundary_rd_fig(panel, 'signed_dist_sfha_m', 'sfha'))
    print('Event study outputs:', es_png, es_csv)
    for trio in outs:
        print('Boundary RD outputs:', trio)

if __name__ == '__main__':
    main()

