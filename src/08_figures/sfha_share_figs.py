#!/usr/bin/env python3
"""
Module: sfha_share_figs.py
Purpose: Generate SFHA sales share time series figures.

This module creates visualizations showing the share of sales occurring
inside the SFHA over time, both county-wide and within the RD caliper window.

Input Files
-----------
- data_work/panel_parcel_month.parquet
- data_work/sfha_sales_monthly.csv (optional)

Output Files
------------
- data_work/tab_sfha_share_county.csv
- data_work/fig_sfha_share_county.png
- data_work/tab_sfha_share_rd.csv
- data_work/fig_sfha_share_rd.png

Functions
---------
ensure_env : Verify virtual environment is activated
main : Generate SFHA share figures
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
COUNTY_MONTHLY = Path('data_work/sfha_sales_monthly.csv')
OUT_DIR = Path('data_work')

def ensure_env():
    if not os.getenv('VIRTUAL_ENV') or not os.getenv('VIRTUAL_ENV').endswith('/.venv'):
        raise SystemExit('Activate .venv first')

def main():
    """
    Generate SFHA sales share time series figures.

    Creates county-wide and RD-caliper SFHA share plots showing the
    proportion of sales occurring inside the SFHA boundary over time.
    """
    ensure_env()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if COUNTY_MONTHLY.exists():
        m = pd.read_csv(COUNTY_MONTHLY)
        if '1' in m.columns and '0' in m.columns:
            m = m.rename(columns={'1':'SFHA','0':'Non_SFHA'})
        if 'SFHA' in m.columns and 'Non_SFHA' in m.columns:
            m['share_sfha'] = m['SFHA'] / (m['SFHA'] + m['Non_SFHA']).replace(0, np.nan)
            m.to_csv(OUT_DIR / 'tab_sfha_share_county.csv', index=False)
            plt.figure(figsize=(10,4))
            plt.plot(range(len(m)), m['share_sfha'])
            plt.title('County-wide SFHA sales share by month')
            plt.ylabel('SFHA share'); plt.xlabel('Month index')
            plt.tight_layout(); plt.savefig(OUT_DIR / 'fig_sfha_share_county.png', dpi=150); plt.close()
    if PANEL.exists():
        pm = pd.read_parquet(PANEL)
        if 'inside_sfha' in pm.columns and 'ym' in pm.columns and 'signed_dist_sfha_m' in pm.columns:
            pm = pm[np.isfinite(pm['signed_dist_sfha_m'])]
            pm = pm[pm['signed_dist_sfha_m'].abs() <= 300]
            agg = pm.groupby(['ym','inside_sfha'])['sold_this_month'].sum().unstack().fillna(0)
            if 1 in agg.columns and 0 in agg.columns:
                share = agg[1] / (agg[1] + agg[0]).replace(0, np.nan)
                out_df = share.rename('share_sfha').reset_index()
                out_df.to_csv(OUT_DIR / 'tab_sfha_share_rd.csv', index=False)
                x = pd.to_datetime(out_df['ym'])
                plt.figure(figsize=(10,4))
                plt.plot(x, out_df['share_sfha'])
                plt.title('RD-caliper SFHA sales share by month (±300m)')
                plt.ylabel('SFHA share'); plt.xlabel('Month')
                plt.tight_layout(); plt.savefig(OUT_DIR / 'fig_sfha_share_rd.png', dpi=150); plt.close()

if __name__ == '__main__':
    main()

