#!/usr/bin/env python3
"""
Module: build_panels.py
Purpose: Build parcel-month panel for event study and RD analysis.

This module creates the main analysis panel by expanding parcels across
the event window (±24 months from March 2019), merging treatments and
sales, and computing sold indicators and log prices per parcel-month.

Input Files
-----------
- data_work/parcels_sfr.gpkg
- data_work/parcel_treatments_rd.parquet (or parcel_treatments_with_rings.parquet)
- data_work/sales_buyer_features.parquet
- results/integration_run/parcels_with_classification.csv (optional)

Output Files
------------
- data_work/panel_parcel_month.parquet
- data_work/panel_micro_counts.parquet

Functions
---------
ensure_env : Verify virtual environment is activated
expand_months : Create parcel × month grid
main : Execute panel construction pipeline
"""
from __future__ import annotations
import os, sys
from pathlib import Path
import pandas as pd
import numpy as np
import geopandas as gpd

PARCELS = Path('data_work/parcels_sfr.gpkg')
TREAT_BASE = Path('data_work/parcel_treatments_with_rings.parquet')
TREAT_RD = Path('data_work/parcel_treatments_rd.parquet')
SALES = Path('data_work/sales_buyer_features.parquet')
OUT1 = Path('data_work/panel_parcel_month.parquet')
OUT2 = Path('data_work/panel_micro_counts.parquet')
CLS = Path('results/integration_run/parcels_with_classification.csv')

def ensure_env():
    if not os.getenv('VIRTUAL_ENV') or not os.getenv('VIRTUAL_ENV').endswith('/.venv'):
        print('ERROR: Please activate project .venv (source .venv/bin/activate) before running.', file=sys.stderr)
        sys.exit(1)

def expand_months(parcels: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """
    Create parcel × month grid for panel construction.

    Parameters
    ----------
    parcels : pd.DataFrame
        DataFrame with 'parcel_id' column.
    start : str
        Start month (YYYY-MM format).
    end : str
        End month (YYYY-MM format).

    Returns
    -------
    pd.DataFrame
        Grid with parcel_id and ym columns.
    """
    months = pd.period_range(start, end, freq='M').astype(str)
    idx = pd.MultiIndex.from_product([parcels['parcel_id'], months], names=['parcel_id','ym'])
    return idx.to_frame(index=False)

def main():
    """
    Execute panel construction pipeline.

    Builds parcel-month panel for event window [-24, +24] months around
    March 2019, merges treatment indicators and sales data, computes
    sold indicators and log prices, optionally attaches owner metadata.

    Raises
    ------
    SystemExit
        If required input files do not exist.
    """
    ensure_env()
    if not PARCELS.exists():
        raise SystemExit(f'Missing {PARCELS}')
    treat_path = TREAT_RD if TREAT_RD.exists() else TREAT_BASE
    if not treat_path.exists():
        raise SystemExit(f'Missing treatments parquet')
    if not SALES.exists():
        raise SystemExit(f'Missing {SALES}')
    p = gpd.read_file(PARCELS)
    t = pd.read_parquet(treat_path)
    s = pd.read_parquet(SALES)
    p['parcel_id'] = p['parcel_id'].astype(str)
    t['parcel_id'] = t['parcel_id'].astype(str)
    s['parcel_id'] = s['parcel_id'].astype(str)
    s['ym'] = pd.to_datetime(s['sale_date']).dt.to_period('M').astype(str)
    # Optional: attach snapshot owner form/scale at parcel level from classification CSV
    owner_map = None
    if CLS.exists():
        try:
            head = pd.read_csv(CLS, nrows=0).columns.tolist()
            use = [c for c in ['Parcel_ID','predicted_owner_type','Current_Ow','owner_name'] if c in head]
            meta = pd.read_csv(CLS, usecols=use)
            meta['Parcel_ID'] = meta['Parcel_ID'].astype(str)
            meta.rename(columns={'predicted_owner_type':'owner_form_snapshot'}, inplace=True)
            if 'Current_Ow' in meta.columns:
                meta['owner_key'] = meta['Current_Ow'].astype(str).str.strip()
            elif 'owner_name' in meta.columns:
                meta['owner_key'] = meta['owner_name'].astype(str).str.strip()
            counts = meta.groupby('owner_key').size().rename('owner_parcel_count')
            meta = meta.merge(counts, on='owner_key', how='left')
            meta['owner_scale_snapshot'] = meta['owner_parcel_count'].apply(lambda x: 'multi' if pd.notna(x) and x>1 else 'single')
            owner_map = meta[['Parcel_ID','owner_form_snapshot','owner_scale_snapshot']].drop_duplicates()
            owner_map.rename(columns={'Parcel_ID':'parcel_id'}, inplace=True)
        except Exception as e:
            owner_map = None
    # Event window: -24 to +24 months around 2019-03
    start = (pd.Period('2019-03', freq='M') - 24).strftime('%Y-%m')
    end = (pd.Period('2019-03', freq='M') + 24).strftime('%Y-%m')
    # Optionally restrict parcels to RD windows to keep panel tractable
    restrict_rd = os.getenv('PANEL_SCOPE', 'rd_only') == 'rd_only'
    pids = p[['parcel_id']].copy()
    if restrict_rd:
        rd_mask_cols = [c for c in t.columns if c.startswith('rd_')]
        mask = False
        if rd_mask_cols:
            mask = t[rd_mask_cols].any(axis=1)
        # also include inside flags if present
        for c in ['inside_inund','inside_sfha']:
            if c in t.columns:
                mask = mask | (t[c]==1)
        keep_ids = set(t.loc[mask, 'parcel_id'].astype(str))
        pids = pids[pids['parcel_id'].isin(keep_ids)]
    base = expand_months(pids, start, end)
    pm = base.merge(t, on='parcel_id', how='left')
    # Sold indicator and median log price per parcel×month
    sold_m = s.groupby(['parcel_id','ym']).agg(n_sales=('sale_id','size'),
                                              med_price=('price_real', 'median')).reset_index()
    sold_m['log_price'] = np.log(sold_m['med_price'])
    pm = pm.merge(sold_m[['parcel_id','ym','n_sales','log_price']], on=['parcel_id','ym'], how='left')
    pm['sold_this_month'] = (pm['n_sales'].fillna(0) > 0).astype(int)
    pm = pm.merge(p[['parcel_id','neighborhood_id']], on='parcel_id', how='left')
    if owner_map is not None:
        pm = pm.merge(owner_map, on='parcel_id', how='left')
    OUT1.parent.mkdir(parents=True, exist_ok=True)
    pm.to_parquet(OUT1, index=False)
    mc = pm.groupby(['neighborhood_id','ym']).agg(sales_count=('sold_this_month','sum')).reset_index()
    mc.to_parquet(OUT2, index=False)
    print('Wrote', OUT1, 'and', OUT2)

if __name__ == '__main__':
    main()
