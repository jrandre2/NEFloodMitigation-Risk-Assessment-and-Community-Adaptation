#!/usr/bin/env python3
"""
Module: label_parties.py
Purpose: Classify buyer and seller organizational form using ML-based owner classification.

This module enriches sales transactions with owner metadata from the classification
snapshot, including owner type (Individual, LLC, Corporation, Trust, Other),
portfolio scale (single vs. multi-parcel), and owner-parcel locality indicators.

Input Files
-----------
- data_work/sales_parcel_joined.parquet
- results/integration_run/parcels_with_classification.csv

Output Files
------------
- data_work/sales_parties_labeled.parquet

Functions
---------
ensure_env : Verify virtual environment is activated
main : Execute the party labeling pipeline
"""
from __future__ import annotations
import os, sys
from pathlib import Path
import pandas as pd

SALES = Path('data_work/sales_parcel_joined.parquet')
CLS = Path('results/integration_run/parcels_with_classification.csv')
OUT = Path('data_work/sales_parties_labeled.parquet')

def ensure_env():
    if not os.getenv('VIRTUAL_ENV') or not os.getenv('VIRTUAL_ENV').endswith('/.venv'):
        print('ERROR: Please activate project .venv (source .venv/bin/activate) before running.', file=sys.stderr)
        sys.exit(1)

def main():
    """
    Execute party labeling pipeline.

    Reads sales and classification data, extracts owner metadata (type, portfolio
    scale, ZIP codes, locality), merges onto sales transactions, and writes to
    parquet format. Uses snapshot-based buyer derivation as proxy.

    Raises
    ------
    SystemExit
        If required input files do not exist.
    """
    ensure_env()
    if not SALES.exists():
        raise SystemExit(f'Missing {SALES}; run 01_link first')
    if not CLS.exists():
        raise SystemExit(f'Missing classification CSV at {CLS}')
    s = pd.read_parquet(SALES)
    # Minimal owner metadata from classification snapshot (proxy for buyer)
    head = pd.read_csv(CLS, nrows=0).columns.tolist()
    use = [c for c in ['Parcel_ID','predicted_owner_type','Current_Ow','OwnerZIP5','ParcelZIP5','Zip','Ph_Zip5'] if c in head]
    meta = pd.read_csv(CLS, usecols=use)
    meta['Parcel_ID'] = meta['Parcel_ID'].astype(str)
    # Owner key & type
    if 'Current_Ow' in meta.columns:
        meta['owner_key'] = meta['Current_Ow'].astype(str).str.strip()
    meta.rename(columns={'predicted_owner_type':'owner_form_snapshot'}, inplace=True)
    # Compute portfolio size on snapshot
    if 'owner_key' in meta.columns:
        counts = meta.groupby('owner_key').size().rename('owner_parcel_count')
        meta = meta.merge(counts, on='owner_key', how='left')
        meta['owner_scale_snapshot'] = meta['owner_parcel_count'].apply(lambda x: 'multi' if pd.notna(x) and x>1 else 'single')
    # Approx buyer ZIPs and locality
    meta['owner_zip'] = meta['OwnerZIP5'] if 'OwnerZIP5' in meta.columns else meta.get('Ph_Zip5')
    meta['situs_zip'] = meta['ParcelZIP5'] if 'ParcelZIP5' in meta.columns else meta.get('Zip')
    for c in ['owner_zip','situs_zip']:
        if c in meta.columns:
            meta[c] = meta[c].astype(str).str.extract(r'(\d{5})', expand=False)
    if 'owner_zip' in meta.columns and 'situs_zip' in meta.columns:
        meta['LocalOwner'] = (meta['owner_zip'] == meta['situs_zip']).astype(int)
    # Merge
    keep_cols = ['Parcel_ID']
    for c in ['owner_key','owner_form_snapshot','owner_scale_snapshot','owner_zip','situs_zip','LocalOwner']:
        if c in meta.columns:
            keep_cols.append(c)
    out = s.merge(meta[keep_cols], left_on='parcel_id', right_on='Parcel_ID', how='left')
    out.drop(columns=['Parcel_ID'], inplace=True, errors='ignore')
    # Flag that this is snapshot-based, not true buyer extraction
    out['buyer_from_snapshot'] = 1
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT, index=False)
    print('Wrote', OUT, f'({len(out):,} rows)')

if __name__ == '__main__':
    main()

