#!/usr/bin/env python3
"""
Module: buyer_proximity.py
Purpose: Calculate buyer-parcel proximity features for ownership analysis.

This module enriches sales data with buyer proximity indicators, including
local owner flags and ZIP-based distance bands, using snapshot metadata
from the classification file.

Input Files
-----------
- data_work/sales_clean.parquet

Output Files
------------
- data_work/sales_buyer_features.parquet

Functions
---------
ensure_env : Verify virtual environment is activated
main : Execute buyer feature creation pipeline
"""
from __future__ import annotations
import os, sys
from pathlib import Path
import pandas as pd

RAW = Path('data_work/sales_clean.parquet')
OUT = Path('data_work/sales_buyer_features.parquet')

def ensure_env():
    if not os.getenv('VIRTUAL_ENV') or not os.getenv('VIRTUAL_ENV').endswith('/.venv'):
        print('ERROR: Please activate project .venv (source .venv/bin/activate) before running.', file=sys.stderr)
        sys.exit(1)

def main():
    """
    Execute buyer feature creation pipeline.

    Reads cleaned sales, extracts local owner indicator from snapshot,
    creates ZIP-based proximity bands, and writes enriched sales to
    parquet format.

    Raises
    ------
    SystemExit
        If input file does not exist.
    """
    ensure_env()
    if not RAW.exists():
        raise SystemExit(f'Missing {RAW}; run clean_sales first')
    df = pd.read_parquet(RAW)
    # Proximity proxy: LocalOwner from snapshot metadata
    if 'LocalOwner' in df.columns:
        df['buyer_local'] = df['LocalOwner']
    else:
        df['buyer_local'] = None
    # Bands placeholder (same ZIP vs other) if zip fields available
    if 'owner_zip' in df.columns and 'situs_zip' in df.columns:
        df['buyer_zip_band'] = df.apply(lambda r: 'same_zip' if pd.notna(r['owner_zip']) and pd.notna(r['situs_zip']) and r['owner_zip']==r['situs_zip'] else 'other', axis=1)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT, index=False)
    print('Wrote', OUT, f'({len(df):,} rows)')

if __name__ == '__main__':
    main()

