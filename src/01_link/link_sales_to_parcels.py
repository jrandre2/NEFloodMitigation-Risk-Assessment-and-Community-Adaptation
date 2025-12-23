#!/usr/bin/env python3
"""
Module: link_sales_to_parcels.py
Purpose: Link sales transactions to parcel geometries using parcel identifiers.

This module joins sales data to parcel attributes by matching on APN (Assessor
Parcel Number). In the current MVP implementation, the join is already embedded
via parcel_id; this module adds a join quality flag for downstream tracking.

Input Files
-----------
- data_work/sales_raw.parquet

Output Files
------------
- data_work/sales_parcel_joined.parquet

Functions
---------
ensure_env : Verify virtual environment is activated
main : Execute the parcel linkage pipeline
"""
from __future__ import annotations
import os, sys
from pathlib import Path
import pandas as pd

SALES = Path('data_work/sales_raw.parquet')
OUT = Path('data_work/sales_parcel_joined.parquet')

def ensure_env():
    if not os.getenv('VIRTUAL_ENV') or not os.getenv('VIRTUAL_ENV').endswith('/.venv'):
        print('ERROR: Please activate project .venv (source .venv/bin/activate) before running.', file=sys.stderr)
        sys.exit(1)

def main():
    """
    Execute parcel linkage pipeline.

    Reads sales data and attaches a join quality flag indicating the linkage
    method used. Writes the result to parquet format.

    Raises
    ------
    SystemExit
        If the input sales file does not exist.
    """
    ensure_env()
    if not SALES.exists():
        raise SystemExit(f'Missing {SALES}; run 00_ingest first')
    df = pd.read_parquet(SALES)
    # At MVP, the parcel join is already embedded via parcel_id; attach a simple join_quality flag
    df['join_quality'] = 'by_parcel_id'
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT, index=False)
    print('Wrote', OUT, f'({len(df):,} rows)')

if __name__ == '__main__':
    main()

