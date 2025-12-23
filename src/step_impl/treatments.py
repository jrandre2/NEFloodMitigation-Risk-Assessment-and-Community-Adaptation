#!/usr/bin/env python3
"""
Module: treatments.py
Purpose: Assign treatment and exposure indicators to parcels.

This module creates the parcel-level treatment assignment file containing SFHA
membership indicators (majority-area and 10% overlap rules), 2019 inundation
status, and parcel coordinates for spatial analysis.

Input Files
-----------
- results/integration_run/sfr_regression_data.csv

Output Files
------------
- data_work/parcel_treatments.parquet

Functions
---------
ensure_env : Verify virtual environment is activated
main : Execute treatment assignment pipeline
"""
from __future__ import annotations
import os, sys
from pathlib import Path
import pandas as pd
import numpy as np

REG = Path('results/integration_run/sfr_regression_data.csv')
OUT = Path('data_work/parcel_treatments.parquet')

def ensure_env():
    if not os.getenv('VIRTUAL_ENV') or not os.getenv('VIRTUAL_ENV').endswith('/.venv'):
        print('ERROR: Please activate project .venv (source .venv/bin/activate) before running.', file=sys.stderr)
        sys.exit(1)

def main():
    """
    Execute treatment assignment pipeline.

    Reads regression data, extracts SFHA indicators (majority and 10% overlap),
    inundation status, and coordinates. Writes treatment file to parquet format.

    Raises
    ------
    SystemExit
        If the regression data file does not exist.
    """
    ensure_env()
    if not REG.exists():
        raise SystemExit(f'Missing regression data at {REG}')
    df = pd.read_csv(REG, low_memory=False)
    # Canonicalize columns
    idcol = 'parcel_id' if 'parcel_id' in df.columns else 'Parcel_ID'
    out = pd.DataFrame({
        'parcel_id': df[idcol].astype(str),
        'sfha_majority': pd.to_numeric(df.get('in_sfha'), errors='coerce'),
        'sfha_10pct': pd.to_numeric(df.get('in_sfha_correct'), errors='coerce'),
        'inund_201903': pd.to_numeric(df.get('was_inundated'), errors='coerce'),
        'x': pd.to_numeric(df.get('x_coord'), errors='coerce'),
        'y': pd.to_numeric(df.get('y_coord'), errors='coerce'),
    })
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT, index=False)
    print('Wrote', OUT)

if __name__ == '__main__':
    main()

