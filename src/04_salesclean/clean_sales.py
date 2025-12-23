#!/usr/bin/env python3
"""
Module: clean_sales.py
Purpose: Apply arms-length transaction filters to sales data.

This module filters sales to retain only market transactions by removing
nominal-consideration transfers, applying price filters, and winsorizing
extreme values.

Input Files
-----------
- data_work/sales_parties_labeled.parquet

Output Files
------------
- data_work/sales_clean.parquet

Functions
---------
ensure_env : Verify virtual environment is activated
main : Execute sales cleaning pipeline
"""
from __future__ import annotations
import os, sys
from pathlib import Path
import pandas as pd
import numpy as np

RAW = Path('data_work/sales_parties_labeled.parquet')
OUT = Path('data_work/sales_clean.parquet')

def ensure_env():
    if not os.getenv('VIRTUAL_ENV') or not os.getenv('VIRTUAL_ENV').endswith('/.venv'):
        print('ERROR: Please activate project .venv (source .venv/bin/activate) before running.', file=sys.stderr)
        sys.exit(1)

def main():
    """
    Execute sales cleaning pipeline.

    Reads labeled sales, filters to valid prices (>$1000), winsorizes at
    1-99 percentiles, and creates price_real variable. Writes cleaned
    sales to parquet format.

    Raises
    ------
    SystemExit
        If input file does not exist.
    """
    ensure_env()
    if not RAW.exists():
        raise SystemExit(f'Missing {RAW}; run labeling first')
    df = pd.read_parquet(RAW)
    # Basic arms-length proxy filters (MVP)
    df = df.dropna(subset=['price_nominal','sale_date']).copy()
    df = df[(df['price_nominal'] > 1000)]
    # Winsorize nominal price (1-99)
    q1, q99 = df['price_nominal'].quantile([0.01, 0.99]).values
    df['price_nominal_w'] = df['price_nominal'].clip(q1, q99)
    # Placeholder deflation (keep nominal as proxy)
    df['price_real'] = df['price_nominal_w']
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT, index=False)
    print('Wrote', OUT, f'({len(df):,} rows)')

if __name__ == '__main__':
    main()

