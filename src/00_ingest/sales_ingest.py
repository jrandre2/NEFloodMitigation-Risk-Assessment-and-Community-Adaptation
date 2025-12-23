#!/usr/bin/env python3
"""
Module: sales_ingest.py
Purpose: Load and preprocess raw sales transaction data from county assessor files.

This module ingests sales data from the classification and regression CSV outputs,
standardizes field names, parses dates, filters invalid prices, and creates unique
sale identifiers for downstream processing.

Input Files
-----------
- results/integration_run/parcels_with_classification.csv
- results/integration_run/sfr_regression_data.csv

Output Files
------------
- data_work/sales_raw.parquet

Functions
---------
ensure_env : Verify virtual environment is activated
main : Execute the sales ingestion pipeline
"""
from __future__ import annotations
import os, sys
from pathlib import Path
import pandas as pd
import numpy as np

CLS = Path('results/integration_run/parcels_with_classification.csv')
REG = Path('results/integration_run/sfr_regression_data.csv')
OUT = Path('data_work/sales_raw.parquet')

def ensure_env():
    if not os.getenv('VIRTUAL_ENV') or not os.getenv('VIRTUAL_ENV').endswith('/.venv'):
        print('ERROR: Please activate project .venv (source .venv/bin/activate) before running.', file=sys.stderr)
        sys.exit(1)

def _safe_cols(path: Path) -> list[str]:
    """
    Read only column headers from a CSV file.

    Parameters
    ----------
    path : Path
        Path to the CSV file.

    Returns
    -------
    list[str]
        List of column names in the file.
    """
    return pd.read_csv(path, nrows=0).columns.tolist()

def main():
    """
    Execute sales data ingestion pipeline.

    Reads sales transaction data from classification and regression CSV files,
    standardizes field names, parses dates, filters out invalid prices, creates
    unique sale identifiers, and writes to parquet format.

    Raises
    ------
    SystemExit
        If neither source file exists or no usable sale fields are found.
    """
    ensure_env()
    if not CLS.exists() and not REG.exists():
        raise SystemExit('Need at least one source (classification or regression) to derive sales')

    rows = []

    # 1) Try classification CSV for raw sale fields if available
    if CLS.exists():
        cols = _safe_cols(CLS)
        use = [c for c in ['Parcel_ID','Sales_Date_x','Sales_Valu_x','Sales_Date_y','Sales_Valu_y','Instrument','DeedType','Legal','Legal_Desc'] if c in cols]
        if use:
            cls = pd.read_csv(CLS, usecols=use)
            cls['Parcel_ID'] = cls['Parcel_ID'].astype(str)
            # Prefer _x then _y
            sale_date = pd.to_datetime(cls['Sales_Date_x'] if 'Sales_Date_x' in cls.columns else cls.get('Sales_Date_y'), errors='coerce')
            price = pd.to_numeric(cls['Sales_Valu_x'] if 'Sales_Valu_x' in cls.columns else cls.get('Sales_Valu_y'), errors='coerce')
            part = pd.DataFrame({
                'parcel_id': cls['Parcel_ID'],
                'sale_date': sale_date,
                'price_nominal': price,
                'instrument': cls.get('Instrument', cls.get('DeedType')),
                'legal_desc': cls.get('Legal', cls.get('Legal_Desc')),
            })
            part = part.dropna(subset=['sale_date'])
            rows.append(part)

    # 2) Supplement from regression data (if sales present)
    if REG.exists():
        cols = _safe_cols(REG)
        use = [c for c in ['parcel_id','Parcel_ID','sale_date','log_sales_value'] if c in cols]
        if use:
            reg = pd.read_csv(REG, usecols=use)
            pid = 'parcel_id' if 'parcel_id' in reg.columns else 'Parcel_ID'
            reg['parcel_id'] = reg[pid].astype(str)
            reg['sale_date'] = pd.to_datetime(reg['sale_date'], errors='coerce')
            reg['price_nominal'] = np.exp(pd.to_numeric(reg.get('log_sales_value'), errors='coerce'))
            part = reg[['parcel_id','sale_date','price_nominal']].dropna(subset=['sale_date'])
            rows.append(part)

    if not rows:
        raise SystemExit('No usable sale fields found in sources')

    sales = pd.concat(rows, ignore_index=True)
    # Normalize price
    sales['price_nominal'] = pd.to_numeric(sales['price_nominal'], errors='coerce')
    sales = sales.dropna(subset=['price_nominal'])
    sales = sales[sales['price_nominal'] > 0]
    # Create sale_id
    sales['sale_id'] = sales['parcel_id'] + '::' + sales['sale_date'].astype(str)
    # Deduplicate
    sales = sales.drop_duplicates(subset=['sale_id'])

    OUT.parent.mkdir(parents=True, exist_ok=True)
    sales.to_parquet(OUT, index=False)
    print('Wrote', OUT, f'({len(sales):,} rows)')

if __name__ == '__main__':
    main()

