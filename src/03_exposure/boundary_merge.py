#!/usr/bin/env python3
"""
Module: boundary_merge.py
Purpose: Merge chunked boundary distance calculations into a single file.

This module combines all chunk-specific parquet files produced by
boundary_chunk_dist.py into a single consolidated parcel boundary
distances file, removing duplicates.

Input Files
-----------
- data_work/parcel_boundary_distances_part_*_of_*.parquet

Output Files
------------
- data_work/parcel_boundary_distances.parquet

Functions
---------
ensure_env : Verify virtual environment is activated
main : Execute merge operation
"""
from __future__ import annotations
import os
from pathlib import Path
import pandas as pd

OUT_DIR = Path('data_work')
OUT = OUT_DIR / 'parcel_boundary_distances.parquet'

def ensure_env():
    if not os.getenv('VIRTUAL_ENV') or not os.getenv('VIRTUAL_ENV').endswith('/.venv'):
        raise SystemExit('Activate .venv first')

def main():
    """
    Execute chunk merge operation.

    Reads all chunk parquet files, concatenates, removes duplicate parcel
    entries, and writes the consolidated boundary distances file.

    Raises
    ------
    SystemExit
        If no chunk files are found.
    """
    ensure_env()
    parts = sorted(OUT_DIR.glob('parcel_boundary_distances_part_*_of_*.parquet'))
    if not parts:
        raise SystemExit('No part files found')
    dfs = [pd.read_parquet(p) for p in parts]
    df = pd.concat(dfs, ignore_index=True)
    df.drop_duplicates(subset=['parcel_id'], inplace=True)
    df.to_parquet(OUT, index=False)
    print('Wrote', OUT, f'({len(df):,} rows from {len(parts)} parts)')

if __name__ == '__main__':
    main()

