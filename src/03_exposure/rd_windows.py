#!/usr/bin/env python3
"""
Module: rd_windows.py
Purpose: Define RD caliper windows and ring indicators for boundary analysis.

This module merges boundary distances with parcel treatments and creates
caliper window indicators (±150m, ±300m) for RD analysis, as well as
distance-based ring labels for near-but-dry parcels outside the boundary.

Input Files
-----------
- data_work/parcel_treatments_with_rings.parquet
- data_work/parcel_boundary_distances.parquet

Output Files
------------
- data_work/parcel_treatments_rd.parquet

Functions
---------
ensure_env : Verify virtual environment is activated
ring_from_dist : Classify distance into ring categories
main : Execute RD window creation
"""
from __future__ import annotations
import os, sys
from pathlib import Path
import pandas as pd
import numpy as np

BASE = Path('data_work/parcel_treatments_with_rings.parquet')
DIST = Path('data_work/parcel_boundary_distances.parquet')
OUT = Path('data_work/parcel_treatments_rd.parquet')

def ensure_env():
    if not os.getenv('VIRTUAL_ENV') or not os.getenv('VIRTUAL_ENV').endswith('/.venv'):
        print('ERROR: Please activate project .venv (source .venv/bin/activate) before running.', file=sys.stderr)
        sys.exit(1)

def ring_from_dist(d: float) -> str:
    """
    Classify distance into ring categories.

    Parameters
    ----------
    d : float
        Distance in meters from boundary.

    Returns
    -------
    str
        Ring label: '0_250m', '250_500m', '500_1000m', or 'gt_1000m'.
    """
    if pd.isna(d):
        return np.nan
    if d <= 250: return '0_250m'
    if d <= 500: return '250_500m'
    if d <= 1000: return '500_1000m'
    return 'gt_1000m'

def main():
    """
    Execute RD window and ring creation pipeline.

    Merges boundary distances onto parcel treatments, creates RD caliper
    indicators (±150m, ±300m for SFHA and inundation), and assigns ring
    labels to parcels outside boundaries.

    Raises
    ------
    SystemExit
        If base treatment file does not exist.
    """
    ensure_env()
    if not BASE.exists():
        raise SystemExit(f'Missing {BASE}; run exposure_features first')
    if not DIST.exists():
        print(f'WARNING: {DIST} not found. RD flags will be unavailable.')
        df = pd.read_parquet(BASE)
        OUT.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(OUT, index=False)
        print('Wrote', OUT)
        return
    base = pd.read_parquet(BASE)
    dist = pd.read_parquet(DIST)
    base['parcel_id'] = base['parcel_id'].astype(str)
    dist['parcel_id'] = dist['parcel_id'].astype(str)
    df = base.merge(dist, on='parcel_id', how='left')
    # SFHA calipers
    df['rd_sfha_150'] = df['signed_dist_sfha_m'].abs() <= 150
    df['rd_sfha_300'] = df['signed_dist_sfha_m'].abs() <= 300
    # Inundation calipers
    df['rd_inund_150'] = df['signed_dist_inund_m'].abs() <= 150
    df['rd_inund_300'] = df['signed_dist_inund_m'].abs() <= 300
    # Near‑dry rings (outside only)
    df['sfha_ring'] = np.where(df['signed_dist_sfha_m']>0, df['signed_dist_sfha_m'].apply(ring_from_dist), np.nan)
    df['inund_ring'] = np.where(df['signed_dist_inund_m']>0, df['signed_dist_inund_m'].apply(ring_from_dist), np.nan)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT, index=False)
    print('Wrote', OUT)

if __name__ == '__main__':
    main()

