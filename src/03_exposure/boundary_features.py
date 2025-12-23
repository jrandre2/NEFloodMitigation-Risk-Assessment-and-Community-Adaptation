#!/usr/bin/env python3
"""
Module: boundary_features.py
Purpose: Create near-but-dry ring features for Poisson substitution analysis.

This module enriches parcel treatments with distance-based ring indicators
for the near-but-dry analysis. Parcels outside hazard zones are classified
into distance bands (0-250m, 250-500m, 500-1000m, >1000m) from the boundary.

Input Files
-----------
- data_work/parcel_treatments.parquet
- results/spatial_risk_analysis/distance_to_inundation_analysis.csv (optional)

Output Files
------------
- data_work/parcel_treatments_with_rings.parquet

Functions
---------
ensure_env : Verify virtual environment is activated
main : Execute ring feature creation
"""
from __future__ import annotations
import os, sys
from pathlib import Path
import pandas as pd
import numpy as np

TREAT = Path('data_work/parcel_treatments.parquet')
ALT_INUND = Path('results/spatial_risk_analysis/distance_to_inundation_analysis.csv')
OUT = Path('data_work/parcel_treatments_with_rings.parquet')

def ensure_env():
    if not os.getenv('VIRTUAL_ENV') or not os.getenv('VIRTUAL_ENV').endswith('/.venv'):
        print('ERROR: Please activate project .venv (source .venv/bin/activate) before running.', file=sys.stderr)
        sys.exit(1)

def main():
    """
    Execute ring feature creation pipeline.

    Reads parcel treatments, optionally merges precomputed inundation distances,
    and creates near-but-dry ring labels for parcels outside the hazard zone.

    Raises
    ------
    SystemExit
        If treatment file does not exist.
    """
    ensure_env()
    if not TREAT.exists():
        raise SystemExit(f'Missing {TREAT}; run build_treatments first')
    df = pd.read_parquet(TREAT)
    # Try to augment with precomputed distance to inundation if available
    if ALT_INUND.exists():
        alt = pd.read_csv(ALT_INUND, usecols=['parcel_id','distance_to_inundation','distance_band'])
        alt['parcel_id'] = alt['parcel_id'].astype(str)
        df['parcel_id'] = df['parcel_id'].astype(str)
        df = df.merge(alt, on='parcel_id', how='left')
    else:
        df['distance_to_inundation'] = np.nan
        df['distance_band'] = np.nan
    # Near‑but‑dry rings (placeholder derived from distance_to_inundation if present)
    def ring_label(d):
        if pd.isna(d):
            return np.nan
        if d <= 250: return '0_250m'
        if d <= 500: return '250_500m'
        if d <= 1000: return '500_1000m'
        return 'gt_1000m'
    df['near_dry_ring'] = df.apply(lambda r: ring_label(r['distance_to_inundation']) if (r.get('inund_201903',0)!=1) else np.nan, axis=1)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT, index=False)
    print('Wrote', OUT)

if __name__ == '__main__':
    main()

