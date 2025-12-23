#!/usr/bin/env python3
"""
Module: parcels.py
Purpose: Build single-family residential (SFR) parcel base layer with geometries.

This module creates the foundational parcel GeoPackage used throughout the analysis.
It extracts SFR parcels from classification and regression outputs, attaches spatial
coordinates, and creates point geometries for centroid-based spatial operations.

Input Files
-----------
- results/integration_run/parcels_with_classification.csv
- results/integration_run/sfr_regression_data.csv

Output Files
------------
- data_work/parcels_sfr.gpkg

Functions
---------
ensure_env : Verify virtual environment is activated
pick : Select first available column from candidates
main : Execute parcel base layer construction
"""
from __future__ import annotations
import os, sys
from pathlib import Path
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point

REG = Path('results/integration_run/sfr_regression_data.csv')
CLS = Path('results/integration_run/parcels_with_classification.csv')
OUT = Path('data_work/parcels_sfr.gpkg')

def ensure_env():
    if not os.getenv('VIRTUAL_ENV') or not os.getenv('VIRTUAL_ENV').endswith('/.venv'):
        print('ERROR: Please activate project .venv (source .venv/bin/activate) before running.', file=sys.stderr)
        sys.exit(1)

def pick(colnames: list[str], *cands: str) -> str | None:
    """
    Select the first available column name from a list of candidates.

    Parameters
    ----------
    colnames : list[str]
        Available column names to search.
    *cands : str
        Candidate column names in order of preference.

    Returns
    -------
    str or None
        First matching column name, or None if no match found.
    """
    for c in cands:
        if c in colnames:
            return c
    return None

def main():
    """
    Execute parcel base layer construction.

    Reads classification and regression data, filters to SFR parcels, extracts
    coordinates and attributes, creates point geometries, and writes to GeoPackage.

    Raises
    ------
    SystemExit
        If required input files do not exist.
    """
    ensure_env()
    if not CLS.exists():
        raise SystemExit(f'Missing classification CSV at {CLS}')
    if not REG.exists():
        raise SystemExit(f'Missing regression data at {REG}')
    # Read minimal columns for speed
    head = pd.read_csv(CLS, nrows=0).columns.tolist()
    use = [c for c in ['Parcel_ID','GIS_Acres_x','Bldg_Count','CentroidX','CentroidY','Neighborho_x','Neighborho_y','FloodZoneCategory','ActualInundated'] if c in head]
    cls = pd.read_csv(CLS, usecols=use)
    cls['Parcel_ID'] = cls['Parcel_ID'].astype(str)
    # SFR filter: prefer Property_P_x from CLS if present; otherwise fallback to REG
    if 'Property_P_x' in head:
        use_sfr = pd.read_csv(CLS, usecols=['Parcel_ID','Property_P_x'])
        use_sfr['Parcel_ID'] = use_sfr['Parcel_ID'].astype(str)
        cls = cls.merge(use_sfr, on='Parcel_ID', how='left')
        cls = cls[cls['Property_P_x'] == 1].copy()
    else:
        reg = pd.read_csv(REG, usecols=['parcel_id'] if 'parcel_id' in pd.read_csv(REG, nrows=0).columns else ['Parcel_ID'])
        # If SFR already filtered in REG, proceed without additional filter
    # Join core log variables from REG
    reg_cols = pd.read_csv(REG, nrows=0).columns.tolist()
    idcol = 'parcel_id' if 'parcel_id' in reg_cols else 'Parcel_ID'
    keep = [c for c in [idcol,'log_total_value','log_acres','was_inundated','in_sfha','in_sfha_correct'] if c in reg_cols]
    reg = pd.read_csv(REG, usecols=keep)
    reg[idcol] = reg[idcol].astype(str)
    reg = reg.drop_duplicates(subset=[idcol])
    cls = cls.merge(reg, left_on='Parcel_ID', right_on=idcol, how='left')

    # Canonical fields
    neighborhood = cls.get('Neighborho_x', cls.get('Neighborho_y'))
    x = pd.to_numeric(cls.get('CentroidX', np.nan), errors='coerce')
    y = pd.to_numeric(cls.get('CentroidY', np.nan), errors='coerce')
    g = gpd.GeoDataFrame({
        'parcel_id': cls['Parcel_ID'].astype(str),
        'neighborhood_id': neighborhood,
        'land_use': 'SFR',
        'value_log': cls.get('log_total_value'),
        'acres_log': cls.get('log_acres', np.log(pd.to_numeric(cls.get('GIS_Acres_x'), errors='coerce').clip(lower=0.001))),
        'bldg_age_log': np.nan,  # placeholder unless build-year available in source
        'has_building': (pd.to_numeric(cls.get('Bldg_Count', 0), errors='coerce').fillna(0) > 0).astype(int),
        'sfha_majority': cls.get('in_sfha', (cls.get('FloodZoneCategory')=='SFHA').astype(int) if 'FloodZoneCategory' in cls.columns else np.nan),
        'sfha_10pct': cls.get('in_sfha_correct', np.nan),
        'inund_201903': cls.get('ActualInundated', cls.get('was_inundated', np.nan)),
        'x': x,
        'y': y,
    })
    g['geometry'] = [Point(xy) if np.isfinite(xy[0]) and np.isfinite(xy[1]) else None for xy in zip(g['x'], g['y'])]
    g = gpd.GeoDataFrame(g, geometry='geometry', crs='EPSG:4326')
    OUT.parent.mkdir(parents=True, exist_ok=True)
    g.to_file(OUT, layer='parcels_sfr', driver='GPKG')
    print('Wrote', OUT)

if __name__ == '__main__':
    main()

