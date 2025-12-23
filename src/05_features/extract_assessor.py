#!/usr/bin/env python3
"""
Extract Douglas County parcels from Nebraska statewide assessor GDB.

Input:  statewide parcel/NE_2023_statewideparcels.gdb
Output: data_work/assessor_raw.parquet

This script extracts parcel data for Douglas County (County_ID = '055')
from the Nebraska statewide assessor geodatabase and saves it as parquet
for further processing.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

# Ensure we're in project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.chdir(PROJECT_ROOT)

# Force unbuffered output for real-time logging
sys.stdout.reconfigure(line_buffering=True)

def log(msg: str):
    """Print timestamped log message."""
    ts = time.strftime('%H:%M:%S')
    print(f'[{ts}] {msg}', flush=True)

log('Script starting...')
log(f'Working directory: {os.getcwd()}')

try:
    log('Importing geopandas...')
    import geopandas as gpd
    log(f'  geopandas version: {gpd.__version__}')
    log('Importing pandas...')
    import pandas as pd
    log(f'  pandas version: {pd.__version__}')
except ImportError as e:
    log(f'ERROR: Missing required package: {e}')
    print('Install with: pip install geopandas pandas pyarrow')
    sys.exit(1)

# Configuration
GDB_PATH = Path('statewide parcel/NE_2023_statewideparcels.gdb')
LAYER = 'StatewideParcels_Current'
COUNTY_ID = '055'  # Douglas County FIPS suffix

# Fields to extract (subset for performance and relevance)
FIELDS = [
    # Identification
    'Parcel_ID', 'State_PID', 'County_ID',
    # Building characteristics
    'BuildingYear', 'ImpSF', 'QualImp', 'CondImp',
    # Values
    'Total_Assessed_Value', 'Land_Value', 'Improvements_Value',
    # Parcel characteristics
    'GIS_Acres', 'Acres_Deeded',
    # Classification
    'Property_Parcel_Type', 'Property_Parcel_Status', 'Zoning',
    'Classification_Code', 'Property_SubClass',
    # Location
    'Neighborhood', 'Location', 'City_Size', 'Parcel_Size',
    # Owner info
    'Current_Owner_Name', 'Ownership_Type',
    # Address
    'Situs_Address', 'Ph_City', 'Ph_Zip5',
]

OUT_DIR = Path('data_work')
OUT_FILE = OUT_DIR / 'assessor_raw.parquet'


def main():
    """Extract Douglas County parcels from statewide GDB."""
    log('=' * 60)
    log('ASSESSOR DATA EXTRACTION')
    log('=' * 60)

    # Verify input exists
    log(f'Checking for geodatabase at: {GDB_PATH}')
    if not GDB_PATH.exists():
        raise SystemExit(f'ERROR: Geodatabase not found: {GDB_PATH}')

    gdb_size = sum(f.stat().st_size for f in GDB_PATH.rglob('*') if f.is_file()) / 1024 / 1024
    log(f'  GDB exists, size: {gdb_size:.1f} MB')

    log(f'Source: {GDB_PATH}')
    log(f'Layer: {LAYER}')
    log(f'Filter: County_ID = {COUNTY_ID!r}')

    # Check available engines
    log('Checking available GDB read engines...')
    try:
        import pyogrio
        log(f'  pyogrio available: {pyogrio.__version__}')
        engine = 'pyogrio'
    except ImportError:
        log('  pyogrio not available, will use fiona')
        engine = 'fiona'

    # Read with SQL filter for efficiency
    log(f'Reading geodatabase with {engine} engine...')
    log('  (This may take several minutes for a 735 MB file)')

    start_time = time.time()
    try:
        log('  Calling gpd.read_file()...')
        gdf = gpd.read_file(
            GDB_PATH,
            layer=LAYER,
            where=f"County_ID = '{COUNTY_ID}'",
            engine=engine
        )
        elapsed = time.time() - start_time
        log(f'  Read completed in {elapsed:.1f} seconds')
    except Exception as e:
        elapsed = time.time() - start_time
        log(f'ERROR reading GDB after {elapsed:.1f}s: {e}')
        log('Trying alternative read method (full read + filter)...')
        start_time = time.time()
        gdf = gpd.read_file(GDB_PATH, layer=LAYER, engine=engine)
        log(f'  Full read completed in {time.time() - start_time:.1f}s')
        log(f'  Filtering to County_ID = {COUNTY_ID}...')
        gdf = gdf[gdf['County_ID'] == COUNTY_ID]

    log(f'Loaded {len(gdf):,} Douglas County parcels')

    # Report available columns
    log(f'Available columns ({len(gdf.columns)}):')
    for col in sorted(gdf.columns):
        log(f'  - {col}')

    # Keep only needed fields + geometry
    log('Selecting requested fields...')
    available_fields = [c for c in FIELDS if c in gdf.columns]
    missing_fields = [c for c in FIELDS if c not in gdf.columns]

    if missing_fields:
        log(f'Note: {len(missing_fields)} requested fields not found:')
        for f in missing_fields:
            log(f'  - {f}')

    keep_cols = available_fields + ['geometry']
    gdf = gdf[keep_cols]
    log(f'Kept {len(keep_cols)} columns')

    # Check CRS and reproject to WGS84
    log(f'Original CRS: {gdf.crs}')
    log('Reprojecting to WGS84 (EPSG:4326)...')
    start_time = time.time()
    gdf = gdf.to_crs('EPSG:4326')
    log(f'  Reprojection completed in {time.time() - start_time:.1f}s')
    log(f'Reprojected to: {gdf.crs}')

    # Basic data quality checks
    log('--- Data Quality Summary ---')
    log(f"Parcel_ID nulls: {gdf['Parcel_ID'].isna().sum():,}")
    log(f"Parcel_ID duplicates: {gdf['Parcel_ID'].duplicated().sum():,}")

    if 'BuildingYear' in gdf.columns:
        non_null = gdf['BuildingYear'].notna().sum()
        log(f"BuildingYear coverage: {non_null:,} / {len(gdf):,} ({100*non_null/len(gdf):.1f}%)")

    if 'Total_Assessed_Value' in gdf.columns:
        has_value = (gdf['Total_Assessed_Value'] > 0).sum()
        log(f"Has assessed value: {has_value:,} / {len(gdf):,} ({100*has_value/len(gdf):.1f}%)")

    if 'Property_Parcel_Type' in gdf.columns:
        log('Property type distribution:')
        for ptype, count in gdf['Property_Parcel_Type'].value_counts().head(10).items():
            log(f'  {ptype}: {count:,}')

    # Save output
    OUT_DIR.mkdir(exist_ok=True)
    log(f'Saving to {OUT_FILE}...')
    start_time = time.time()
    gdf.to_parquet(OUT_FILE)
    log(f'  Parquet write completed in {time.time() - start_time:.1f}s')

    # Verify output
    size_mb = OUT_FILE.stat().st_size / 1024 / 1024
    log(f'Wrote {OUT_FILE} ({size_mb:.1f} MB)')

    log('=' * 60)
    log('EXTRACTION COMPLETE')
    log('=' * 60)

    return gdf


if __name__ == '__main__':
    main()
