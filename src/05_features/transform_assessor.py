#!/usr/bin/env python3
"""
Transform and clean assessor data for analysis.

Input:  data_work/assessor_raw.parquet
Output: data_work/assessor_clean.parquet

This script:
1. Parses BuildingYear (string) to integer year_built
2. Computes building_age = 2019 - year_built
3. Creates log transforms of numeric variables
4. Maps property type codes to labels
5. Standardizes parcel_id format
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

# Ensure we're in project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.chdir(PROJECT_ROOT)

try:
    import pandas as pd
    import numpy as np
except ImportError as e:
    print(f'ERROR: Missing required package: {e}')
    print('Install with: pip install pandas numpy')
    sys.exit(1)

# Configuration
IN_FILE = Path('data_work/assessor_raw.parquet')
OUT_FILE = Path('data_work/assessor_clean.parquet')

# Property type codes from data dictionary
PROPERTY_TYPE_MAP = {
    '0': 'None',
    '1': 'SFR',
    '2': 'Multi-Family',
    '3': 'Commercial',
    '4': 'Industrial',
    '5': 'Agriculture',
    '6': 'Recreational',
    '7': 'Mobile Home',
    '8': 'Minerals-NP',
    '9': 'Minerals-P',
    '10': 'State',
    '11': 'Exempt',
    '12': 'Other',
}

ZONING_MAP = {
    '0': 'None',
    '1': 'SF',
    '2': 'MF',
    '3': 'Commercial',
    '4': 'Industrial',
    '5': 'Agriculture',
    '6': 'Recreational',
    '7': 'Other',
}

LOCATION_MAP = {
    '0': 'None',
    '01': 'Urban',
    '1': 'Urban',
    '02': 'Suburban',
    '2': 'Suburban',
    '03': 'Rural',
    '3': 'Rural',
}


def parse_building_year(year_str) -> float:
    """
    Parse BuildingYear string to integer.

    Handles various formats:
    - "1965" -> 1965
    - "1965-1970" -> 1965 (takes first year)
    - "N/A", None, "" -> np.nan
    - Invalid values -> np.nan

    Returns
    -------
    float
        Year as float (to allow NaN) or np.nan if invalid
    """
    if pd.isna(year_str):
        return np.nan

    year_str = str(year_str).strip()

    if year_str in ('', 'N/A', 'NA', 'None', 'null', '0'):
        return np.nan

    # Try direct integer conversion
    try:
        year = int(float(year_str))
        if 1800 <= year <= 2024:
            return float(year)
        return np.nan
    except (ValueError, TypeError):
        pass

    # Try extracting first 4-digit year from string (e.g., "1965-1970")
    match = re.search(r'\b(18|19|20)\d{2}\b', year_str)
    if match:
        return float(match.group())

    return np.nan


def main():
    """Transform and clean assessor data."""
    print('=' * 60)
    print('ASSESSOR DATA TRANSFORMATION')
    print('=' * 60)

    # Load raw data
    if not IN_FILE.exists():
        raise SystemExit(f'ERROR: Input file not found: {IN_FILE}\n'
                        f'Run extract_assessor.py first.')

    print(f'\nLoading {IN_FILE}...')
    df = pd.read_parquet(IN_FILE)
    print(f'Loaded {len(df):,} parcels')

    # Standardize parcel_id
    print('\nStandardizing parcel_id...')
    df['parcel_id'] = df['Parcel_ID'].astype(str).str.strip()

    # Parse building year
    print('Parsing building years...')
    if 'BuildingYear' in df.columns:
        df['year_built'] = df['BuildingYear'].apply(parse_building_year)
        df['building_age'] = 2019 - df['year_built']  # Age at flood event
        df['building_age'] = df['building_age'].clip(lower=0)  # No negative ages

        # Report parsing results
        parsed = df['year_built'].notna().sum()
        print(f'  Parsed: {parsed:,} / {len(df):,} ({100*parsed/len(df):.1f}%)')
        if parsed > 0:
            print(f'  Year range: {df["year_built"].min():.0f} - {df["year_built"].max():.0f}')
            print(f'  Age range: {df["building_age"].min():.0f} - {df["building_age"].max():.0f} years')
    else:
        print('  WARNING: BuildingYear column not found')
        df['year_built'] = np.nan
        df['building_age'] = np.nan

    # Log transforms (use log1p to handle zeros)
    print('\nCreating log transforms...')
    numeric_cols = {
        'Total_Assessed_Value': 'log_assessed_value',
        'Land_Value': 'log_land_value',
        'Improvements_Value': 'log_improvement_value',
        'GIS_Acres': 'log_acres',
        'ImpSF': 'log_impsf',
    }

    for src_col, dst_col in numeric_cols.items():
        if src_col in df.columns:
            df[dst_col] = np.log1p(pd.to_numeric(df[src_col], errors='coerce').fillna(0))
            print(f'  {src_col} -> {dst_col}')
        else:
            df[dst_col] = np.nan
            print(f'  {src_col} (missing) -> {dst_col} = NaN')

    # Property type mapping
    print('\nMapping property types...')
    if 'Property_Parcel_Type' in df.columns:
        df['property_type_label'] = df['Property_Parcel_Type'].astype(str).map(PROPERTY_TYPE_MAP)
        df['is_sfr'] = (df['Property_Parcel_Type'].astype(str) == '1').astype(int)
        print(f'  SFR parcels: {df["is_sfr"].sum():,}')
    else:
        df['property_type_label'] = None
        df['is_sfr'] = 0

    # Zoning mapping
    if 'Zoning' in df.columns:
        df['zoning_label'] = df['Zoning'].astype(str).map(ZONING_MAP)

    # Location mapping
    if 'Location' in df.columns:
        df['location_label'] = df['Location'].astype(str).map(LOCATION_MAP)

    # Property status
    if 'Property_Parcel_Status' in df.columns:
        df['is_improved'] = (df['Property_Parcel_Status'].astype(str) == '1').astype(int)
        print(f'  Improved parcels: {df["is_improved"].sum():,}')
    else:
        df['is_improved'] = 0

    # Standardize neighborhood for FE matching
    if 'Neighborhood' in df.columns:
        df['neighborhood'] = df['Neighborhood'].astype(str).str.strip().str.upper()

    # Quality/Condition
    for col in ['QualImp', 'CondImp']:
        if col in df.columns:
            df[col.lower()] = df[col].astype(str).str.strip()

    # Select output columns
    out_cols = [
        # Identification
        'parcel_id',
        # Building characteristics
        'year_built', 'building_age',
        # Values (raw)
        'Total_Assessed_Value', 'Land_Value', 'Improvements_Value',
        # Values (log)
        'log_assessed_value', 'log_land_value', 'log_improvement_value',
        # Parcel size
        'GIS_Acres', 'log_acres',
        # Building size
        'ImpSF', 'log_impsf',
        # Classification
        'Property_Parcel_Type', 'property_type_label', 'is_sfr',
        'Zoning', 'zoning_label',
        'is_improved',
        # Location
        'neighborhood',
        # Quality
        'qualimp', 'condimp',
        # Owner
        'Ownership_Type',
        # Classification codes
        'Classification_Code',
    ]

    # Keep geometry for spatial validation
    if 'geometry' in df.columns:
        out_cols.append('geometry')

    # Filter to columns that exist
    out_cols = [c for c in out_cols if c in df.columns]
    df_out = df[out_cols].copy()

    # Summary statistics
    print('\n--- Transform Summary ---')
    print(f'Output columns: {len(out_cols)}')
    print(f'Output rows: {len(df_out):,}')

    for col in ['year_built', 'Total_Assessed_Value', 'GIS_Acres', 'ImpSF']:
        if col in df_out.columns:
            coverage = df_out[col].notna().mean() * 100
            print(f'{col}: {coverage:.1f}% coverage')

    # Save
    print(f'\nSaving to {OUT_FILE}...')
    df_out.to_parquet(OUT_FILE, index=False)

    size_mb = OUT_FILE.stat().st_size / 1024 / 1024
    print(f'Wrote {OUT_FILE} ({size_mb:.1f} MB)')

    print('\n' + '=' * 60)
    print('TRANSFORMATION COMPLETE')
    print('=' * 60)

    return df_out


if __name__ == '__main__':
    main()
