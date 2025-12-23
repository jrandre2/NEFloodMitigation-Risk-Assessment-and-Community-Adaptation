#!/usr/bin/env python3
"""
Integrate assessor covariates into analysis datasets.

Input:  data_work/assessor_clean.parquet
        data_work/parcel_boundary_distances.parquet
        data_work/panel_parcel_month.parquet (optional)
Output: data_work/parcel_covariates_full.parquet
        data_work/panel_parcel_month_enriched.parquet (if panel exists)

This script merges assessor covariates into the boundary distance data
and optionally into the panel data for use in covariate balance tests
and other analyses.
"""
from __future__ import annotations

import os
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
ASSESSOR_FILE = Path('data_work/assessor_clean.parquet')
BOUNDARY_FILE = Path('data_work/parcel_boundary_distances.parquet')
PANEL_FILE = Path('data_work/panel_parcel_month.parquet')

OUT_COVARIATES = Path('data_work/parcel_covariates_full.parquet')
OUT_PANEL = Path('data_work/panel_parcel_month_enriched.parquet')

# Covariates to include in balance tests
BALANCE_COVARIATES = [
    'year_built',
    'building_age',
    'Total_Assessed_Value',
    'log_assessed_value',
    'Land_Value',
    'log_land_value',
    'Improvements_Value',
    'log_improvement_value',
    'GIS_Acres',
    'log_acres',
    'ImpSF',
    'log_impsf',
    'is_sfr',
    'is_improved',
    'neighborhood',
    'property_type_label',
    'zoning_label',
]


def main():
    """Integrate assessor covariates into analysis datasets."""
    print('=' * 60)
    print('ASSESSOR DATA INTEGRATION')
    print('=' * 60)

    # Load assessor data
    if not ASSESSOR_FILE.exists():
        raise SystemExit(f'ERROR: Assessor file not found: {ASSESSOR_FILE}\n'
                        f'Run transform_assessor.py first.')

    print(f'\nLoading {ASSESSOR_FILE}...')
    assessor = pd.read_parquet(ASSESSOR_FILE)
    assessor['parcel_id'] = assessor['parcel_id'].astype(str).str.strip()
    print(f'Assessor parcels: {len(assessor):,}')

    # Load boundary data
    if not BOUNDARY_FILE.exists():
        raise SystemExit(f'ERROR: Boundary file not found: {BOUNDARY_FILE}')

    print(f'\nLoading {BOUNDARY_FILE}...')
    boundary = pd.read_parquet(BOUNDARY_FILE)
    boundary['parcel_id'] = boundary['parcel_id'].astype(str).str.strip()
    print(f'Boundary parcels: {len(boundary):,}')

    # Prepare merge columns
    merge_cols = ['parcel_id'] + [c for c in BALANCE_COVARIATES if c in assessor.columns]
    print(f'\nMerge columns: {len(merge_cols) - 1} covariates')

    # Drop duplicates in assessor data (keep first)
    assessor_dedup = assessor[merge_cols].drop_duplicates('parcel_id')
    print(f'Unique assessor parcels: {len(assessor_dedup):,}')

    # Merge
    print('\nMerging assessor covariates onto boundary data...')
    merged = boundary.merge(
        assessor_dedup,
        on='parcel_id',
        how='left'
    )

    # Report coverage
    print('\n--- Covariate Coverage ---')
    for col in BALANCE_COVARIATES:
        if col in merged.columns:
            coverage = merged[col].notna().mean()
            print(f'{col}: {100*coverage:.1f}%')

    # Save covariates file
    print(f'\nSaving {OUT_COVARIATES}...')
    merged.to_parquet(OUT_COVARIATES, index=False)
    size_mb = OUT_COVARIATES.stat().st_size / 1024 / 1024
    print(f'Wrote {OUT_COVARIATES} ({size_mb:.1f} MB)')

    # Optionally update panel
    if PANEL_FILE.exists():
        print(f'\n--- Updating Panel ---')
        print(f'Loading {PANEL_FILE}...')
        panel = pd.read_parquet(PANEL_FILE)
        panel['parcel_id'] = panel['parcel_id'].astype(str).str.strip()
        print(f'Panel rows: {len(panel):,}')

        # Check for existing columns
        overlap = set(panel.columns) & set(merge_cols) - {'parcel_id'}
        if overlap:
            print(f'WARNING: Overlapping columns will be overwritten: {overlap}')
            # Drop overlapping columns from panel before merge
            panel = panel.drop(columns=list(overlap))

        panel_merged = panel.merge(
            assessor_dedup,
            on='parcel_id',
            how='left'
        )

        print(f'Saving {OUT_PANEL}...')
        panel_merged.to_parquet(OUT_PANEL, index=False)
        size_mb = OUT_PANEL.stat().st_size / 1024 / 1024
        print(f'Wrote {OUT_PANEL} ({size_mb:.1f} MB)')
    else:
        print(f'\nNote: Panel file not found ({PANEL_FILE}), skipping panel enrichment')

    # Summary statistics on merged data
    print('\n--- Merged Data Summary ---')
    print(f'Total parcels: {len(merged):,}')

    # RD window coverage
    if 'signed_dist_sfha_m' in merged.columns:
        for caliper in [100, 200, 300]:
            window = merged[np.abs(merged['signed_dist_sfha_m']) <= caliper]
            if len(window) > 0:
                year_cov = window['year_built'].notna().mean() if 'year_built' in window.columns else 0
                value_cov = window['Total_Assessed_Value'].notna().mean() if 'Total_Assessed_Value' in window.columns else 0
                print(f'±{caliper}m: {len(window):,} parcels, {100*year_cov:.1f}% year_built, {100*value_cov:.1f}% value')

    print('\n' + '=' * 60)
    print('INTEGRATION COMPLETE')
    print('=' * 60)

    return merged


if __name__ == '__main__':
    main()
