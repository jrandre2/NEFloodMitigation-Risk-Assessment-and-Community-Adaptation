#!/usr/bin/env python3
"""
Validate assessor data and check match rates with existing parcel data.

Input:  data_work/assessor_clean.parquet
        data_work/parcel_boundary_distances.parquet
Output: data_work/diagnostics/assessor_match_report.csv

This script validates the assessor data by checking:
1. Match rate with existing parcel_boundary_distances data
2. Coverage of key fields
3. Data quality issues
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
except ImportError as e:
    print(f'ERROR: Missing required package: {e}')
    print('Install with: pip install pandas')
    sys.exit(1)

# Configuration
ASSESSOR_FILE = Path('data_work/assessor_clean.parquet')
BOUNDARY_FILE = Path('data_work/parcel_boundary_distances.parquet')
OUT_DIR = Path('data_work/diagnostics')


def main():
    """Validate assessor data and report match rates."""
    print('=' * 60)
    print('ASSESSOR DATA VALIDATION')
    print('=' * 60)

    # Load assessor data
    if not ASSESSOR_FILE.exists():
        raise SystemExit(f'ERROR: Assessor file not found: {ASSESSOR_FILE}\n'
                        f'Run transform_assessor.py first.')

    print(f'\nLoading {ASSESSOR_FILE}...')
    assessor = pd.read_parquet(ASSESSOR_FILE)
    assessor['parcel_id'] = assessor['parcel_id'].astype(str).str.strip()
    print(f'Assessor parcels: {len(assessor):,}')

    # Load existing boundary data
    if not BOUNDARY_FILE.exists():
        print(f'\nWARNING: Boundary file not found: {BOUNDARY_FILE}')
        print('Cannot compute match rates. Skipping validation.')
        return None

    print(f'\nLoading {BOUNDARY_FILE}...')
    existing = pd.read_parquet(BOUNDARY_FILE)
    existing['parcel_id'] = existing['parcel_id'].astype(str).str.strip()
    print(f'Existing parcels: {len(existing):,}')

    # Match analysis
    print('\n--- Match Analysis ---')
    assessor_ids = set(assessor['parcel_id'])
    existing_ids = set(existing['parcel_id'])

    matched = assessor_ids & existing_ids
    assessor_only = assessor_ids - existing_ids
    existing_only = existing_ids - assessor_ids

    print(f'Matched: {len(matched):,} ({100*len(matched)/len(existing_ids):.1f}% of existing)')
    print(f'Assessor only: {len(assessor_only):,}')
    print(f'Existing only: {len(existing_only):,}')

    # Check if we need ID format adjustments
    if len(matched) < len(existing_ids) * 0.5:
        print('\nWARNING: Low match rate. Checking ID formats...')

        # Sample IDs
        print('\nSample assessor IDs:')
        for pid in list(assessor_ids)[:5]:
            print(f'  {pid!r} (len={len(pid)})')

        print('\nSample existing IDs:')
        for pid in list(existing_ids)[:5]:
            print(f'  {pid!r} (len={len(pid)})')

        # Try stripping leading zeros
        assessor_stripped = {str(int(pid)) if pid.isdigit() else pid for pid in assessor_ids}
        existing_stripped = {str(int(pid)) if pid.isdigit() else pid for pid in existing_ids}
        matched_stripped = assessor_stripped & existing_stripped

        if len(matched_stripped) > len(matched):
            print(f'\nMatch rate improves with integer conversion:')
            print(f'  Original match: {len(matched):,}')
            print(f'  Stripped match: {len(matched_stripped):,}')

    # SFR subset analysis
    print('\n--- SFR Subset ---')
    if 'is_sfr' in assessor.columns:
        sfr_assessor = assessor[assessor['is_sfr'] == 1]
        sfr_ids = set(sfr_assessor['parcel_id'])
        sfr_matched = sfr_ids & existing_ids
        print(f'SFR in assessor: {len(sfr_ids):,}')
        print(f'SFR matched: {len(sfr_matched):,}')
    else:
        print('No is_sfr column in assessor data')

    # RD window subset
    print('\n--- RD Window (±300m) ---')
    if 'signed_dist_sfha_m' in existing.columns:
        import numpy as np
        rd_parcels = existing[np.abs(existing['signed_dist_sfha_m']) <= 300]
        rd_ids = set(rd_parcels['parcel_id'])
        rd_matched = rd_ids & assessor_ids
        print(f'RD parcels: {len(rd_ids):,}')
        print(f'RD matched: {len(rd_matched):,} ({100*len(rd_matched)/len(rd_ids):.1f}%)')
    else:
        print('No signed_dist_sfha_m column in existing data')

    # Coverage analysis
    print('\n--- Field Coverage in Assessor Data ---')
    coverage_fields = ['year_built', 'Total_Assessed_Value', 'GIS_Acres', 'ImpSF', 'neighborhood']
    for field in coverage_fields:
        if field in assessor.columns:
            non_null = assessor[field].notna().sum()
            pct = 100 * non_null / len(assessor)
            print(f'{field}: {non_null:,} / {len(assessor):,} ({pct:.1f}%)')

    # Write report
    OUT_DIR.mkdir(exist_ok=True)
    report = pd.DataFrame({
        'metric': [
            'assessor_total',
            'existing_total',
            'matched',
            'assessor_only',
            'existing_only',
            'match_rate',
        ],
        'value': [
            len(assessor),
            len(existing),
            len(matched),
            len(assessor_only),
            len(existing_only),
            len(matched) / len(existing_ids) if len(existing_ids) > 0 else 0,
        ]
    })

    out_file = OUT_DIR / 'assessor_match_report.csv'
    report.to_csv(out_file, index=False)
    print(f'\nWrote {out_file}')

    print('\n' + '=' * 60)
    print('VALIDATION COMPLETE')
    print('=' * 60)

    return {
        'matched': len(matched),
        'match_rate': len(matched) / len(existing_ids) if len(existing_ids) > 0 else 0,
    }


if __name__ == '__main__':
    main()
