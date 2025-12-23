#!/usr/bin/env python3
"""
Module: run_all_diagnostics.py
Purpose: Single entry point to run all diagnostic and robustness analyses.

This module orchestrates the execution of all diagnostic scripts and
generates a comprehensive diagnostics report.

Output Files
------------
- data_work/diagnostics/comprehensive_diagnostics_report.md
- data_work/diagnostics/diagnostics_summary.json
"""
from __future__ import annotations
import argparse
import importlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# Add src directory to path for imports
src_dir = Path(__file__).parent.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

PANEL = Path('data_work/panel_parcel_month.parquet')
OUT_DIR = Path('data_work/diagnostics')


def ensure_env():
    if not os.getenv('VIRTUAL_ENV') or not os.getenv('VIRTUAL_ENV').endswith('/.venv'):
        print('ERROR: Please activate project .venv', file=sys.stderr)
        sys.exit(1)


def run_all(
    boundary: str = 'inund',
    caliper: int = 300,
    skip_permutation: bool = False
):
    """
    Run complete diagnostics suite.

    Parameters
    ----------
    boundary : str
        Primary boundary for analysis.
    caliper : int
        Caliper window.
    skip_permutation : bool
        Skip slow permutation tests.
    """
    ensure_env()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    results = {
        'run_timestamp': datetime.now().isoformat(),
        'boundary': boundary,
        'caliper_m': caliper,
        'modules': {}
    }

    print('=' * 70)
    print('COMPREHENSIVE DIAGNOSTICS SUITE')
    print('Freeze and Flight - Peer Review Robustness Analysis')
    print('=' * 70)
    print(f'Primary boundary: {boundary}')
    print(f'Caliper: ±{caliper}m')
    print(f'Timestamp: {results["run_timestamp"]}')
    print()

    # Helper to import modules from 07_estimation
    def import_module(name):
        return importlib.import_module(f'07_estimation.{name}')

    # 1. RD Diagnostics (McCrary, pre-trends, bandwidth)
    print('\n' + '=' * 70)
    print('MODULE 1: RD IDENTIFICATION DIAGNOSTICS')
    print('=' * 70)
    try:
        mod = import_module('rd_diagnostics')
        mod.main()
        results['modules']['rd_diagnostics'] = 'SUCCESS'
    except Exception as e:
        print(f'ERROR: {e}')
        results['modules']['rd_diagnostics'] = f'FAILED: {e}'

    # 2. Event Study
    print('\n' + '=' * 70)
    print('MODULE 2: EVENT STUDY')
    print('=' * 70)
    try:
        mod = import_module('event_study')
        mod.main(boundaries=[boundary, 'sfha'], calipers=[caliper])
        results['modules']['event_study'] = 'SUCCESS'
    except Exception as e:
        print(f'ERROR: {e}')
        results['modules']['event_study'] = f'FAILED: {e}'

    # 3. Placebo Tests
    print('\n' + '=' * 70)
    print('MODULE 3: PLACEBO AND FALSIFICATION TESTS')
    print('=' * 70)
    try:
        mod = import_module('placebo_tests')
        n_perm = 100 if skip_permutation else 500
        mod.main(boundary=boundary, caliper=caliper, n_permutations=n_perm)
        results['modules']['placebo_tests'] = 'SUCCESS'
    except Exception as e:
        print(f'ERROR: {e}')
        results['modules']['placebo_tests'] = f'FAILED: {e}'

    # 4. Spatial Econometrics
    print('\n' + '=' * 70)
    print('MODULE 4: SPATIAL ECONOMETRICS')
    print('=' * 70)
    try:
        mod = import_module('spatial_econometrics')
        mod.main(boundary=boundary, caliper=caliper)
        results['modules']['spatial_econometrics'] = 'SUCCESS'
    except Exception as e:
        print(f'ERROR: {e}')
        results['modules']['spatial_econometrics'] = f'FAILED: {e}'

    # 5. Selection Correction
    print('\n' + '=' * 70)
    print('MODULE 5: SELECTION CORRECTION')
    print('=' * 70)
    try:
        mod = import_module('selection_correction')
        mod.main(boundary=boundary, caliper=caliper)
        results['modules']['selection_correction'] = 'SUCCESS'
    except Exception as e:
        print(f'ERROR: {e}')
        results['modules']['selection_correction'] = f'FAILED: {e}'

    # 6. Quantile Effects
    print('\n' + '=' * 70)
    print('MODULE 6: QUANTILE EFFECTS')
    print('=' * 70)
    try:
        mod = import_module('quantile_effects')
        mod.main(boundary=boundary, caliper=caliper)
        results['modules']['quantile_effects'] = 'SUCCESS'
    except Exception as e:
        print(f'ERROR: {e}')
        results['modules']['quantile_effects'] = f'FAILED: {e}'

    # 7. Extended Horizon
    print('\n' + '=' * 70)
    print('MODULE 7: EXTENDED HORIZON')
    print('=' * 70)
    try:
        mod = import_module('extended_horizon')
        mod.main(boundary=boundary, caliper=caliper)
        results['modules']['extended_horizon'] = 'SUCCESS'
    except Exception as e:
        print(f'ERROR: {e}')
        results['modules']['extended_horizon'] = f'FAILED: {e}'

    # 8. Mechanism Analysis
    print('\n' + '=' * 70)
    print('MODULE 8: MECHANISM ANALYSIS')
    print('=' * 70)
    try:
        mod = import_module('mechanism_analysis')
        mod.main(boundary=boundary, caliper=caliper)
        results['modules']['mechanism_analysis'] = 'SUCCESS'
    except Exception as e:
        print(f'ERROR: {e}')
        results['modules']['mechanism_analysis'] = f'FAILED: {e}'

    # Save summary
    summary_path = OUT_DIR / 'diagnostics_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nWrote {summary_path}')

    # Generate report
    generate_report(results)

    # Final summary
    print('\n' + '=' * 70)
    print('DIAGNOSTICS SUITE COMPLETE')
    print('=' * 70)
    n_success = sum(1 for v in results['modules'].values() if v == 'SUCCESS')
    n_total = len(results['modules'])
    print(f'Modules completed: {n_success}/{n_total}')
    print(f'Output directory: {OUT_DIR}')


def generate_report(results: dict):
    """Generate comprehensive markdown report."""
    lines = [
        '# Comprehensive Diagnostics Report',
        '',
        f"**Generated**: {results['run_timestamp']}",
        f"**Primary Boundary**: {results['boundary']}",
        f"**Caliper**: ±{results['caliper_m']}m",
        '',
        '---',
        '',
        '## Module Execution Summary',
        '',
        '| Module | Status |',
        '|--------|--------|',
    ]

    for module, status in results['modules'].items():
        status_icon = '✅' if status == 'SUCCESS' else '❌'
        lines.append(f'| {module} | {status_icon} {status} |')

    lines.extend([
        '',
        '---',
        '',
        '## Key Outputs',
        '',
        '### Identification Diagnostics',
        '- `mccrary_density_test.csv` - Manipulation test',
        '- `pretrends_ftest.csv` - Parallel trends test',
        '- `bandwidth_sensitivity.csv` - Bandwidth robustness',
        '- `donut_rd.csv` - Spillover robustness',
        '',
        '### Placebo Tests',
        '- `placebo_event_dates.csv` - Fake event dates',
        '- `placebo_boundaries.csv` - Shifted boundaries',
        '- `permutation_pvalues.csv` - Randomization inference',
        '',
        '### Spatial Analysis',
        '- `conley_se_comparison.csv` - Spatial standard errors',
        '- `moran_residual_test.csv` - Residual autocorrelation',
        '- `spatial_lag_results.csv` - SAR model',
        '- `spatial_error_results.csv` - SEM model',
        '',
        '### Selection Correction',
        '- `heckman_results.csv` - Heckman two-step',
        '- `ipw_results.csv` - Inverse probability weighting',
        '- `lee_bounds.csv` - Lee (2009) bounds',
        '',
        '### Distribution',
        '- `quantile_did.csv` - Quantile treatment effects',
        '',
        '### Persistence',
        '- `dynamic_effects_extended.csv` - Extended event study',
        '- `persistence_analysis.csv` - Effect persistence',
        '',
        '### Mechanisms',
        '- `mechanism_insurance.csv` - Insurance channel',
        '- `mechanism_credit.csv` - Credit constraints',
        '- `heterogeneity_by_chars.csv` - Property characteristics',
        '- `mechanism_summary.csv` - Summary table',
        '',
        '---',
        '',
        '## Figures Generated',
        '',
        '- `fig_event_study_*.png` - Event studies with CI',
        '- `fig_mccrary_*.png` - Density discontinuity',
        '- `fig_bandwidth_*.png` - Bandwidth sensitivity',
        '- `fig_placebo_*.png` - Placebo tests',
        '- `fig_permutation_distribution.png` - Permutation null',
        '- `fig_selection_correction_comparison.png` - Selection methods',
        '- `fig_quantile_effects.png` - Quantile effects',
        '- `fig_extended_event_study.png` - Extended horizon',
        '- `fig_mechanism_heterogeneity.png` - Heterogeneous effects',
        '',
    ])

    report_path = OUT_DIR / 'comprehensive_diagnostics_report.md'
    report_path.write_text('\n'.join(lines))
    print(f'Wrote {report_path}')


def main():
    parser = argparse.ArgumentParser(
        description='Run all diagnostics for Freeze and Flight analysis.'
    )
    parser.add_argument('--boundary', '-b', default='inund',
                        help='Primary boundary (default: inund)')
    parser.add_argument('--caliper', '-c', type=int, default=300,
                        help='Caliper in meters (default: 300)')
    parser.add_argument('--skip-permutation', action='store_true',
                        help='Skip slow permutation tests')
    args = parser.parse_args()

    run_all(
        boundary=args.boundary,
        caliper=args.caliper,
        skip_permutation=args.skip_permutation
    )


if __name__ == '__main__':
    main()
