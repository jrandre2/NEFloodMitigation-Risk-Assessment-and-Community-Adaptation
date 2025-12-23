#!/usr/bin/env python3
"""
Module: pipeline.py
Purpose: Main orchestration CLI for the Freeze and Flight analysis pipeline.

This module provides a command-line interface to execute individual stages of
the flood analysis pipeline. Each stage processes data and produces intermediate
or final outputs used in the boundary RD-in-panel analysis.

Commands
--------
build_parcels : Build parcel base layer
    Output: data_work/parcels_sfr.gpkg
build_treatments : Assign treatment/exposure indicators
    Output: data_work/parcel_treatments.parquet
ingest_sales : Load raw sales data
link_sales : Link sales to parcels
label_parties : Classify owner types
exposure_features : Calculate boundary exposure
clean_sales : Apply arms-length filters
buyer_features : Calculate buyer proximity
build_panels : Create parcel-month panels
boundary_from_gdb : Extract boundaries from geodatabase
event_study : Run event study estimation
rd_windows : Define RD caliper windows
boundary_prepare : Prepare boundary geometries
boundary_chunk : Calculate distances (chunked)
boundary_merge : Merge distance calculations
rd_summary : Run main RD DiD estimation
rd_diagnostics : Run RD identification diagnostics
spatial_econometrics : Run spatial econometrics (Conley SE, SAR, SEM)
placebo_tests : Run placebo and falsification tests
selection_correction : Run selection correction models (Heckman, IPW, Lee)
quantile_effects : Run quantile treatment effects analysis
extended_horizon : Run extended horizon persistence analysis
mechanism_analysis : Run mechanism investigation (insurance, credit, info)
run_all_diagnostics : Run complete diagnostics suite

Usage
-----
    python src/pipeline.py build_parcels
    python src/pipeline.py build_treatments
    python src/pipeline.py event_study

Notes
-----
Requires activation of project virtual environment before running.
"""
from __future__ import annotations
import os, sys, argparse

def ensure_env():
    if not os.getenv('VIRTUAL_ENV') or not os.getenv('VIRTUAL_ENV').endswith('/.venv'):
        print('ERROR: Please activate project .venv (source .venv/bin/activate) before running.', file=sys.stderr)
        sys.exit(1)

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Flood Project Pipeline')
    sub = p.add_subparsers(dest='cmd', required=True)
    sub.add_parser('build_parcels')
    sub.add_parser('build_treatments')
    sub.add_parser('ingest_sales')
    sub.add_parser('link_sales')
    sub.add_parser('label_parties')
    sub.add_parser('exposure_features')
    sub.add_parser('clean_sales')
    sub.add_parser('buyer_features')
    sub.add_parser('build_panels')
    sub.add_parser('boundary_from_gdb')
    sub.add_parser('event_study')
    sub.add_parser('rd_windows')
    sub.add_parser('boundary_prepare')
    pchunk = sub.add_parser('boundary_chunk')
    pchunk.add_argument('--chunk-index', type=int, required=True)
    pchunk.add_argument('--chunk-total', type=int, required=True)
    sub.add_parser('boundary_merge')

    # RD Summary with options
    p_rd = sub.add_parser('rd_summary')
    p_rd.add_argument('--boundary', '-b', nargs='+', default=['inund'],
                      help='Boundary types (default: inund)')
    p_rd.add_argument('--caliper', '-c', nargs='+', type=int, default=[300],
                      help='Caliper distances in meters (default: 300)')
    p_rd.add_argument('--spatial-se', action='store_true',
                      help='Use Conley spatial standard errors')
    p_rd.add_argument('--conley-cutoff', type=float, default=5.0,
                      help='Conley SE cutoff in km (default: 5.0)')

    # RD Diagnostics
    sub.add_parser('rd_diagnostics')

    # Spatial Econometrics
    p_spatial = sub.add_parser('spatial_econometrics')
    p_spatial.add_argument('--boundary', '-b', default='inund',
                           help='Boundary type (default: inund)')
    p_spatial.add_argument('--caliper', '-c', type=int, default=300,
                           help='Caliper in meters (default: 300)')

    # Placebo Tests
    p_placebo = sub.add_parser('placebo_tests')
    p_placebo.add_argument('--boundary', '-b', default='inund',
                           help='Boundary type (default: inund)')
    p_placebo.add_argument('--caliper', '-c', type=int, default=300,
                           help='Caliper in meters (default: 300)')
    p_placebo.add_argument('--n-permutations', '-n', type=int, default=500,
                           help='Number of permutations (default: 500)')

    # Selection Correction
    p_select = sub.add_parser('selection_correction')
    p_select.add_argument('--boundary', '-b', default='inund',
                          help='Boundary type (default: inund)')
    p_select.add_argument('--caliper', '-c', type=int, default=300,
                          help='Caliper in meters (default: 300)')

    # Quantile Effects
    p_quant = sub.add_parser('quantile_effects')
    p_quant.add_argument('--boundary', '-b', default='inund',
                         help='Boundary type (default: inund)')
    p_quant.add_argument('--caliper', '-c', type=int, default=300,
                         help='Caliper in meters (default: 300)')

    # Extended Horizon
    p_horizon = sub.add_parser('extended_horizon')
    p_horizon.add_argument('--boundary', '-b', default='inund',
                           help='Boundary type (default: inund)')
    p_horizon.add_argument('--caliper', '-c', type=int, default=300,
                           help='Caliper in meters (default: 300)')

    # Mechanism Analysis
    p_mech = sub.add_parser('mechanism_analysis')
    p_mech.add_argument('--boundary', '-b', default='inund',
                        help='Boundary type (default: inund)')
    p_mech.add_argument('--caliper', '-c', type=int, default=300,
                        help='Caliper in meters (default: 300)')

    # Run All Diagnostics
    p_all = sub.add_parser('run_all_diagnostics')
    p_all.add_argument('--boundary', '-b', default='inund',
                       help='Primary boundary (default: inund)')
    p_all.add_argument('--caliper', '-c', type=int, default=300,
                       help='Caliper in meters (default: 300)')
    p_all.add_argument('--skip-permutation', action='store_true',
                       help='Skip slow permutation tests')

    return p.parse_args()

def main():
    ensure_env()
    args = parse_args()
    if args.cmd == 'build_parcels':
        # Import via relative path
        from step_impl import parcels as _parcels
        _parcels.main()
    elif args.cmd == 'build_treatments':
        from step_impl import treatments as _treat
        _treat.main()
    elif args.cmd == 'ingest_sales':
        from step_impl import __init__ as _pkg
        import importlib
        mod = importlib.import_module('00_ingest.sales_ingest')
        mod.main()
    elif args.cmd == 'link_sales':
        import importlib
        mod = importlib.import_module('01_link.link_sales_to_parcels')
        mod.main()
    elif args.cmd == 'label_parties':
        import importlib
        mod = importlib.import_module('02_labels.label_parties')
        mod.main()
    elif args.cmd == 'exposure_features':
        import importlib
        mod = importlib.import_module('03_exposure.boundary_features')
        mod.main()
    elif args.cmd == 'clean_sales':
        import importlib
        mod = importlib.import_module('04_salesclean.clean_sales')
        mod.main()
    elif args.cmd == 'buyer_features':
        import importlib
        mod = importlib.import_module('05_features.buyer_proximity')
        mod.main()
    elif args.cmd == 'build_panels':
        import importlib
        mod = importlib.import_module('06_panels.build_panels')
        mod.main()
    elif args.cmd == 'event_study':
        import importlib
        mod = importlib.import_module('07_estimation.event_study')
        mod.main()
    elif args.cmd == 'boundary_from_gdb':
        import importlib
        mod = importlib.import_module('03_exposure.boundary_from_gdb')
        mod.main()
    elif args.cmd == 'rd_windows':
        import importlib
        mod = importlib.import_module('03_exposure.rd_windows')
        mod.main()
    elif args.cmd == 'boundary_prepare':
        import importlib
        mod = importlib.import_module('03_exposure.boundary_prepare')
        mod.main()
    elif args.cmd == 'boundary_chunk':
        import importlib
        mod = importlib.import_module('03_exposure.boundary_chunk_dist')
        mod.main()
    elif args.cmd == 'boundary_merge':
        import importlib
        mod = importlib.import_module('03_exposure.boundary_merge')
        mod.main()
    elif args.cmd == 'rd_summary':
        import importlib
        mod = importlib.import_module('07_estimation.rd_summary')
        mod.main(
            boundaries=args.boundary,
            calipers=args.caliper,
            spatial_se=args.spatial_se,
            conley_cutoff_km=args.conley_cutoff
        )
    elif args.cmd == 'rd_diagnostics':
        import importlib
        mod = importlib.import_module('07_estimation.rd_diagnostics')
        mod.main()
    elif args.cmd == 'spatial_econometrics':
        import importlib
        mod = importlib.import_module('07_estimation.spatial_econometrics')
        mod.main(boundary=args.boundary, caliper=args.caliper)
    elif args.cmd == 'placebo_tests':
        import importlib
        mod = importlib.import_module('07_estimation.placebo_tests')
        mod.main(
            boundary=args.boundary,
            caliper=args.caliper,
            n_permutations=args.n_permutations
        )
    elif args.cmd == 'selection_correction':
        import importlib
        mod = importlib.import_module('07_estimation.selection_correction')
        mod.main(boundary=args.boundary, caliper=args.caliper)
    elif args.cmd == 'quantile_effects':
        import importlib
        mod = importlib.import_module('07_estimation.quantile_effects')
        mod.main(boundary=args.boundary, caliper=args.caliper)
    elif args.cmd == 'extended_horizon':
        import importlib
        mod = importlib.import_module('07_estimation.extended_horizon')
        mod.main(boundary=args.boundary, caliper=args.caliper)
    elif args.cmd == 'mechanism_analysis':
        import importlib
        mod = importlib.import_module('07_estimation.mechanism_analysis')
        mod.main(boundary=args.boundary, caliper=args.caliper)
    elif args.cmd == 'run_all_diagnostics':
        import importlib
        mod = importlib.import_module('07_estimation.run_all_diagnostics')
        mod.run_all(
            boundary=args.boundary,
            caliper=args.caliper,
            skip_permutation=args.skip_permutation
        )

if __name__ == '__main__':
    main()
