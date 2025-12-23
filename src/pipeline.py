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

if __name__ == '__main__':
    main()
