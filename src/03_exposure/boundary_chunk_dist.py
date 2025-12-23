#!/usr/bin/env python3
"""
Module: boundary_chunk_dist.py
Purpose: Calculate boundary distances in parallel chunks for large datasets.

This module processes a subset of parcel centroids (chunk i of n total) and
calculates signed distances to SFHA and inundation boundaries. Designed for
parallel execution to handle large parcel datasets efficiently.

Input Files
-----------
- data_work/parcel_centroids.gpkg
- data_work/sfha_union.gpkg
- data_work/sfha_boundary.gpkg
- data_work/inund_union.gpkg
- data_work/inund_boundary.gpkg

Output Files
------------
- data_work/parcel_boundary_distances_part_{i}_of_{n}.parquet

Functions
---------
ensure_env : Verify virtual environment is activated
parse_args : Parse chunk index and total from command line
main : Execute chunked distance calculation
"""
from __future__ import annotations
import os, sys, argparse
from pathlib import Path
import geopandas as gpd
import pandas as pd
import numpy as np

OUT_DIR = Path('data_work')

def ensure_env():
    if not os.getenv('VIRTUAL_ENV') or not os.getenv('VIRTUAL_ENV').endswith('/.venv'):
        print('ERROR: Please activate project .venv (source .venv/bin/activate) before running.', file=sys.stderr)
        sys.exit(1)

def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for chunk processing.

    Returns
    -------
    argparse.Namespace
        Arguments with chunk_index (0-based) and chunk_total.
    """
    p = argparse.ArgumentParser()
    p.add_argument('--chunk-index', type=int, required=True)
    p.add_argument('--chunk-total', type=int, required=True)
    return p.parse_args()

def main():
    """
    Execute chunked boundary distance calculation.

    Selects parcels where index mod total equals chunk_index, calculates
    signed distances to SFHA and inundation boundaries, and writes to
    a chunk-specific parquet file for later merging.

    Raises
    ------
    SystemExit
        If chunk parameters are invalid or input files missing.
    """
    ensure_env()
    args = parse_args()
    idx = args.chunk_index; tot = args.chunk_total
    if tot < 1 or not (0 <= idx < tot):
        raise SystemExit('Invalid chunk parameters')
    cent = gpd.read_file(OUT_DIR / 'parcel_centroids.gpkg', layer='parcel_centroids')
    sfha_poly = gpd.read_file(OUT_DIR / 'sfha_union.gpkg', layer='sfha_union')
    sfha_bound = gpd.read_file(OUT_DIR / 'sfha_boundary.gpkg', layer='sfha_boundary')
    inund_poly = gpd.read_file(OUT_DIR / 'inund_union.gpkg', layer='inund_union')
    inund_bound = gpd.read_file(OUT_DIR / 'inund_boundary.gpkg', layer='inund_boundary')

    cent = cent.reset_index(drop=True)
    sel = cent.index % tot == idx
    part = cent.loc[sel].copy()

    poly_sf = sfha_poly.geometry.iloc[0]
    bnd_sf = sfha_bound.geometry.iloc[0]
    poly_in = inund_poly.geometry.iloc[0]
    bnd_in = inund_bound.geometry.iloc[0]

    inside_sfha = part.geometry.within(poly_sf)
    dist_sf_poly = part.geometry.distance(poly_sf)
    part['dist_to_sfha_edge_m'] = dist_sf_poly.astype(float)
    inside_idx = np.where(inside_sfha.values)[0]
    if inside_idx.size > 0:
        vals = []
        for i in inside_idx:
            vals.append(float(part.geometry.iloc[i].distance(bnd_sf)))
        part.loc[part.index[inside_idx], 'dist_to_sfha_edge_m'] = vals
    part['inside_sfha'] = inside_sfha.astype(int)
    part['signed_dist_sfha_m'] = np.where(part['inside_sfha']==1, -part['dist_to_sfha_edge_m'], part['dist_to_sfha_edge_m'])

    inside_inund = part.geometry.within(poly_in)
    dist_in_poly = part.geometry.distance(poly_in)
    part['dist_to_inund_edge_m'] = dist_in_poly.astype(float)
    in_idx = np.where(inside_inund.values)[0]
    if in_idx.size > 0:
        vals = []
        for i in in_idx:
            vals.append(float(part.geometry.iloc[i].distance(bnd_in)))
        part.loc[part.index[in_idx], 'dist_to_inund_edge_m'] = vals
    part['inside_inund'] = inside_inund.astype(int)
    part['signed_dist_inund_m'] = np.where(part['inside_inund']==1, -part['dist_to_inund_edge_m'], part['dist_to_inund_edge_m'])

    out = part[['parcel_id','dist_to_sfha_edge_m','inside_sfha','signed_dist_sfha_m','dist_to_inund_edge_m','inside_inund','signed_dist_inund_m']].copy()
    out_path = OUT_DIR / f'parcel_boundary_distances_part_{idx+1}_of_{tot}.parquet'
    out.to_parquet(out_path, index=False)
    print('Wrote', out_path, f'({len(out):,} rows)')

if __name__ == '__main__':
    main()

