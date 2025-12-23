#!/usr/bin/env python3
"""
Module: boundary_from_gdb.py
Purpose: Extract SFHA and inundation boundaries and calculate signed distances.

This module reads parcel and hazard data directly from an ArcGIS geodatabase,
calculates signed Euclidean distances from parcel centroids to SFHA and 2019
inundation boundaries (negative = inside, positive = outside).

Input Files
-----------
- OwnerDistanceProject.gdb (Parcels, FIRM, InundationPolygon layers)

Output Files
------------
- data_work/parcel_boundary_distances.parquet

Functions
---------
ensure_env : Verify virtual environment is activated
main : Execute distance calculation pipeline
"""
from __future__ import annotations
import os, sys
from pathlib import Path
import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.ops import unary_union

GDB = Path('/Users/jesseandrews/Documents/ArcGIS/Projects/OwnerDistanceProject/OwnerDistanceProject.gdb')
OUT = Path('data_work/parcel_boundary_distances.parquet')

def ensure_env():
    if not os.getenv('VIRTUAL_ENV') or not os.getenv('VIRTUAL_ENV').endswith('/.venv'):
        print('ERROR: Please activate project .venv (source .venv/bin/activate) before running.', file=sys.stderr)
        sys.exit(1)

def main():
    """
    Execute boundary distance calculation from geodatabase.

    Reads parcels and hazard layers, calculates signed distances from each
    parcel centroid to SFHA and inundation boundaries. Distance is negative
    for parcels inside the hazard zone, positive for outside.

    Raises
    ------
    SystemExit
        If geodatabase not found or SFHA features empty after filtering.
    """
    ensure_env()
    if not GDB.exists():
        raise SystemExit(f'GDB not found: {GDB}')

    parcels = gpd.read_file(GDB, layer='Parcels')
    firm = gpd.read_file(GDB, layer='FIRM')
    inund = gpd.read_file(GDB, layer='InundationPolygon')

    target_epsg = 32615
    parcels = parcels.to_crs(epsg=target_epsg)
    firm = firm.to_crs(epsg=target_epsg)
    inund = inund.to_crs(epsg=target_epsg)

    pcent = parcels[['Parcel_ID']].copy()
    pcent['geometry'] = parcels.geometry.representative_point()
    pcent = gpd.GeoDataFrame(pcent[['Parcel_ID','geometry']], geometry='geometry', crs=f'EPSG:{target_epsg}')
    pcent['parcel_id'] = pcent['Parcel_ID'].astype(str)

    sfha = firm.copy()
    if 'SFHA_TF' in sfha.columns:
        sfha = sfha[sfha['SFHA_TF'].astype(str).str.upper().isin(['T','Y','1'])]
    elif 'FLD_ZONE' in sfha.columns:
        sfha = sfha[sfha['FLD_ZONE'].astype(str).str.upper().str.startswith(tuple(['A','V']))]
    if sfha.empty:
        raise SystemExit('FIRM layer had no SFHA features after filtering')
    sfha_union = unary_union(sfha.geometry)
    # Boundary (simplified for speed when used)
    sfha_boundary = sfha_union.boundary.simplify(2.0, preserve_topology=True)

    inund_union = unary_union(inund.geometry)
    inund_boundary = inund_union.boundary.simplify(2.0, preserve_topology=True)

    def dist_to_boundary(pt, boundary):
        try:
            return float(pt.distance(boundary))
        except Exception:
            return np.nan

    # Distance to polygon: zero if inside; boundary distance if outside
    dist_to_sfha_poly = pcent.geometry.distance(sfha_union)
    inside_sfha = pcent.geometry.within(sfha_union)
    # Start with polygon distance; fix for inside points using boundary distance
    pcent['dist_to_sfha_edge_m'] = dist_to_sfha_poly.astype(float)
    inside_idx = np.where(inside_sfha.values)[0]
    if inside_idx.size > 0:
        # compute only for inside points
        vals = []
        for i in inside_idx:
            vals.append(dist_to_boundary(pcent.geometry.iloc[i], sfha_boundary))
        pcent.loc[pd.Index(pcent.index[inside_idx]), 'dist_to_sfha_edge_m'] = vals
    pcent['inside_sfha'] = inside_sfha.astype(int)
    pcent['signed_dist_sfha_m'] = np.where(pcent['inside_sfha']==1, -pcent['dist_to_sfha_edge_m'], pcent['dist_to_sfha_edge_m'])

    dist_to_inund_poly = pcent.geometry.distance(inund_union)
    inside_inund = pcent.geometry.within(inund_union)
    pcent['dist_to_inund_edge_m'] = dist_to_inund_poly.astype(float)
    in_inund_idx = np.where(inside_inund.values)[0]
    if in_inund_idx.size > 0:
        vals = []
        for i in in_inund_idx:
            vals.append(dist_to_boundary(pcent.geometry.iloc[i], inund_boundary))
        pcent.loc[pd.Index(pcent.index[in_inund_idx]), 'dist_to_inund_edge_m'] = vals
    pcent['inside_inund'] = inside_inund.astype(int)
    pcent['signed_dist_inund_m'] = np.where(pcent['inside_inund']==1, -pcent['dist_to_inund_edge_m'], pcent['dist_to_inund_edge_m'])

    out = pcent.drop(columns=['geometry','Parcel_ID'])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT, index=False)
    print('Wrote', OUT)

if __name__ == '__main__':
    main()
