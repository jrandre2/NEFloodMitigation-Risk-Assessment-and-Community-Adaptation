#!/usr/bin/env python3
"""
Module: boundary_prepare.py
Purpose: Prepare SFHA and inundation boundary geometries from geodatabase.

This module extracts parcel centroids, SFHA polygons, and 2019 inundation polygons
from an ArcGIS geodatabase. It creates union geometries and boundary lines for
efficient distance calculations in subsequent pipeline stages.

Input Files
-----------
- OwnerDistanceProject.gdb (Parcels, FIRM, InundationPolygon layers)

Output Files
------------
- data_work/parcel_centroids.gpkg
- data_work/sfha_union.gpkg
- data_work/sfha_boundary.gpkg
- data_work/inund_union.gpkg
- data_work/inund_boundary.gpkg

Functions
---------
ensure_env : Verify virtual environment is activated
main : Execute boundary preparation pipeline
"""
from __future__ import annotations
import os, sys
from pathlib import Path
import geopandas as gpd
from shapely.ops import unary_union

GDB = Path('/Users/jesseandrews/Documents/ArcGIS/Projects/OwnerDistanceProject/OwnerDistanceProject.gdb')
OUT_DIR = Path('data_work')

def ensure_env():
    if not os.getenv('VIRTUAL_ENV') or not os.getenv('VIRTUAL_ENV').endswith('/.venv'):
        print('ERROR: Please activate project .venv (source .venv/bin/activate) before running.', file=sys.stderr)
        sys.exit(1)

def main():
    """
    Execute boundary preparation pipeline.

    Reads parcels and hazard layers from geodatabase, creates parcel centroids,
    unions SFHA and inundation polygons, extracts simplified boundary lines,
    and writes all geometries to GeoPackage files.

    Raises
    ------
    SystemExit
        If geodatabase not found or SFHA features empty after filtering.
    """
    ensure_env()
    if not GDB.exists():
        raise SystemExit(f'GDB not found: {GDB}')

    target_epsg = 32615
    # Read layers
    parcels = gpd.read_file(GDB, layer='Parcels').to_crs(epsg=target_epsg)
    firm = gpd.read_file(GDB, layer='FIRM').to_crs(epsg=target_epsg)
    inund = gpd.read_file(GDB, layer='InundationPolygon').to_crs(epsg=target_epsg)

    # Parcel centroids
    cent = parcels[['Parcel_ID']].copy()
    cent['geometry'] = parcels.geometry.representative_point()
    cent = gpd.GeoDataFrame(cent, geometry='geometry', crs=f'EPSG:{target_epsg}')
    cent = cent.rename(columns={'Parcel_ID':'parcel_id'})
    cent['parcel_id'] = cent['parcel_id'].astype(str)

    # SFHA union
    sfha = firm.copy()
    if 'SFHA_TF' in sfha.columns:
        sfha = sfha[sfha['SFHA_TF'].astype(str).str.upper().isin(['T','Y','1'])]
    elif 'FLD_ZONE' in sfha.columns:
        sfha = sfha[sfha['FLD_ZONE'].astype(str).str.upper().str.startswith(tuple(['A','V']))]
    if sfha.empty:
        raise SystemExit('FIRM layer had no SFHA features after filtering')
    sfha_union = unary_union(sfha.geometry)
    sfha_bound = gpd.GeoDataFrame(geometry=[sfha_union.boundary.simplify(2.0, preserve_topology=True)], crs=f'EPSG:{target_epsg}')
    sfha_poly = gpd.GeoDataFrame(geometry=[sfha_union], crs=f'EPSG:{target_epsg}')

    # Inundation union
    inund_union = unary_union(inund.geometry)
    inund_bound = gpd.GeoDataFrame(geometry=[inund_union.boundary.simplify(2.0, preserve_topology=True)], crs=f'EPSG:{target_epsg}')
    inund_poly = gpd.GeoDataFrame(geometry=[inund_union], crs=f'EPSG:{target_epsg}')

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cent.to_file(OUT_DIR / 'parcel_centroids.gpkg', layer='parcel_centroids', driver='GPKG')
    sfha_poly.to_file(OUT_DIR / 'sfha_union.gpkg', layer='sfha_union', driver='GPKG')
    sfha_bound.to_file(OUT_DIR / 'sfha_boundary.gpkg', layer='sfha_boundary', driver='GPKG')
    inund_poly.to_file(OUT_DIR / 'inund_union.gpkg', layer='inund_union', driver='GPKG')
    inund_bound.to_file(OUT_DIR / 'inund_boundary.gpkg', layer='inund_boundary', driver='GPKG')
    print('Prepared centroids and unions/boundaries in data_work/*.gpkg')

if __name__ == '__main__':
    main()

