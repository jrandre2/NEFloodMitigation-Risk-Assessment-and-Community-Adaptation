# Technical Notes — NEFloodMitigation

This document captures dependency requirements, script conventions, and workflow details for
analysts reproducing or extending the geospatial analyses in this repository.

---

## Script Inventory and Dependencies

Scripts live in `Data/Geospatial-Scripts/`. Several have no `.py` extension because they were
uploaded directly to GitHub without one; they are Python source files and can be run as such.

### Dependency tiers

| Dependency | Scripts affected | Notes |
|---|---|---|
| **ArcPy** (ArcGIS Pro ≥ 3.x) | `ACS_FLDZONE_by_Area`, `Bootstrap_Parameter_Testing_Pipeline.py`, `Bootstrap_OmniScript`, `BootstrapMetricsBorderPerturbations`, `BootstrapMetricsScript`, `NFIP_Bootstrap`, `NFIP_Bootstrap_Parameter_Sampling`, `Multithread_LiDAR_to_GeoTIFF` | Requires a licensed ArcGIS Pro installation; cannot be run in a pure open-source Python environment. |
| **Open-source GIS stack** | `ACS_Spatial_Regression_Models.py`, `CenPy_ACS_to_Shapefile.py`, `DEM_to_Points.py`, `Points_to_Tracts.py`, `Owner_distance.py`, `NFIPPolicyDescriptivesBootstrap.py` | Requires `geopandas`, `libpysal`, `spreg`, `esda`, `gdal`, `cenpy`, `statsmodels`, `pandas`, `numpy`. Install via `pip` or `conda`. |
| **Douglas County sub-scripts** | `DouglasOwners/DefiningOwners.py`, `DouglasOwners/Owner-Distance_Single-Family.py`, `DouglasOwners/ResidentialOwnerNameClassification.py` | Uses `pandas`, `fuzzywuzzy`/`rapidfuzz` for name matching; no ArcPy required. |

---

## Bootstrap Spatial Disaggregation Workflow

The core analysis disaggregates FEMA NFIP (National Flood Insurance Program) claims records from
county-level geographic identifiers down to census-tract level using an ensemble bootstrap approach.

1. **Prepare buildings layer** — Load parcel/structure footprints; filter by occupancy type and
   flood zone. Requires an ArcGIS feature class or shapefile of structures.
2. **Load claims** — Read `Data/NE_FEMA_Claims.csv`; match on county FIPS, flood zone code, and
   structure type.
3. **Bootstrap iterations** — For each iteration: sample buildings proportional to their
   elevation/flood-zone stratum; assign claim counts; accumulate iteration statistics.
4. **Output table** — Write per-tract bootstrap count statistics to a geodatabase table or CSV.

Key configuration parameters (set in the config dict at the top of each bootstrap script):
- `n_iterations` — number of bootstrap draws (default 1000)
- `output_gdb` — path to ArcGIS geodatabase for outputs
- `stratify_by` — flood zone codes to use for stratification (`FZ_AE`, `FZ_X`, etc.)

---

## ACS Spatial Regression Workflow

`ACS_Spatial_Regression_Models.py` fits spatial lag and spatial error regression models
(via `spreg`) linking American Community Survey (ACS) socioeconomic variables to NFIP flood
claim rates at the census-tract level.

- Input: a shapefile of Nebraska census tracts with pre-joined ACS and FEMA flood-zone fields
  (path set at the top of the script — update to your local path before running).
- Variables: percent in flood zone (`percent_in`); demographic percentages (Black, Hispanic,
  poverty, disability); median household income; broadband access; vehicle availability.
- Outputs: Moran's I spatial autocorrelation statistic, OLS baseline, ML spatial lag, ML spatial
  error model coefficients printed to console.

---

## LiDAR Processing

`Multithread_LiDAR_to_GeoTIFF` converts LiDAR point-cloud files (.las/.laz) to GeoTIFF
elevation rasters using multithreaded ArcPy processing. Set the input folder path and output
coordinate system in the script header before running.

---

## Property Owner Classification (Douglas County)

Scripts in `Data/Geospatial-Scripts/DouglasOwners/` classify property owners in Douglas County
as absentee investor or owner-occupant using:

- Name-matching heuristics (corporate/LLC name patterns) via `ResidentialOwnerNameClassification.py`
- Distance from registered owner address to parcel via `Owner-Distance_Single-Family.py` and
  `Owner_distance.py` (threshold configurable in script)
- Aggregation logic in `DefiningOwners.py`

---

## Data Notes

- `Data/NE_FEMA_Claims.csv` — ~6,000 records from FEMA's OpenFEMA dataset covering Nebraska
  flood insurance claims. Includes building and contents damage amounts, policy information, and
  geospatial identifiers (county FIPS, census tract, coordinates). Field-level definitions are
  in `Data/FEMA_Claims_Data_Dictionary.JSON`. ACS tract-level variable definitions are in
  `Data/ACS_Tract_Data_Dictionary.json` and `Data/ACS_Variables_Defined.json`.
- Large raster, shapefile, and geodatabase outputs are excluded from version control via
  `.gitignore`. Raw geospatial inputs (shapefiles, GeoTIFFs, .gdb) must be obtained separately.
- The `FEMA_Claims_Data_Themes.svg` diagram provides a visual overview of the thematic groupings
  in the claims dataset.

---

## Environment Setup (open-source stack)

```bash
conda create -n neflood python=3.10
conda activate neflood
conda install -c conda-forge geopandas libpysal esda spreg gdal cenpy statsmodels pandas numpy
pip install rapidfuzz
```

For ArcPy scripts, use the Python environment bundled with ArcGIS Pro (clone it before adding
additional packages).
