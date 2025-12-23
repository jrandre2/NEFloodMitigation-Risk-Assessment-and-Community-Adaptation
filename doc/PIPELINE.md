# Analysis Pipeline Documentation

This document describes the data processing and analysis pipeline for the Freeze and Flight project.

---

## Pipeline Overview

The pipeline consists of 9 stages that transform raw data into publication-ready results:

```
[Pre-pipeline: build_parcels, build_treatments]
    ↓
00_ingest → 01_link → 02_labels → 03_exposure → 04_salesclean → 05_features → 06_panels → 07_estimation → 08_figures
```

All commands are run via `python src/pipeline.py <command>`.

---

## Pre-Pipeline Steps

### Build Parcels (`step_impl/parcels.py`)

**Command**: `python src/pipeline.py build_parcels`

**Purpose**: Build single-family residential (SFR) parcel base layer with geometries.

**Input**:
- `results/integration_run/parcels_with_classification.csv`
- `results/integration_run/sfr_regression_data.csv`

**Output**:
- `data_work/parcels_sfr.gpkg` - GeoPackage with parcel point geometries (EPSG:4326)

**Key Operations**:
- Filter to SFR parcels (Property_P_x == 1)
- Extract parcel IDs, coordinates, neighborhood, and log-transformed attributes
- Create SFHA and inundation indicators from source data
- Generate point geometries from parcel centroids

---

### Build Treatments (`step_impl/treatments.py`)

**Command**: `python src/pipeline.py build_treatments`

**Purpose**: Assign treatment and exposure indicators to parcels.

**Input**:
- `results/integration_run/sfr_regression_data.csv`

**Output**:
- `data_work/parcel_treatments.parquet`

**Key Variables Created**:
- `sfha_majority` - SFHA status by majority-area rule
- `sfha_10pct` - SFHA status by 10% overlap rule
- `inund_201903` - Actual 2019 inundation status
- `x`, `y` - Parcel coordinates

---

## Stage Details

### Stage 00: Sales Ingestion (`00_ingest/`)

**Script**: `sales_ingest.py`

**Purpose**: Load and preprocess raw sales transaction data from county assessor files.

**Input**:
- Raw sales CSV from Douglas County Register of Deeds

**Output**:
- Cleaned sales DataFrame with standardized fields

**Key Operations**:
- Parse transaction dates
- Standardize party name fields
- Filter to study period (±24 months from March 2019)

---

### Stage 01: Parcel Linkage (`01_link/`)

**Script**: `link_sales_to_parcels.py`

**Purpose**: Link sales transactions to parcel geometries using APN/legal description matching.

**Input**:
- Cleaned sales data
- Parcel GIS layer (Douglas County Assessor)

**Output**:
- Sales joined to parcel attributes and geometries

**Key Operations**:
- Match on APN (Assessor Parcel Number)
- Flag unmatched/split/merged parcels
- Attach parcel attributes (land use, building age, assessed value)

---

### Stage 02: Party Labeling (`02_labels/`)

**Script**: `label_parties.py`

**Purpose**: Classify buyer and seller organizational form using ML-based owner classification.

**Input**:
- Sales with party name strings

**Output**:
- Sales with owner type labels (Individual, LLC, Corporation, Trust, Other)

**Key Operations**:
- Apply fine-tuned BERT classifier to owner name strings
- Derive portfolio scale (single-parcel vs. multi-parcel)
- Snapshot-based buyer derivation for transaction context

---

### Stage 03: Exposure Calculation (`03_exposure/`)

**Scripts**:
- `boundary_prepare.py` - Prepare SFHA and inundation boundaries
- `boundary_from_gdb.py` - Extract boundaries from geodatabase
- `boundary_chunk_dist.py` - Calculate signed distances (chunked processing)
- `boundary_merge.py` - Merge distance calculations
- `boundary_features.py` - Create exposure features
- `rd_windows.py` - Define RD caliper windows
- `run_boundary_all.py` - Orchestrate boundary processing

**Purpose**: Calculate parcel-level exposure to SFHA boundaries and 2019 inundation.

**Input**:
- Parcel centroids
- FEMA SFHA polygons (zones A/AE/AO/AH)
- 2019 inundation boundary (Sentinel-2 derived)

**Output**:
- `parcel_boundary_distances.parquet` - Signed distances for all parcels
- `sfha_boundary.gpkg` - SFHA boundary line
- `inund_boundary.gpkg` - Inundation boundary line

**Key Variables**:
- `signed_dist_sfha_m` - Signed Euclidean distance to SFHA boundary (negative = inside)
- `signed_dist_inund_m` - Signed distance to 2019 inundation edge
- `inside_sfha` - Binary indicator for SFHA location

**CLI Commands for Boundary Processing**:

```bash
# Option 1: All-in-one (for small datasets)
python src/pipeline.py boundary_from_gdb

# Option 2: Chunked processing (for large datasets)
python src/pipeline.py boundary_prepare                      # Prepare geometries
python src/pipeline.py boundary_chunk --chunk-index 0 --chunk-total 4  # Process chunk 0 of 4
python src/pipeline.py boundary_chunk --chunk-index 1 --chunk-total 4  # Process chunk 1 of 4
python src/pipeline.py boundary_chunk --chunk-index 2 --chunk-total 4  # Process chunk 2 of 4
python src/pipeline.py boundary_chunk --chunk-index 3 --chunk-total 4  # Process chunk 3 of 4
python src/pipeline.py boundary_merge                        # Combine chunks

# Create RD windows and rings
python src/pipeline.py rd_windows
```

**Note**: The boundary scripts reference a hardcoded geodatabase path that must be modified for your environment. See [Configuration & Environment](#configuration--environment) section.

---

### Stage 04: Sales Cleaning (`04_salesclean/`)

**Script**: `clean_sales.py`

**Purpose**: Apply arms-length transaction filters.

**Input**:
- Linked sales with owner labels

**Output**:
- Clean sales restricted to market transactions

**Filters Applied**:
- Remove nominal-consideration transfers (< $1,000)
- Exclude intra-family gifts
- Exclude sheriff/foreclosure deeds
- Exclude intra-entity cleanups

---

### Stage 05: Feature Engineering (`05_features/`)

**Scripts**:
- `buyer_proximity.py` - Calculate buyer-parcel proximity (main)
- `process_dem.py` - Extract elevation from DEM tiles
- `process_building_footprints.py` - Process Microsoft building footprints
- `extract_assessor.py` - Extract assessor data from GDB
- `transform_assessor.py` - Transform assessor data
- `validate_assessor.py` - Validate assessor data
- `integrate_assessor.py` - Integrate assessor data into pipeline

#### buyer_proximity.py (Main)

**Command**: `python src/pipeline.py buyer_features`

**Purpose**: Calculate buyer-parcel proximity features.

**Input**:
- `data_work/sales_clean.parquet`

**Output**:
- `data_work/sales_buyer_features.parquet`

**Key Variables**:
- `buyer_local` - Binary indicator for same-ZIP ownership
- `buyer_zip_band` - Distance band category (same_zip vs other)

---

#### process_dem.py

**Purpose**: Extract elevation values at parcel centroids from USGS DEM tiles.

**Input**:
- `GIS_Data/Elevation/USGS_3DEP_10m/*.tif` - USGS 3DEP DEM tiles
- `data_work/parcel_boundary_distances.parquet` - Parcel centroids

**Output**:
- `data_work/parcel_elevation.parquet`

**Key Variables**:
- `elevation_m` - Elevation at parcel centroid (meters)

**Usage**: `python src/05_features/process_dem.py`

---

#### process_building_footprints.py

**Purpose**: Calculate building footprint area per parcel from Microsoft footprints.

**Input**:
- `GIS_Data/Building_Footprints/Nebraska.geojson` - Microsoft building footprints
- `data_work/parcel_boundary_distances.parquet` - Parcel boundaries

**Output**:
- `data_work/parcel_building_footprints.parquet`

**Key Variables**:
- `footprint_sqm` - Total building footprint area in square meters

**Usage**: `python src/05_features/process_building_footprints.py`

---

#### Stage 05b: Assessor Data ETL Pipeline

The assessor scripts extract, transform, validate, and integrate parcel attributes from the Nebraska statewide assessor geodatabase. This pipeline enables covariate balance tests and covariate-adjusted RD estimation.

**Run Order**:
```bash
# Run in sequence (each step depends on previous)
python src/05_features/extract_assessor.py      # ~7 sec
python src/05_features/transform_assessor.py    # ~5 sec
python src/05_features/validate_assessor.py     # ~3 sec
python src/05_features/integrate_assessor.py    # ~10 sec
```

**Step 1: extract_assessor.py**

Extracts Douglas County (County_ID = '055') parcels from the statewide geodatabase.

| | |
|---|---|
| **Input** | `statewide parcel/NE_2023_statewideparcels.gdb` (735 MB) |
| **Output** | `data_work/assessor_raw.parquet` (51 MB, 212,314 parcels) |
| **Method** | Uses ogr2ogr for extraction, geopandas for parquet conversion |
| **CRS** | Reprojects from NAD83/Nebraska ftUS to WGS84 (EPSG:4326) |

**Key Fields Extracted**:
- `Parcel_ID`, `BuildingYear`, `ImpSF`, `QualImp`, `CondImp` - Building characteristics
- `Total_Assessed_Value`, `Land_Value`, `Improvements_Value` - Values
- `GIS_Acres`, `Property_Parcel_Type`, `Zoning`, `Neighborhood` - Parcel info
- `Current_Owner_Name`, `Ownership_Type` - Owner info

**Step 2: transform_assessor.py**

Cleans and derives analysis-ready features from raw assessor data.

| | |
|---|---|
| **Input** | `data_work/assessor_raw.parquet` |
| **Output** | `data_work/assessor_clean.parquet` (34 MB) |

**Transformations Applied**:
- Parse `BuildingYear` (string) → `year_built` (int) with multi-format handling
- Compute `building_age` = 2019 - year_built
- Create log transforms: `log_assessed_value`, `log_land_value`, `log_acres`, `log_impsf`
- Map `Property_Parcel_Type` codes to labels (1=SFR, 2=Multi-Family, etc.)
- Derive `is_sfr` and `is_improved` binary indicators

**Coverage Statistics**:
- year_built: 88.1% coverage (186,974 / 212,314)
- Total_Assessed_Value: 100% coverage
- SFR parcels: 170,143 (80.2%)

**Step 3: validate_assessor.py**

Validates data quality and match rates against existing project parcel data.

| | |
|---|---|
| **Input** | `data_work/assessor_clean.parquet`, `data_work/parcel_boundary_distances.parquet` |
| **Output** | `data_work/diagnostics/assessor_match_report.csv` |

**Validation Checks**:
- Parcel ID match rate: 100% (212,312 / 212,312 matched)
- SFR subset coverage verification
- RD window (±300m) parcel coverage

**Step 4: integrate_assessor.py**

Merges 17 assessor covariates into analysis datasets.

| | |
|---|---|
| **Input** | `data_work/assessor_clean.parquet`, `data_work/parcel_boundary_distances.parquet`, `data_work/panel_parcel_month.parquet` |
| **Output** | `data_work/parcel_covariates_full.parquet` (17 MB), `data_work/panel_parcel_month_enriched.parquet` (5.4 MB) |

**Covariates Integrated** (17 variables):
`year_built`, `building_age`, `Total_Assessed_Value`, `log_assessed_value`, `Land_Value`, `log_land_value`, `Improvements_Value`, `log_improvement_value`, `GIS_Acres`, `log_acres`, `ImpSF`, `log_impsf`, `is_sfr`, `is_improved`, `neighborhood`, `property_type_label`, `zoning_label`

---

### Stage 06: Panel Construction (`06_panels/`)

**Script**: `build_panels.py`

**Purpose**: Build parcel-month panel for event study and RD analysis.

**Input**:
- All processed sales and features
- Parcel exposure measures

**Output**:
- `panel_parcel_month.parquet` - Balanced parcel×month panel
- `panel_micro_counts.parquet` - Microcell monthly sales counts

**Panel Structure**:
- Event time: t ∈ [-24, +24] months relative to March 2019
- Unit: parcel-month (for sale incidence) or microcell-month (for counts)

---

### Stage 07: Estimation (`07_estimation/`)

**Scripts**:
- `event_study.py` - Dynamic treatment effects by event time
- `rd_summary.py` - Boundary RD summary statistics
- `rd_summary_by_group.py` - RD by owner type/distance subgroups
- `poisson_ring_models.py` - Near-but-dry ring count models
- `rd_diagnostics.py` - RD identification diagnostics
- `buyer_composition_did.py` - Buyer composition DiD analysis
- `covariate_balance.py` - Covariate balance tests at boundary
- `spatial_econometrics.py` - Spatial lag/error models, Conley SEs
- `placebo_tests.py` - Falsification and placebo tests
- `selection_correction.py` - Heckman, IPW, Lee bounds
- `quantile_effects.py` - Quantile treatment effects
- `extended_horizon.py` - Extended time horizon analysis
- `mechanism_analysis.py` - Mechanism investigation
- `run_all_diagnostics.py` - Orchestrate all diagnostics

**Purpose**: Estimate treatment effects at SFHA and inundation boundaries.

**Models**:

1. **Event Study** (Dynamic DiD):
   ```
   Y_it = Σ β_τ [Inside_i × 1{t=τ}] + α_i + γ_nt + ε_it
   ```

2. **Boundary RD-in-Panel**:
   - Sale rate DiD within ±150m and ±300m windows
   - Log-price DiD for transactions within windows

3. **Poisson Ring Models**:
   ```
   log E[y_ct] = α_c + δ_t + θ_1[Ring_0-250 × Post] + θ_2[Ring_250-300 × Post]
   ```

**Output**:
- `event_study_summary.parquet` - Event study coefficients
- `tab_rd_sfha.csv` - SFHA boundary RD results
- `tab_rd_inund.csv` - Inundation boundary RD results

---

#### Stage 07b: Identification Diagnostics

This subsection documents the comprehensive identification diagnostics that validate the RD design assumptions.

**Run Order**:
```bash
# Run diagnostics (after panel construction)
python src/07_estimation/rd_diagnostics.py
python src/07_estimation/covariate_balance.py
python src/07_estimation/buyer_composition_did.py
```

---

##### rd_diagnostics.py

**Purpose**: Implement comprehensive RD identification diagnostics for manuscript validation.

**Input**:
- `data_work/panel_parcel_month.parquet`
- `data_work/parcel_boundary_distances.parquet`

**Output**:
- `data_work/diagnostics/mccrary_density_test.csv`
- `data_work/diagnostics/pretrends_ftest.csv`
- `data_work/diagnostics/bandwidth_sensitivity.csv`
- `data_work/diagnostics/donut_rd.csv`
- `figures/fig_mccrary_*.png`, `figures/fig_bandwidth_*.png`

**Diagnostic Tests Implemented**:

| Test | Purpose | Key Result |
|------|---------|------------|
| McCrary Density | Tests for manipulation at boundary | SFHA: z=7.78, p<0.001 (significant discontinuity) |
| Pre-Trends F-Test | Tests parallel trends assumption | SFHA: F=3.33, p<0.001 (violated); Inund: passes |
| Bandwidth Sensitivity | Robustness across calipers | SFHA DiD stable 150-400m; Inund unstable |
| Donut RD | Addresses SUTVA/spillover concerns | Tests at 50m-250m exclusion widths |

**Usage**: `python src/07_estimation/rd_diagnostics.py`

**Reference**: McCrary (2008) for density test methodology.

---

##### covariate_balance.py

**Purpose**: Test whether parcels inside vs. outside SFHA boundary are balanced on pre-treatment covariates.

**Input**:
- `data_work/parcel_covariates_full.parquet` (from Stage 05b)

**Output**:
- `data_work/diagnostics/covariate_balance.csv` (all parcels)
- `data_work/diagnostics/covariate_balance_sfr.csv` (SFR improved only)
- `figures/fig_covariate_balance.png`

**Methodology**:
For each covariate, estimate: `Y_i = α + β·inside_sfha_i + ε_i`

Tests conducted at three calipers: ±100m, ±200m, ±300m.

**Key Results (SFR Improved, ±300m)**:

| Covariate | Inside | Outside | p-value | Balanced? |
|-----------|--------|---------|---------|-----------|
| Year Built | 1963 | 1980 | <0.001 | No |
| Assessed Value | $180k | $246k | <0.001 | No |
| Lot Size | 1.02 ac | 0.31 ac | <0.001 | No |
| Building SF | 1,445 | 1,699 | <0.001 | No |

**Summary**: 0/8 covariates balanced at 200m and 300m calipers for SFR improved parcels.

**Usage**: `python src/07_estimation/covariate_balance.py`

---

##### buyer_composition_did.py

**Purpose**: Analyze whether post-flood purchases shift toward different buyer types (LLCs, portfolio investors).

**Input**: `data_work/panel_parcel_month.parquet`

**Output**:
- `data_work/diagnostics/buyer_composition_did.csv`
- `data_work/diagnostics/buyer_composition_summary.csv`
- `figures/fig_buyer_composition_event_study.png`

**Key Finding**: LLC share DECREASED inside SFHA post-flood (from 45% to 27%), contradicting the "investor acquisition" hypothesis.

**Usage**: `python src/07_estimation/buyer_composition_did.py`

---

#### Stage 07c: Robustness and Extension Modules

These modules implement comprehensive robustness checks and methodological extensions for peer-review validation.

**Run All Diagnostics (Recommended)**:
```bash
python src/pipeline.py run_all_diagnostics --boundary inund --caliper 300
```

Or run individual modules:

```bash
# Spatial econometrics (Conley SEs, SAR, SEM)
python src/pipeline.py spatial_econometrics -b inund -c 300

# Placebo and falsification tests
python src/pipeline.py placebo_tests -b inund -c 300 -n 500

# Selection correction (Heckman, IPW, Lee bounds)
python src/pipeline.py selection_correction -b inund -c 300

# Quantile treatment effects
python src/pipeline.py quantile_effects -b inund -c 300

# Extended horizon persistence analysis
python src/pipeline.py extended_horizon -b inund -c 300

# Mechanism analysis
python src/pipeline.py mechanism_analysis -b inund -c 300
```

---

##### spatial_econometrics.py

**Purpose**: Address spatial autocorrelation with full spatial econometrics suite.

**Input**:
- `data_work/panel_parcel_month.parquet`

**Output**:
- `data_work/diagnostics/conley_se_comparison.csv`
- `data_work/diagnostics/spatial_lag_results.csv`
- `data_work/diagnostics/spatial_error_results.csv`
- `data_work/diagnostics/moran_residual_test.csv`

**Methods Implemented**:

| Method | Description |
|--------|-------------|
| Conley SEs | Spatial HAC standard errors (configurable cutoff, default 5km) |
| Spatial Lag (SAR) | y = ρWy + Xβ + ε |
| Spatial Error (SEM) | y = Xβ + u, where u = λWu + ε |
| Moran's I | Residual spatial autocorrelation test |

**Dependencies**: `libpysal`, `spreg`, `esda`

**Usage**: `python src/pipeline.py spatial_econometrics -b inund -c 300`

---

##### placebo_tests.py

**Purpose**: Falsification tests to validate causal identification.

**Input**:
- `data_work/panel_parcel_month.parquet`

**Output**:
- `data_work/diagnostics/placebo_event_dates.csv`
- `data_work/diagnostics/placebo_boundaries.csv`
- `data_work/diagnostics/permutation_pvalues.csv`
- `data_work/diagnostics/leave_one_out.csv`
- `data_work/diagnostics/triple_difference.csv`
- `figures/fig_placebo_*.png`
- `figures/fig_permutation_distribution.png`

**Tests Implemented**:

| Test | Description |
|------|-------------|
| Placebo Event Dates | Test DiD at fake dates (2017-03, 2018-03, 2020-03, 2021-03) |
| Placebo Boundaries | Shift boundary by ±50m, ±100m |
| Permutation Inference | Randomization inference (default 500 permutations) |
| Leave-One-Out | Drop each month and re-estimate |
| Triple Difference | Use SFHA as additional control for inundation |

**Usage**: `python src/pipeline.py placebo_tests -b inund -c 300 -n 500`

---

##### selection_correction.py

**Purpose**: Address selection bias in price models (only observe sold properties).

**Input**:
- `data_work/panel_parcel_month.parquet`

**Output**:
- `data_work/diagnostics/heckman_results.csv`
- `data_work/diagnostics/ipw_results.csv`
- `data_work/diagnostics/lee_bounds.csv`
- `data_work/diagnostics/selection_comparison.csv`
- `figures/fig_selection_correction_comparison.png`

**Methods Implemented**:

| Method | Description |
|--------|-------------|
| Heckman Two-Step | Selection model with inverse Mills ratio |
| IPW | Inverse probability weighting |
| Lee Bounds | Lee (2009) trimming bounds under selection |

**Usage**: `python src/pipeline.py selection_correction -b inund -c 300`

---

##### quantile_effects.py

**Purpose**: Estimate distributional effects across the price distribution.

**Input**:
- `data_work/panel_parcel_month.parquet`

**Output**:
- `data_work/diagnostics/quantile_did.csv`
- `data_work/diagnostics/distribution_shift_test.csv`
- `figures/fig_quantile_effects.png`

**Methods Implemented**:

| Method | Description |
|--------|-------------|
| Quantile DiD | DiD at τ = {0.1, 0.25, 0.5, 0.75, 0.9} |
| K-S Test | Kolmogorov-Smirnov distribution shift test |

**Usage**: `python src/pipeline.py quantile_effects -b inund -c 300`

---

##### extended_horizon.py

**Purpose**: Extend analysis beyond ±24 months for persistence testing.

**Input**:
- `data_work/panel_parcel_month.parquet`

**Output**:
- `data_work/diagnostics/dynamic_effects_extended.csv`
- `data_work/diagnostics/persistence_analysis.csv`
- `figures/fig_extended_event_study.png`

**Analysis**:
- Extended event study (±36 months)
- Persistence analysis across time windows (0-6m, 6-12m, 12-18m, 18-24m, 24-36m)

**Usage**: `python src/pipeline.py extended_horizon -b inund -c 300`

---

##### mechanism_analysis.py

**Purpose**: Investigate channels through which flood affects housing market.

**Input**:
- `data_work/panel_parcel_month.parquet`
- `data_work/parcel_covariates_full.parquet`

**Output**:
- `data_work/diagnostics/mechanism_insurance.csv`
- `data_work/diagnostics/mechanism_credit.csv`
- `data_work/diagnostics/heterogeneity_by_chars.csv`
- `data_work/diagnostics/mechanism_buyer_composition.csv`
- `data_work/diagnostics/mechanism_summary.csv`
- `figures/fig_mechanism_heterogeneity.png`

**Mechanisms Tested**:

| Channel | Test |
|---------|------|
| Insurance | Compare SFHA (mandatory) vs inundation (optional insurance) |
| Credit Constraints | Heterogeneity by property value terciles |
| Property Characteristics | Effects by building age, assessed value, lot size |
| Buyer Composition | LLC share, portfolio buyer share changes |

**Usage**: `python src/pipeline.py mechanism_analysis -b inund -c 300`

---

##### run_all_diagnostics.py

**Purpose**: Single entry point to run complete diagnostics suite.

**Output**:
- `data_work/diagnostics/comprehensive_diagnostics_report.md`
- `data_work/diagnostics/diagnostics_summary.json`
- All outputs from individual modules above

**Modules Orchestrated** (8 total):
1. RD Diagnostics (McCrary, pre-trends, bandwidth)
2. Event Study
3. Placebo Tests
4. Spatial Econometrics
5. Selection Correction
6. Quantile Effects
7. Extended Horizon
8. Mechanism Analysis

**Usage**: `python src/pipeline.py run_all_diagnostics -b inund -c 300 [--skip-permutation]`

---

### Stage 08: Figures (`08_figures/`)

**Scripts**:
- `make_figures.py` - Generate publication figures
- `make_rd_report.py` - Create RD summary report
- `sfha_share_figs.py` - SFHA share time series

**Purpose**: Generate publication-quality figures and summary reports.

**Output**:
- `fig_event_study.png` - Event study coefficient plot
- `fig_rd_sfha_sale_rate.png` - SFHA boundary sale rate RD
- `fig_rd_sfha_price.png` - SFHA boundary price RD
- `fig_sfha_share_rd.png` - SFHA share time series
- `report_rd_summary.md` - Consolidated results report

---

## Running the Pipeline

### Full Pipeline

```bash
python src/pipeline.py
```

### Individual Stages

```bash
# Build parcel base layer
python src/pipeline.py build_parcels

# Build treatment assignments
python src/pipeline.py build_treatments

# Run estimation
python src/07_estimation/rd_summary.py
python src/07_estimation/poisson_ring_models.py
```

### Project-Specific Analysis

```bash
# Pre/post sales analysis
python projects/flood_2019_sales_douglas/analyze_pre_post_sales.py

# Spatial Conley standard errors
python projects/flood_2019_sales_douglas/run_spatial_pre_post_conley.py
```

---

## Data Dependencies

| Stage | Requires |
|-------|----------|
| 00 | Raw sales CSV |
| 01 | Parcel shapefile/geopackage |
| 02 | Trained BERT classifier |
| 03 | FEMA NFHL, Sentinel-2 inundation |
| 04-08 | Output from previous stages |

---

## Output Locations

| Output | Location |
|--------|----------|
| Intermediate data | `data_work/` |
| Figures | `figures/` and `data_work/fig_*.png` |
| Results tables | `data_work/tab_*.csv` |
| Reports | `data_work/report_*.md` |
| Diagnostic results | `data_work/diagnostics/` |

---

## Configuration & Environment

### Virtual Environment

All pipeline scripts require activation of the project virtual environment before running:

```bash
source .venv/bin/activate
```

Scripts check for `VIRTUAL_ENV` ending with `/.venv` and will exit with an error if not activated.

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `PANEL_SCOPE` | Panel restriction: `rd_only` restricts to parcels with RD flags | `rd_only` |
| `VIRTUAL_ENV` | Path to activated virtual environment (set automatically) | Required |

### Hardcoded Paths

The following paths are hardcoded in the source code and must be modified for your environment:

| Script | Path | Description |
|--------|------|-------------|
| `03_exposure/boundary_from_gdb.py` | `/Users/jesseandrews/Documents/ArcGIS/Projects/OwnerDistanceProject/OwnerDistanceProject.gdb` | ArcGIS geodatabase with parcel, FIRM, and inundation layers |
| `03_exposure/boundary_prepare.py` | Same as above | Same GDB path |
| `05_features/extract_assessor.py` | `statewide parcel/NE_2023_statewideparcels.gdb` | Nebraska statewide parcel geodatabase |

To modify these paths, edit the `GDB_PATH` constant at the top of each script.

### CRS Conventions

| Data Type | CRS | EPSG Code |
|-----------|-----|-----------|
| Parcel geometries (output) | WGS84 | EPSG:4326 |
| Boundary distance calculations | UTM Zone 15N | EPSG:32615 |
| DEM tiles | NAD83 | Varies by tile |

### Event Window

The analysis uses a fixed event window centered on the March 2019 Missouri River flood:

- **Event month**: March 2019 (event_m = 0)
- **Event window**: March 2017 to March 2021 (±24 months)
- **Pre-period**: event_m ∈ [-24, -1]
- **Post-period**: event_m ∈ [0, +24]
