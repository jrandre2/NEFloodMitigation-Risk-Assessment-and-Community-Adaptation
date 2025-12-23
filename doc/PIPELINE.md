# Analysis Pipeline Documentation

This document describes the data processing and analysis pipeline for the Freeze and Flight project.

---

## Pipeline Overview

The pipeline consists of 9 stages that transform raw data into publication-ready results:

```
00_ingest → 01_link → 02_labels → 03_exposure → 04_salesclean → 05_features → 06_panels → 07_estimation → 08_figures
```

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

**Script**: `buyer_proximity.py`

**Purpose**: Calculate buyer-parcel proximity features.

**Input**:
- Clean sales with buyer addresses
- ZIP code centroid file

**Output**:
- Sales with proximity measures

**Key Variables**:
- `buyer_dist_km` - Great-circle distance from buyer mailing ZIP to parcel
- `local_owner` - Binary indicator for same-ZIP ownership
- Distance bands: same ZIP, other Douglas County ZIP, adjoining county, other NE, other state

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
