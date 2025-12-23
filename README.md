# Freeze and Flight

**Liquidity and Spatial Sorting in Housing Markets After a Major Flood**

This project analyzes housing market responses to the March 2019 Missouri River flood in Douglas County, Nebraska using boundary difference-in-differences (RD-in-panel) designs.

---

## Project Overview

Floods make risk salient, but housing markets can adjust through thin liquidity and sorting rather than clean, precisely estimated price cuts. This study uses a boundary RD-in-panel design at FEMA Special Flood Hazard Area (SFHA) lines and at the 2019 inundation edge to examine:

1. **Sale rates and prices** inside vs. outside the SFHA after the flood
2. **Transaction substitution** to near-but-dry parcels outside hazard boundaries
3. **Buyer composition changes** (organizational form, portfolio scale, proximity)

## Key Findings

- Within ±300m of the SFHA boundary, the inside parcel-month sale rate fell **~38%** relative to outside
- Microcell Poisson models show post-event increases in sales just outside the line (rate ratios ≈ **1.44** in 0-250m)
- Price-level contrasts are negative but imprecise (log-price DiD ≈ -0.26 to -0.35)
- The share of boundary-window sales occurring inside the SFHA falls by **~1.1 percentage points** after flood
- Raw SFHA price discount of **-40%** shrinks to **-15%** after controlling for housing characteristics (covariate-adjusted RD)

---

## Prerequisites

### System Requirements

- Python 3.8 or higher
- ~10GB disk space for data files
- ArcGIS geodatabase access (optional, for boundary processing from scratch)

### Required Data Files

The pipeline expects the following data sources:

| Data | Path | Description |
|------|------|-------------|
| Classification output | `results/integration_run/parcels_with_classification.csv` | Owner classification results |
| Regression data | `results/integration_run/sfr_regression_data.csv` | SFR parcel regression dataset |
| FEMA NFHL | ArcGIS geodatabase | Special Flood Hazard Area boundaries |
| 2019 Inundation | ArcGIS geodatabase | Sentinel-2 derived flood extent |
| NE Statewide Assessor | `statewide parcel/NE_2023_statewideparcels.gdb` | Nebraska assessor parcel data (735 MB) |
| DEM tiles (optional) | `GIS_Data/Elevation/USGS_3DEP_10m/*.tif` | USGS 3DEP 10m elevation |
| Building footprints (optional) | `GIS_Data/Building_Footprints/Nebraska.geojson` | Microsoft building footprints |

### Path Configuration

Some scripts contain hardcoded paths that must be modified for your environment. See [doc/PIPELINE.md#configuration--environment](doc/PIPELINE.md#configuration--environment) for details.

---

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/jesseandrews/freeze-and-flight.git
cd freeze-and-flight

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Pipeline

```bash
# IMPORTANT: Always activate the virtual environment first
source .venv/bin/activate

# Run specific pipeline stages
python src/pipeline.py build_parcels
python src/pipeline.py build_treatments
python src/pipeline.py build_panels
python src/pipeline.py event_study

# Stage 05b: Assessor data ETL (run in order)
python src/05_features/extract_assessor.py
python src/05_features/transform_assessor.py
python src/05_features/validate_assessor.py
python src/05_features/integrate_assessor.py

# Stage 07b: Identification diagnostics
python src/07_estimation/rd_diagnostics.py
python src/07_estimation/covariate_balance.py
python src/07_estimation/buyer_composition_did.py
```

### Chunked Boundary Processing

For large datasets, use chunked parallel processing:

```bash
# Prepare boundary geometries
python src/pipeline.py boundary_prepare

# Process in parallel chunks (run simultaneously)
python src/pipeline.py boundary_chunk --chunk-index 0 --chunk-total 4 &
python src/pipeline.py boundary_chunk --chunk-index 1 --chunk-total 4 &
python src/pipeline.py boundary_chunk --chunk-index 2 --chunk-total 4 &
python src/pipeline.py boundary_chunk --chunk-index 3 --chunk-total 4 &
wait

# Merge results
python src/pipeline.py boundary_merge
```

For detailed pipeline documentation, see [doc/PIPELINE.md](doc/PIPELINE.md).

---

## Project Structure

```text
Freeze and Flight/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── setup.py                     # Package setup
│
├── doc/                         # Documentation
│   ├── PIPELINE.md             # Pipeline stage documentation
│   ├── METHODOLOGY.md          # Statistical methods
│   ├── DATA_DICTIONARY.md      # Variable definitions
│   └── CRITIQUE_FINDINGS.md    # RD diagnostic results and interpretation
│
├── manuscript/                  # Article materials
│   ├── drafts/                 # Main manuscript versions
│   ├── tables/                 # Tables
│   ├── supplementary/          # Supplementary materials
│   └── correspondence/         # Reviewer correspondence
│
├── src/                         # Pipeline code
│   ├── step_impl/              # Pre-pipeline: parcels.py, treatments.py
│   ├── 00_ingest/              # Sales data ingestion
│   ├── 01_link/                # Link sales to parcels
│   ├── 02_labels/              # Label parties (owner types)
│   ├── 03_exposure/            # SFHA/inundation boundary exposure
│   ├── 04_salesclean/          # Clean sales data
│   ├── 05_features/            # Buyer proximity, DEM, footprints, assessor
│   ├── 06_panels/              # Build parcel-month panels
│   ├── 07_estimation/          # Event study, Poisson, RD, diagnostics
│   ├── 08_figures/             # Figure generation
│   └── pipeline.py             # Main pipeline orchestration
│
├── projects/
│   └── flood_2019_sales_douglas/  # Project-specific analysis scripts
│
├── scripts/                     # Spatial modeling scripts
├── legacy/                      # Legacy analysis scripts
├── data_work/                   # Processed data (parquet, gpkg, figures)
│   └── diagnostics/            # RD diagnostic outputs
├── figures/                     # Publication-quality figures
├── GIS_Data/                    # GIS source data
├── related_manuscripts/         # Related Douglas County papers
└── notebooks/                   # Jupyter notebooks
```

---

## Methods

### Primary Design: Boundary RD-in-Panel

For sale incidence:

```
Pr(Sale_it = 1) = Σ β_τ [1{Inside_i} × 1{t=τ}] + α_i + γ_n(i),t + ε_it
```

Windows: ±150m and ±300m from SFHA boundary

### Secondary: Near-but-Dry Ring Models

Poisson regression for microcell sales counts:

```
log E[y_ct] = α_c + δ_t + θ₁[1{Ring 0-250m} × 1{Post}] + θ₂[1{Ring 250-300m} × 1{Post}]
```

For detailed methodology, see [doc/METHODOLOGY.md](doc/METHODOLOGY.md).

---

## Data Sources

- **Parcels**: Douglas County Assessor GIS (polygon geometry, attributes)
- **SFHA**: FEMA National Flood Hazard Layer (zones A/AE/AO/AH)
- **2019 Inundation**: Sentinel-2 derived inundation boundary
- **Sales**: County assessor/register of deeds files
- **Building Footprints**: Microsoft building footprints

---

## Related Papers

1. **Who Owns the Floodplain?** - Organizational form, portfolio scale, and regulatory flood exposure in Douglas County, Nebraska
2. **Owner Proximity and Flood Exposure** - Evidence from Douglas County, Nebraska

---

## Citation

```bibtex
@article{andrews2025freeze,
  title={Freeze and Flight: Liquidity and Spatial Sorting in Housing Markets After a Major Flood},
  author={Andrews, Jesse},
  year={2025},
  journal={Working Paper}
}
```

---

## License

MIT License - see [LICENSE](LICENSE) for details.

---

## Author

Jesse Andrews
Texas Tech University
jesse.andrews@ttu.edu
