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

## Current Analysis Status

**Status: PHASE 5 COMPLETE - "Freeze and Rebuild" Narrative**

Investigation of the counterintuitive positive price effect has been completed. The analysis now supports a "Freeze and Rebuild" narrative rather than "Freeze and Flight":

### Primary Findings

| Finding | Result | Interpretation |
|---------|--------|----------------|
| Inundation pre-trends | **PASSES** (p=0.779) | Supports parallel trends assumption |
| SFHA pre-trends | FAILS (p=0.026) | Use inundation as primary boundary |
| Price effect | **+52.8%** (inundation, no trends) | Positive, not negative |
| Composition effect | ~60% explained by observables | Newer/larger properties sell post-flood |
| Buyer composition | LLC share **decreased** post-flood | Not investor acquisition |

### Key Insight

The positive price effect reflects **composition changes** (newer construction, rebuilding) rather than investor acquisition. Properties that sold post-flood inside the inundation zone were systematically newer and larger than pre-flood sales.

### Phase 5 Data Extensions

| Extension | Status | Key Finding |
|-----------|--------|-------------|
| NFIP Claims Integration | Complete | 818 claims in 2019 (15.7× average), validates flood boundary |
| Extended Panel | Complete | t-24 to t+45 months, 8,313 sales, COVID controls added |
| Buyer Distance Geocoding | Infrastructure complete | Census Geocoder integration ready |

**See**: [doc/ANALYSIS_SUMMARY.md](doc/ANALYSIS_SUMMARY.md) for complete findings, [doc/PIPELINE.md](doc/PIPELINE.md) for Phase 5 documentation.

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

# Stage 07c: Robustness and extension modules (run all at once)
python src/pipeline.py run_all_diagnostics -b inund -c 300

# Or run individual robustness modules:
python src/pipeline.py spatial_econometrics -b inund -c 300
python src/pipeline.py placebo_tests -b inund -c 300 -n 500
python src/pipeline.py selection_correction -b inund -c 300
python src/pipeline.py quantile_effects -b inund -c 300
python src/pipeline.py extended_horizon -b inund -c 300
python src/pipeline.py mechanism_analysis -b inund -c 300
python src/pipeline.py trend_analysis -c 300 --start-year 2010
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
│   ├── 07_estimation/          # Event study, Poisson, RD, diagnostics, robustness
│   │   ├── spatial_econometrics.py   # Conley SEs, SAR, SEM
│   │   ├── placebo_tests.py          # Falsification tests
│   │   ├── selection_correction.py   # Heckman, IPW, Lee bounds
│   │   ├── quantile_effects.py       # Quantile DiD
│   │   ├── extended_horizon.py       # Extended time horizon
│   │   ├── mechanism_analysis.py     # Mechanism investigation
│   │   └── run_all_diagnostics.py    # Run all diagnostics
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
├── notebooks/                   # Jupyter notebooks
└── manuscript_quarto/           # Quarto manuscript (HTML/PDF/DOCX)
```

---

## Quarto Manuscript System

The project includes a modern Quarto-based manuscript system in `manuscript_quarto/` that generates publication-ready outputs in multiple formats.

### Quick Start

```bash
cd manuscript_quarto
~/local/quarto/bin/quarto render          # All formats
~/local/quarto/bin/quarto render --to pdf # PDF only
~/local/quarto/bin/quarto preview         # Live preview
```

### Output Formats

| Format | File | Features |
|--------|------|----------|
| HTML | `_output/freeze-rebuild.html` | Interactive TOC, code folding, format links |
| PDF | `_output/freeze-rebuild.pdf` | Journal-ready, letter paper, 1-inch margins |
| DOCX | `_output/freeze-rebuild.docx` | Track changes compatible |

### Manuscript Structure

- **Main manuscript**: `freeze-rebuild.qmd` - Complete "Freeze and Rebuild" paper
- **Appendix A**: Data and Study Area
- **Appendix B**: Identification Diagnostics
- **Appendix C**: Robustness Specifications
- **Appendix D**: Price Decomposition
- **Appendix E**: Mechanism Analysis

### Prerequisites

- Quarto >= 1.4 (install: `brew install quarto` or from [quarto.org](https://quarto.org))
- Python >= 3.9 with pandas, numpy, tabulate
- TinyTeX for PDF output (`quarto install tinytex`)

See [manuscript_quarto/README.md](manuscript_quarto/README.md) for detailed build instructions.

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
