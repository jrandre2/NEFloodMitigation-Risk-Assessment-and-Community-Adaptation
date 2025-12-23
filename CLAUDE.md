# Freeze and Flight - Claude Code Project Instructions

## Project Overview

This is an empirical economics research project analyzing housing market responses to the March 2019 Missouri River flood in Douglas County, Nebraska. The project uses boundary regression discontinuity-in-panel (RD-in-panel) designs at FEMA SFHA and inundation boundaries.

**Author**: Jesse Andrews, Texas Tech University
**Status**: Phase 7 Complete - Quarto manuscript system added

---

## Quick Reference

### Virtual Environment (REQUIRED)

```bash
source .venv/bin/activate
```

All pipeline scripts require the virtual environment. Scripts check for `VIRTUAL_ENV` ending with `/.venv` and will exit if not activated.

### Common Commands

```bash
# Run pipeline stages
python src/pipeline.py <command>

# Key estimation commands
python src/pipeline.py rd_summary -b inund -c 300
python src/pipeline.py event_study
python src/pipeline.py run_all_diagnostics -b inund -c 300

# Figure generation
python src/pipeline.py make_figures

# Quarto manuscript
cd manuscript_quarto
~/local/quarto/bin/quarto render          # All formats (HTML/PDF/DOCX)
~/local/quarto/bin/quarto preview         # Live preview
```

---

## Key Concepts for AI Agents

### Two Boundaries

1. **Inundation (PRIMARY)** - Actual 2019 flood extent from Sentinel-2 imagery. **Passes pre-trends (F=1.50, p=0.152)**. Use this for main results.
2. **SFHA (SECONDARY)** - FEMA regulatory flood hazard boundary. **Fails pre-trends (F=3.33, p<0.001)**. Use for robustness checks only.

### Signed Distance Convention

- **Negative distance = INSIDE** the hazard zone
- **Positive distance = OUTSIDE** the hazard zone
- Variables: `signed_dist_sfha_m`, `signed_dist_inund_m`
- Default caliper window: ±300m from boundary

### Event Time

- `event_m = 0` is March 2019 (flood date)
- Pre-period: `event_m` in [-24, -1]
- Post-period: `event_m` in [0, +24]
- Extended panel: `event_m` in [-24, +45]

### Key Finding

Price effects are **negative but imprecisely estimated** (-0.26 log points, 95% CI: [-0.85, +0.33]) at the inundation boundary, consistent with standard flood risk capitalization. The market exhibits a sharp **liquidity freeze** (38% relative decline in sale rates inside SFHA). **Composition shifts** toward newer/larger properties are detected post-flood, and LLC buyer share actually **decreased** inside the inundation zone.

---

## Pipeline Architecture

```
Pre-pipeline: build_parcels -> build_treatments
     |
     v
00_ingest -> 01_link -> 02_labels -> 03_exposure -> 04_salesclean -> 05_features -> 06_panels -> 07_estimation -> 08_figures -> 09_manuscript
```

Stages must run in order. Stage 09 (Quarto manuscript) is optional and requires Quarto installation.

### Critical Data Files

| File | Purpose |
|------|---------|
| `data_work/panel_parcel_month.parquet` | Main analysis panel (parcel x month) |
| `data_work/panel_parcel_month_extended.parquet` | Extended horizon panel (through 2022) |
| `data_work/parcel_boundary_distances.parquet` | Parcel distances to SFHA/inundation boundaries |
| `data_work/parcels_sfr.gpkg` | SFR parcel geometries |
| `data_work/parcel_covariates_full.parquet` | Assessor covariates merged to parcels |
| `data_work/assessor_clean.parquet` | Cleaned assessor data |
| `src/pipeline.py` | Main CLI orchestrator |

---

## Code Conventions

### File Formats

- **Parquet** for tabular data (pandas/pyarrow)
- **GeoPackage (.gpkg)** for spatial data (geopandas)
- **CRS**: WGS84 (EPSG:4326) for output geometries, UTM Zone 15N (EPSG:32615) for distance calculations

### Python Patterns

```python
# Scripts check for venv activation
def ensure_env():
    venv = os.environ.get("VIRTUAL_ENV", "")
    if not venv.endswith("/.venv"):
        sys.exit("Activate .venv first")

# Common imports
import pandas as pd
import geopandas as gpd
from pathlib import Path
import statsmodels.api as sm
```

### Output Locations

| Type | Location |
|------|----------|
| Processed data | `data_work/` |
| Diagnostic CSVs | `data_work/diagnostics/` |
| Figures | `figures/` |
| Reports | `data_work/report_*.md` |

---

## Estimation Commands Reference

| Task | Command |
|------|---------|
| Main RD-DiD | `python src/pipeline.py rd_summary -b inund -c 300` |
| Event study | `python src/pipeline.py event_study` |
| Full diagnostics | `python src/pipeline.py run_all_diagnostics -b inund -c 300` |
| Spatial SE | `python src/pipeline.py spatial_econometrics -b inund -c 300` |
| Placebo tests | `python src/pipeline.py placebo_tests -b inund -c 300 -n 500` |
| Selection correction | `python src/pipeline.py selection_correction -b inund -c 300` |
| Quantile effects | `python src/pipeline.py quantile_effects -b inund -c 300` |
| Extended horizon | `python src/pipeline.py extended_horizon -b inund -c 300` |
| Mechanism analysis | `python src/pipeline.py mechanism_analysis -b inund -c 300` |
| Trend analysis | `python src/pipeline.py trend_analysis -c 300 --start-year 2010` |
| Investigation suite | `python src/pipeline.py run_investigation -c 300` |

**Flag meanings:**
- `-b inund` or `-b sfha`: Boundary type (use `inund` for primary results)
- `-c 300`: Caliper window in meters (default 300m)
- `-n 500`: Number of permutations for placebo tests

---

## Important Constraints

### DO NOT

- Modify raw data files in `GIS_Data/` or `statewide parcel/`
- Re-run boundary distance calculation if `parcel_boundary_distances.parquet` exists (expensive)
- Use SFHA as primary boundary (fails pre-trends test)
- Push large parquet/gpkg files to git (check `.gitignore`)
- Run git commands without checking OneDrive sync status (see `doc/agents.md`)

### ALWAYS

- Activate `.venv` before running any Python scripts
- Use `-b inund -c 300` flags for estimation commands
- Check `doc/PIPELINE.md` for detailed stage documentation
- Run diagnostics after making estimation changes
- Verify pre-trends pass before interpreting treatment effects

---

## Hardcoded Paths (Require Modification for Other Environments)

| Script | Path | Description |
|--------|------|-------------|
| `src/03_exposure/boundary_from_gdb.py` | `/Users/jesseandrews/Documents/ArcGIS/Projects/OwnerDistanceProject/OwnerDistanceProject.gdb` | ArcGIS GDB with parcel/FIRM/inundation layers |
| `src/03_exposure/boundary_prepare.py` | Same as above | Same GDB path |
| `src/05_features/extract_assessor.py` | `statewide parcel/NE_2023_statewideparcels.gdb` | Nebraska statewide assessor GDB (735 MB) |

---

## Key Results Reference

| Result | Value | Source Script |
|--------|-------|---------------|
| Sale rate DiD (SFHA, 300m) | -0.00089 (38% relative decline) | `rd_summary.py` |
| Sale rate DiD (inund, 300m) | +0.00157 (elevated inside) | `rd_summary.py` |
| Price effect (inund, SFR) | -0.26 log points (95% CI: [-0.85, +0.33]) | `rd_summary.py` |
| Pre-trends F-test (inund) | F=1.50, p=0.152 (PASSES) | `rd_diagnostics.py` |
| Pre-trends F-test (sfha) | F=3.33, p<0.001 (FAILS) | `rd_diagnostics.py` |
| LLC share post-flood | Decreased (not increased) | `buyer_composition_did.py` |
| Composition shift | Newer/larger properties selling post-flood | `oaxaca_blinder_decomposition.py` |
| NFIP claims 2019 | 818 claims (15.7x average) | `nfip_analysis.py` |

---

## Interpreting Diagnostic Results

### Pre-Trends Test
- **p > 0.05**: Parallel trends assumption supported (good)
- **p < 0.05**: Pre-existing differential trends (problematic, use caution)

### McCrary Density Test
- Tests for manipulation/sorting at boundary
- Significant discontinuity at SFHA boundary is expected (topographic constraints on development)
- Inundation boundary density test is more informative

### Covariate Balance
- 0/8 covariates balanced at SFHA boundary (inside parcels are older, smaller, lower value)
- This explains why covariate-adjusted RD estimate (-15%) is smaller than raw estimate (-40%)

---

## Documentation Reference

| Document | Purpose |
|----------|---------|
| `doc/PIPELINE.md` | Complete pipeline documentation (1165 lines) |
| `doc/METHODOLOGY.md` | Statistical methods and identification strategy |
| `doc/DATA_DICTIONARY.md` | Variable definitions and data sources |
| `doc/ANALYSIS_SUMMARY.md` | Current findings and project status |
| `doc/CRITIQUE_FINDINGS.md` | RD diagnostic results interpretation |
| `doc/INVESTIGATION_REPORT.md` | Phase 5 investigation findings |
| `doc/agents.md` | Troubleshooting (OneDrive sync, git issues) |

---

## For Manuscript Work

### Quarto Manuscript (Primary)

Location: `manuscript_quarto/`

| File | Description |
|------|-------------|
| `freeze-rebuild.qmd` | Main manuscript (~25KB) |
| `appendix-a-data.qmd` | Appendix A: Data and Study Area |
| `appendix-b-identification.qmd` | Appendix B: Identification Diagnostics |
| `appendix-c-robustness.qmd` | Appendix C: Robustness Specifications |
| `appendix-d-decomposition.qmd` | Appendix D: Price Decomposition |
| `appendix-e-mechanisms.qmd` | Appendix E: Mechanism Analysis |

**Build commands:**

```bash
cd manuscript_quarto
~/local/quarto/bin/quarto render              # All formats
~/local/quarto/bin/quarto render --to html    # HTML only
~/local/quarto/bin/quarto render --to pdf     # PDF only (requires TinyTeX)
~/local/quarto/bin/quarto preview             # Live preview with hot reload
```

**Output files:** `manuscript_quarto/_output/freeze-rebuild.{html,pdf,docx}`

**Prerequisites:** Quarto >= 1.4 (`~/local/quarto/`), TinyTeX for PDF

### Legacy Markdown (Reference Only)

- `manuscript/drafts/Freeze_and_Flight.md` - Original draft
- `manuscript/tables/Tables_Freeze_Flight.md` - Standalone tables

---

## Large Files (Expensive to Process)

Do not regenerate these unless necessary:

| File | Size | Processing Time |
|------|------|-----------------|
| `statewide parcel/NE_2023_statewideparcels.gdb` | 735 MB | - |
| `data_work/parcel_boundary_distances_part_*` | ~50 MB each (64 chunks) | Hours |
| `GIS_Data/Building_Footprints/Nebraska.geojson` | ~500 MB | - |
| `data_work/assessor_raw.parquet` | 51 MB | ~7 sec |

---

## Troubleshooting

See `doc/agents.md` for OneDrive sync issues and git operations in synced directories.

**Quick fix for git hangs:** `rm -f .git/index.lock` (only if no git operation running)

---

## Dependencies

Key packages: `pandas`, `geopandas`, `statsmodels`, `torch`, `transformers`, `libpysal`, `spreg`

Install: `pip install -r requirements.txt`
