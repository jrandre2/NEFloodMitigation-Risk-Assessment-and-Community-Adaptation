# Changelog

All notable changes to this project will be documented in this file.

---

## [2025-12-23] Phase 7: Quarto Manuscript System

### Summary
Created a modern Quarto-based manuscript system that generates publication-ready outputs in HTML, PDF, and DOCX formats. The manuscript presents the "Freeze and Rebuild" narrative with complete appendices.

### New Directory: `manuscript_quarto/`

**Core Manuscript Files**:
- `freeze-rebuild.qmd` - Main manuscript (~25KB) with complete academic paper
- `appendix-a-data.qmd` - Appendix A: Data and Study Area
- `appendix-b-identification.qmd` - Appendix B: Identification Diagnostics
- `appendix-c-robustness.qmd` - Appendix C: Robustness Specifications
- `appendix-d-decomposition.qmd` - Appendix D: Price Decomposition
- `appendix-e-mechanisms.qmd` - Appendix E: Mechanism Analysis

**Configuration & Support**:
- `_quarto.yml` - Multi-format output configuration (HTML/PDF/DOCX)
- `references.bib` - Bibliography with 23 citations
- `apa.csl` - APA 7th edition citation style
- `code/_common.py` - Python utilities for loading diagnostic CSVs
- Symlinks to `figures/` and `data_work/diagnostics/`

### Key Features

| Feature | Description |
|---------|-------------|
| Live Tables | Python code chunks load diagnostic CSVs at render time |
| Multi-format | Single source generates HTML, PDF, and DOCX |
| Reproducible | Tables regenerate from pipeline outputs automatically |
| Interactive HTML | Code folding, TOC, cross-reference links |

### Build Commands

```bash
cd manuscript_quarto
quarto render              # All formats
quarto render --to html    # HTML only
quarto render --to pdf     # PDF only
quarto preview             # Live preview with hot reload
```

### Local Quarto Installation
- Quarto 1.8.26 installed to `~/local/quarto/`
- TinyTeX installed for PDF rendering

---

## [2025-12-23] Phase 5 Data Extensions & Phase 6 Documentation Update

### Summary
Completed Phase 5 data extensions (NFIP integration, extended panel, geocoding infrastructure) and Phase 6 comprehensive documentation update. Project narrative shifted from "Freeze and Flight" to "Freeze and Rebuild" based on investigation findings.

### Phase 5: Data Extensions

**New Modules Added**:
- `src/00_ingest/load_nfip.py` - NFIP claims data extraction and filtering
- `src/07_estimation/nfip_analysis.py` - NFIP tract-level analysis
- `src/05_features/geocode_buyer_addresses.py` - Owner address geocoding via Census API
- `src/06_panels/extend_panel.py` - Extended panel construction (through Dec 2022)

**New CLI Commands**:
```bash
python src/pipeline.py load_nfip          # Load NFIP claims data
python src/pipeline.py nfip_analyze       # Run NFIP analysis
python src/pipeline.py geocode_addresses  # Geocode owner addresses
python src/pipeline.py panel_extend       # Build extended panel
```

**Key Findings**:
| Extension | Result |
|-----------|--------|
| NFIP Claims | 818 claims in 2019 (15.7× historical average), $37.5M payments |
| Extended Panel | 70 months (t-24 to t+45), 8,313 sales, COVID indicators added |
| Geocoding | Infrastructure complete, Census Geocoder integration ready |

### Phase 6: Documentation Update

**Files Updated**:
- `doc/DATA_DICTIONARY.md` - Added extended panel, NFIP, and buyer distance variable sections
- `doc/PIPELINE.md` - Added Stage 07d (Investigation Modules) and Stage 07e (Supplementary Methods)
- `README.md` - Updated status to "Phase 5 Complete", added Phase 5 findings summary
- `doc/CHANGELOG.md` - Added Phase 5 and Phase 6 entries

**New Documentation Sections**:
- Extended Panel Variables (COVID controls, event_time range)
- NFIP Variables (claims data, tract summary)
- Buyer Distance Variables (geocoded addresses, distance bands)
- Stage 07d: Investigation Modules (7 modules documented)
- Stage 07e: Supplementary Robustness Methods (3 modules documented)

### Narrative Update

Changed project narrative from "Freeze and Flight" to "Freeze and Rebuild":
- Positive price effect explained by composition changes (newer/larger properties)
- LLC share decreased post-flood (contradicts investor acquisition hypothesis)
- Inundation boundary passes pre-trends (p=0.779), used as primary specification

---

## [2025-12-23] Comprehensive Documentation Update

### Summary
Created comprehensive documentation summarizing all diagnostic findings, manuscript revision requirements, and flagged a critical issue requiring further investigation.

### Critical Issue Identified
**Counterintuitive Positive Price Effect**: Analysis shows prices rose faster inside the flood zone post-event, contradicting the "flight" narrative. This requires investigation before manuscript revision decisions can be finalized.

### New Documentation Files

| File | Purpose |
|------|---------|
| `doc/ANALYSIS_SUMMARY.md` | Executive summary of all diagnostic findings |
| `doc/MANUSCRIPT_REVISION_CHECKLIST.md` | Actionable checklist for manuscript revisions (BLOCKED) |
| `doc/RESULTS_INTERPRETATION.md` | Investigation plan for counterintuitive findings |

### Updated Documentation Files

- `doc/CRITIQUE_FINDINGS.md` - Added Finding 9: Differential Geographic Trends
- `doc/METHODOLOGY.md` - Added Differential Trend Analysis section
- `doc/PIPELINE.md` - Added trend_analysis module documentation
- `doc/DATA_DICTIONARY.md` - Added trend analysis output variables
- `README.md` - Added Current Analysis Status section

### Next Steps
1. **BLOCKING**: Investigate counterintuitive positive price effect
2. Determine if selection, composition, or specification explains finding
3. Decide on primary specification after investigation
4. Complete manuscript revisions

---

## [2025-12-23] Peer-Review Robustness Extension

### Summary
Major extension of the analysis pipeline to address peer-review robustness requirements. Added 7 new estimation modules implementing spatial econometrics, selection correction, quantile effects, placebo tests, mechanism analysis, and extended horizon analysis.

### Key Methodological Changes

| Change | Rationale |
|--------|-----------|
| Switch to inundation boundary as primary specification | SFHA pre-trends fail (F=3.33, p<0.001); inundation passes (p=0.34) |
| Added Conley spatial HAC standard errors | Address Moran's I=0.392 spatial autocorrelation |
| Implemented selection correction for price models | Only observe prices for sold properties (selection bias) |
| Added placebo/falsification tests | Strengthen causal identification claims |
| Extended time horizon to ±36 months | Test effect persistence beyond standard window |
| Added quantile treatment effects | Examine distributional heterogeneity |
| Comprehensive mechanism analysis | Test insurance, credit, and buyer composition channels |

### New Files Added

**Estimation Modules** (`src/07_estimation/`):
- `spatial_econometrics.py` - Conley SEs, SAR/SEM models, Moran's I tests
- `placebo_tests.py` - Fake event dates, boundary shifts, permutation inference, triple-diff
- `selection_correction.py` - Heckman two-step, IPW, Lee (2009) bounds
- `quantile_effects.py` - Quantile DiD at τ = {0.1, 0.25, 0.5, 0.75, 0.9}
- `extended_horizon.py` - Extended event study (±36 months), persistence analysis
- `mechanism_analysis.py` - Insurance, credit constraints, buyer composition channels
- `run_all_diagnostics.py` - Orchestrator for complete diagnostics suite

### Modified Files

**Core Analysis**:
- `src/07_estimation/event_study.py` - Rewrote with dual boundary support, confidence intervals, pre-trends F-test
- `src/07_estimation/rd_summary.py` - Changed default to inundation boundary, added CLI arguments
- `src/07_estimation/rd_diagnostics.py` - Added cross-references to new modules

**Pipeline**:
- `src/pipeline.py` - Added 9 new CLI commands with argument handling

**Dependencies**:
- `requirements.txt` - Added libpysal, esda, spreg, mgwr for spatial econometrics

**Documentation**:
- `doc/PIPELINE.md` - Added Stage 07c documenting all new robustness modules
- `doc/METHODOLOGY.md` - Added Extended Robustness Methods section
- `README.md` - Updated project structure and CLI commands

### New CLI Commands

```bash
# Run complete diagnostics suite
python src/pipeline.py run_all_diagnostics -b inund -c 300

# Individual modules
python src/pipeline.py spatial_econometrics -b inund -c 300
python src/pipeline.py placebo_tests -b inund -c 300 -n 500
python src/pipeline.py selection_correction -b inund -c 300
python src/pipeline.py quantile_effects -b inund -c 300
python src/pipeline.py extended_horizon -b inund -c 300
python src/pipeline.py mechanism_analysis -b inund -c 300
```

### New Diagnostic Outputs

**CSV Files** (`data_work/diagnostics/`):
- `mccrary_density_test.csv` - Manipulation test results
- `pretrends_ftest.csv` - Pre-trends F-test results
- `bandwidth_sensitivity.csv` - Caliper robustness
- `donut_rd.csv` - Spillover exclusion tests
- `placebo_event_dates.csv` - Fake event date tests
- `placebo_boundaries.csv` - Shifted boundary tests
- `permutation_pvalues.csv` - Randomization inference
- `leave_one_out.csv` - Leave-one-out stability
- `triple_diff.csv` - Triple difference estimates
- `ipw_results.csv` - IPW estimates
- `lee_bounds.csv` - Lee bounds
- `selection_comparison.csv` - Selection method comparison
- `quantile_did.csv` - Quantile treatment effects
- `distribution_shift_test.csv` - K-S test results
- `dynamic_effects_extended.csv` - Extended event study
- `persistence_analysis.csv` - Effect persistence
- `mechanism_insurance.csv` - Insurance channel
- `mechanism_credit.csv` - Credit constraints
- `heterogeneity_by_chars.csv` - Property characteristic heterogeneity
- `mechanism_summary.csv` - Mechanism evidence summary

**Figures** (`figures/`):
- `fig_mccrary_sfha.png`, `fig_mccrary_inund.png` - Density discontinuity
- `fig_event_study_*.png` - Event studies with CI
- `fig_bandwidth_*.png` - Bandwidth sensitivity
- `fig_placebo_event_dates.png` - Placebo test results
- `fig_placebo_boundaries.png` - Boundary shift tests
- `fig_permutation_distribution.png` - Permutation null distribution
- `fig_quantile_effects.png` - Quantile treatment effects
- `fig_extended_event_study.png` - Extended horizon
- `fig_mechanism_heterogeneity.png` - Heterogeneous effects
- `fig_selection_correction_comparison.png` - Selection methods

### Key Diagnostic Results

**Pre-Trends Tests**:
- SFHA ±300m: F=3.33, p<0.001 (FAIL - pre-trends violated)
- Inundation ±300m: F=1.50, p=0.152 (PASS - supports parallel trends)

**McCrary Density Test**:
- SFHA: z=7.78, p<0.001 (significant discontinuity - expected due to topography)
- Inundation: z=7.48, p<0.001 (significant discontinuity)

**Placebo Tests**:
- Permutation p-value: 0.16 (not significant at 5%)
- Leave-one-out: Stable estimates (max deviation: 0.03 pp)

**Selection Correction**:
- IPW coefficient: -0.26 (SE: 0.30)
- Lee bounds: [-0.42, -0.04]

**Quantile Effects**:
- Larger negative effects at upper quantiles (τ=0.90: -0.72)
- Positive effect at lowest quantile (τ=0.10: +0.35)

**Mechanism Analysis**:
- Insurance channel: SFHA-Inund diff = -0.0025 (supports insurance mechanism)
- Credit constraints: Larger effects for high-value properties

---

## [2025-12-23] Differential Trend Analysis Extension

### Summary
Added supplementary analysis to test for and control differential geographic trends that may confound flood impact estimates. Addresses reviewer concern that flooded areas had high pre-existing housing market growth.

### Key Finding
**Trend controls substantially affect DiD estimates** - coefficient changes by 165% when group-specific trends are added to the model.

### Analysis Details

| Metric | Value |
|--------|-------|
| Pre-event inside zone growth | 28.5%/year |
| Pre-event outside zone growth | 21.7%/year |
| Differential growth | 6.8 percentage points |
| Pre-trend in inside share | -0.0004/month (p=0.56, not significant) |
| Differential price trend | -5,380/month (p=0.08, marginally significant) |

### DiD Comparison

| Model | Coefficient | SE | p-value |
|-------|-------------|-----|---------|
| Without trends | 0.2923 | 0.145 | 0.044 |
| With trends | 0.7757 | 0.278 | 0.005 |

### New Files Added

- `src/07_estimation/trend_analysis.py` - Differential trend testing and trend-adjusted DiD

### New CLI Commands

```bash
# Run trend analysis with extended time window
python src/pipeline.py trend_analysis -c 300 --start-year 2010 --end-year 2022
```

### New Outputs

- `data_work/diagnostics/trend_analysis.csv` - Pre-trend test results
- `data_work/diagnostics/trend_adjusted_did.csv` - DiD with/without trend controls
- `figures/fig_trend_analysis.png` - Trend visualization

### Recommendations

1. Report DiD results with and without trend controls
2. Consider adding group-specific trends to main specification
3. Investigate the counterintuitive positive coefficient (prices rose faster inside flood zone)
4. Extended pre-period (2010-2018) provides better baseline for trend estimation
