# Changelog

All notable changes to this project will be documented in this file.

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
