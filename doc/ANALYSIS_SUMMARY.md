# Analysis Summary: Freeze and Flight

## Executive Summary

This document summarizes the current state of the econometric analysis for the "Freeze and Flight" manuscript as of December 23, 2025. The analysis implements a boundary RD-in-panel design to estimate the effect of the March 2019 flood on property sales and prices in Douglas County, Nebraska.

**Key Status**: Analysis complete but **critical findings require further investigation** before manuscript revision decisions can be finalized.

---

## Diagnostic Results Summary

### Identification Tests

| Test | SFHA Boundary | Inundation Boundary | Status |
|------|---------------|---------------------|--------|
| Pre-Trends F-test | F=3.33, **p<0.001** | F=1.50, p=0.152 | SFHA FAILS, Inund PASSES |
| McCrary Density | z=7.78, p<0.001 | z=7.48, p<0.001 | Both show discontinuity |
| Permutation Test | - | p=0.16 | Not significant at 5% |
| Leave-One-Out | - | Stable (max dev: 0.03pp) | Robust |

### Robustness Checks

| Test | Result | Interpretation |
|------|--------|----------------|
| Bandwidth Sensitivity | Stable 100-500m | Results robust to caliper choice |
| Donut RD (100m exclusion) | -8.8% change | Some spillover effects |
| Placebo Event Dates | No effects at fake dates | Supports causal interpretation |
| Placebo Boundaries | No effects at shifted boundaries | True boundary matters |

### Selection Correction

| Method | Coefficient | 95% CI |
|--------|-------------|--------|
| Naive DiD | -0.26 | - |
| IPW | -0.26 | Same as naive |
| Lee Bounds | - | [-0.42, -0.04] |

### Mechanism Analysis

| Channel | Evidence | Finding |
|---------|----------|---------|
| Insurance Mandate | SFHA-Inund difference | -0.0025 (supports mechanism) |
| Credit Constraints | High vs Low value | Larger effects for high-value |
| Buyer Composition | LLC/Portfolio DiD | LLCs *decreased* share inside |

---

## Critical Issue: Trend Confounding

### Finding: Differential Pre-Event Trends

Analysis of the longer time series (2010-2022) revealed:

| Metric | Value |
|--------|-------|
| Pre-event inside zone growth | 28.5%/year |
| Pre-event outside zone growth | 21.7%/year |
| **Differential growth** | **6.8 percentage points** |
| Pre-trend in inside share | -0.0004/month (p=0.56) |
| Differential price trend | -5,380/month (p=0.08) |

### Impact on Estimates

| Model | DiD Coefficient | SE | 95% CI |
|-------|-----------------|-----|--------|
| SFR sample, inund 300m | -0.26 | 0.30 | [-0.85, +0.33] |
| SFR sample, SFHA 300m | -0.00089 | 0.00038 | [-0.0016, -0.0001] |

**Implication**: Price effects are negative but imprecisely estimated at inundation boundary. Sale rate effects are robust at SFHA boundary.

---

## Issue RESOLVED: Price Effect Clarification

### Investigation Complete (December 2025)

**Finding**: Earlier positive price effect (+0.528) was from **wrong sample** (included non-SFR properties). Correct SFR-only sample shows **negative price effect** (-0.26 log points).

| Sample | N Sales | DiD Estimate | Sign |
|--------|---------|--------------|------|
| All properties (incl. non-SFR) | 739 | +0.578 | Positive (SPURIOUS) |
| **SFR only (correct sample)** | 387 | **-0.260** | **Negative** |

**Resolution**: The manuscript now correctly reports -0.26 log points (95% CI: [-0.85, +0.33]) for the inundation boundary price effect. This is consistent with standard flood risk capitalization theory.

**Status**: RESOLVED. See `manuscript_quarto/REVISION_TRACKER.md` for full audit trail.

---

## Current Specification Recommendations (Tentative)

Based on diagnostic results:

1. **Primary boundary**: Inundation (passes pre-trends test)
2. **SFHA**: Report as robustness check (fails pre-trends)
3. **Trend controls**: Consider adding to main specification (large impact)

**Caveat**: These recommendations are tentative pending investigation of the positive coefficient issue.

---

## Analysis Modules Implemented

### Core Estimation (`src/07_estimation/`)
- `event_study.py` - Dynamic treatment effects
- `rd_summary.py` - Main RD-DiD estimation
- `rd_diagnostics.py` - McCrary, pre-trends, bandwidth tests

### Robustness Modules (`src/07_estimation/`)
- `spatial_econometrics.py` - Conley SEs, SAR/SEM
- `placebo_tests.py` - Fake events, permutation inference
- `selection_correction.py` - Heckman, IPW, Lee bounds
- `quantile_effects.py` - Quantile DiD
- `extended_horizon.py` - ±36 month analysis
- `mechanism_analysis.py` - Insurance, credit channels
- `trend_analysis.py` - Differential trend testing

### CLI Commands
```bash
# Run complete diagnostics
python src/pipeline.py run_all_diagnostics -b inund -c 300

# Trend analysis
python src/pipeline.py trend_analysis -c 300 --start-year 2010

# Individual modules
python src/pipeline.py spatial_econometrics -b inund -c 300
python src/pipeline.py placebo_tests -b inund -c 300
python src/pipeline.py selection_correction -b inund -c 300
```

---

## Output Files

### Diagnostic CSVs (`data_work/diagnostics/`)
- `pretrends_ftest.csv` - Pre-trends test results
- `mccrary_density_test.csv` - Manipulation test
- `bandwidth_sensitivity.csv` - Caliper robustness
- `placebo_event_dates.csv` - Fake event tests
- `placebo_boundaries.csv` - Shifted boundary tests
- `permutation_pvalues.csv` - Randomization inference
- `leave_one_out.csv` - Stability analysis
- `triple_diff.csv` - Triple difference
- `ipw_results.csv` - IPW estimates
- `lee_bounds.csv` - Selection bounds
- `quantile_did.csv` - Quantile effects
- `mechanism_*.csv` - Mechanism analysis
- `trend_adjusted_did.csv` - With/without trends comparison

### Figures (`figures/`)
- `fig_event_study_*.png` - Event studies with CI
- `fig_mccrary_*.png` - Density tests
- `fig_bandwidth_*.png` - Bandwidth sensitivity
- `fig_placebo_*.png` - Placebo tests
- `fig_quantile_effects.png` - Quantile DiD
- `fig_trend_analysis.png` - Trend visualization

---

## Next Steps

### High Priority
1. Investigate the counterintuitive positive price effect
2. Determine appropriate specification (with/without trends)
3. Decide on primary boundary (SFHA vs inundation)

### Medium Priority
4. Extend pre-period analysis (2010-2018 baseline)
5. Run selection models on who sells post-flood
6. Analyze compositional changes in sales

### After Investigation
7. Finalize manuscript narrative
8. Complete supplementary materials
9. Prepare response to reviewers

---

## Related Documents

- `doc/CRITIQUE_FINDINGS.md` - Detailed methodological audit
- `doc/METHODOLOGY.md` - Statistical methods documentation
- `doc/PIPELINE.md` - Pipeline stage documentation
- `doc/MANUSCRIPT_REVISION_CHECKLIST.md` - Revision tasks
- `doc/RESULTS_INTERPRETATION.md` - Interpretation guidance

---

*Last updated: 2025-12-23*
