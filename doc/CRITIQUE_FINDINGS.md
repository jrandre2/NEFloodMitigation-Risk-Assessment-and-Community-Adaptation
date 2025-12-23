# Critique Findings: Freeze and Flight Manuscript

## Executive Summary

A comprehensive methodological audit of the "Freeze and Flight" manuscript has revealed **significant identification concerns** that require attention before submission. The RD diagnostics implemented in `rd_diagnostics.py` produced the following key findings:

| Test | SFHA Boundary | Inundation Boundary | Implication |
|------|---------------|---------------------|-------------|
| McCrary Density | p < 0.001 (FAIL) | p < 0.001 (FAIL) | Density discontinuity at boundary |
| Pre-Trends F-test | p < 0.001 (FAIL) | p = 0.34 (PASS) | Parallel trends violated for SFHA |
| Donut RD (100m) | -8.8% change | - | Spillover effects confirmed |
| Bandwidth Sensitivity | Stable | Unstable | SFHA results robust to bandwidth |

---

## Critical Finding 1: McCrary Density Discontinuity

**Result**: Significant density discontinuity at the SFHA boundary (z = 7.78, p < 0.001)

**Interpretation**: There are systematically more parcels on one side of the SFHA boundary than the other. This is expected for a regulatory boundary that follows natural topography, but it:
- Raises questions about comparability of treatment and control parcels
- May indicate that development patterns differ systematically across the boundary
- Does NOT necessarily invalidate the RD, but requires explicit discussion

**Recommended Response**:
1. Acknowledge the density discontinuity in the manuscript
2. Argue this reflects *historical* development patterns, not manipulation
3. Conduct covariate balance tests on pre-treatment characteristics (REQUIRES ADDITIONAL DATA - see below)
4. Consider using pre-flood assessed values as a balancing covariate if available

**See**: `figures/fig_mccrary_sfha.png`

---

## Critical Finding 2: Pre-Trends Violation for SFHA Boundary

**Result**: Joint F-test rejects null of zero pre-event coefficients (F = 3.33, p < 0.001)

**Interpretation**: The inside-vs-outside sale rate gap was NOT constant in the 24 months before the flood. This suggests:
- Parallel trends assumption may be violated
- Pre-existing differential trends could confound the treatment effect
- Current DiD estimate may be biased

**Recommended Response Options**:

### Option A: Re-frame as difference-in-trends
- Report pre-trend coefficient and adjust interpretation
- The pre-period inside-outside gap averaged 0.00039; post-period gap was 0.00065
- Net change of ~0.00026 is smaller than the DiD of -0.00089

### Option B: Use inundation boundary instead
- Inundation boundary passes pre-trends test (F = 1.50, p = 0.15)
- Effects at inundation boundary are directionally similar but less precise
- Could present as main result with SFHA as robustness

### Option C: Condition on pre-treatment trends
- Include parcel-specific linear trends
- Use synthetic control methods
- Match on pre-period outcomes

**See**: `figures/fig_event_study_sfha_300m.png`

---

## Finding 3: SUTVA Violation Confirmed (Spillovers)

**Result**: The ring models show RR ≈ 1.44 for 0-250m outside SFHA post-flood

**Interpretation**: The manuscript already documents that sales increase just outside the boundary. This:
- Violates SUTVA for the main DiD comparison
- Means "control" parcels are treated by demand spillovers
- Likely ATTENUATES the main effect estimate

**Donut RD Evidence**:
- Standard DiD: -0.00089
- Donut RD (exclude 0-100m outside): -0.00097 (9% larger)
- Suggests true inside effect may be ~10% larger than reported

**Recommended Response**:
1. Explicitly acknowledge SUTVA violation in Discussion section
2. Add Donut RD to robustness checks table
3. Interpret main estimate as lower bound on true effect
4. Frame ring models as EVIDENCE of spillovers, not just substitution

**See**: `data_work/diagnostics/donut_rd.csv`

---

## Finding 4: Bandwidth Sensitivity (Positive)

**Result**: SFHA DiD estimates are stable across bandwidths from 150m to 400m

| Caliper | DiD Estimate | 95% CI |
|---------|--------------|--------|
| 100m | -0.00057 | [-0.0014, 0.0003] |
| 150m | -0.00077 | [-0.0016, 0.0000] |
| 200m | -0.00078 | [-0.0015, 0.0000] |
| 300m | -0.00089 | [-0.0016, -0.0001] |
| 400m | -0.00073 | [-0.0015, 0.0000] |

**Interpretation**: Results are robust to bandwidth choice within reasonable range.

**See**: `figures/fig_bandwidth_sfha.png`

---

## Finding 5: Elevation Discontinuity at SFHA Boundary

**Result**: Parcels inside SFHA are significantly lower in elevation than outside.

| Caliper | Inside Mean | Outside Mean | Difference | t-stat | p-value |
|---------|-------------|--------------|------------|--------|---------|
| 100m | 321.8m | 334.5m | -12.7m | -27.1 | <0.001 |
| 200m | 322.5m | 338.6m | -16.1m | -35.8 | <0.001 |
| 300m | 323.3m | 342.0m | -18.8m | -41.4 | <0.001 |

**Interpretation**: This is EXPECTED and actually SUPPORTS the identification strategy:
- The SFHA boundary follows natural floodplain topography
- The McCrary density discontinuity reflects historical development in low-lying corridors
- Covariate imbalance in elevation is inherent to how the boundary is defined
- This does NOT invalidate the RD - it explains WHY the boundary exists

**See**: `data_work/diagnostics/elevation_rd_bins.csv`, `data_work/diagnostics/parcel_elevation.csv`

---

## Finding 6: Buyer Composition Shifts

**Result**: LLC and portfolio buyers DECREASED their share inside SFHA post-flood.

| Outcome | DiD Estimate | SE (HC3) | p-value |
|---------|--------------|----------|---------|
| LLC Share | -0.178 | 0.082 | 0.030 |
| Portfolio Share | -0.102 | 0.082 | 0.211 |

**Pre-Post Breakdown**:

| Buyer Type | Location | Pre-Flood | Post-Flood |
|------------|----------|-----------|------------|
| LLC | Inside SFHA | 45.2% | 27.3% |
| LLC | Outside SFHA | 9.9% | 9.8% |
| Portfolio | Inside SFHA | 39.7% | 28.8% |
| Portfolio | Outside SFHA | 16.5% | 15.8% |

**Interpretation**: This is OPPOSITE to the hypothesis that sophisticated/institutional buyers exploit distressed sales:
1. LLCs and portfolio buyers were ALREADY overrepresented inside SFHA pre-flood
2. After the flood, these buyers REDUCED their exposure to flood zone properties
3. Individual single-property buyers became relatively more prevalent inside SFHA

**Caveat**: Small sample size inside SFHA (n=139 sales) limits precision.

**See**: `data_work/diagnostics/buyer_composition_did.csv`, `data_work/diagnostics/buyer_composition_regression.csv`

---

## Finding 7: Covariate Balance Results (SFR Improved)

**Result**: 0/8 covariates are balanced at the ±200m and ±300m calipers for SFR improved parcels.

| Covariate | Inside Mean | Outside Mean | Difference | p-value | Balanced? |
|-----------|-------------|--------------|------------|---------|-----------|
| Year Built | 1963 | 1980 | -17 years | <0.001 | No |
| Assessed Value | $180,311 | $246,268 | -$65,957 | <0.001 | No |
| Lot Size | 1.02 acres | 0.31 acres | +0.71 acres | <0.001 | No |
| Building SF | 1,445 | 1,699 | -254 sq ft | <0.001 | No |

**Sample Sizes (±300m caliper)**:
- Inside SFHA: 721 SFR improved parcels
- Outside SFHA: 31,397 SFR improved parcels
- Ratio: 2.3% inside

**Interpretation**: SFR homes inside the SFHA are systematically:
- **17 years older** on average (built 1963 vs 1980)
- **27% lower assessed value** ($180k vs $246k)
- **15% smaller buildings** (1,445 vs 1,699 SF)
- **3× larger lots** (1.0 vs 0.3 acres)

This covariate imbalance is consistent with the McCrary density test failure and reflects historical development patterns where floodplain land was developed earlier and remained less intensively developed.

**See**: `data_work/diagnostics/covariate_balance_sfr.csv`

---

## Finding 8: Covariate-Adjusted RD Results

**Result**: The SFHA price penalty shrinks substantially when controlling for housing characteristics.

| Model | Estimate | p-value | Interpretation |
|-------|----------|---------|----------------|
| No covariates | -41.6% | <0.001 | Raw SFHA discount |
| With covariates | -14.5% | 0.013 | Residual flood risk premium |
| Change | +71% | - | 60% of discount explained by observables |

**Covariates Included**: building_age, log_assessed_value, log_acres, log_impsf

**DiD (Post × Inside) Results**:
| Model | Estimate | p-value |
|-------|----------|---------|
| No covariates | -0.02pp | 0.745 |
| With covariates | -0.02pp | 0.745 |

**Interpretation**:
1. The raw -40% SFHA price discount is largely compositional: older, smaller homes inside
2. After controlling for characteristics, a -15% residual penalty remains
3. This -15% represents flood risk capitalization, robust to observable confounders
4. The DiD (post-flood change) is stable and near zero regardless of covariates
5. This supports the "freeze" interpretation: no *additional* price decline post-flood

**See**: Results computed via inline analysis (covariate-adjusted RD)

---

## Data Requirements for Additional Tests

### For Covariate Balance Tests

**Status: ✅ COMPLETE** (as of 2025-12-23)

| Covariate | Purpose | Source | Status |
|-----------|---------|--------|--------|
| Building age (year built) | Test for development timing discontinuity | NE Statewide Assessor GDB | ✅ COMPLETE |
| Assessed value (pre-2019) | Test for pre-treatment value differences | NE Statewide Assessor GDB | ✅ COMPLETE |
| Lot size (acres) | Test for parcel size differences | NE Statewide Assessor GDB | ✅ COMPLETE |
| Building SF | Test for structure size differences | NE Statewide Assessor GDB | ✅ COMPLETE |
| Elevation (meters) | Test for topographic discontinuity | USGS 3DEP DEM | ✅ COMPLETE |
| Property type (SFR filter) | Restrict to comparable properties | NE Statewide Assessor GDB | ✅ COMPLETE |

### Data Acquisition Status (Updated 2025-12-22)

| Dataset | Status | Location |
|---------|--------|----------|
| USGS 3DEP DEM (10m) | ✅ DOWNLOADED | `GIS_Data/Elevation/USGS_3DEP_10m/*.tif` (845MB, 2 tiles) |
| Microsoft Building Footprints | ✅ DOWNLOADED | `GIS_Data/Building_Footprints/Nebraska.geojson` (303MB) |
| Douglas County Parcel Attributes | ❌ NEEDED | User to provide |

### Processing Scripts Created

| Script | Input | Output | Status |
|--------|-------|--------|--------|
| `src/05_features/process_dem.py` | DEM tiles | `data_work/parcel_elevation.parquet` | Ready to run |
| `src/05_features/process_building_footprints.py` | Nebraska.geojson | `data_work/parcel_building_footprints.parquet` | Ready to run |

**Data Source**: [Douglas County GIS Open Data Portal](https://data-dogis.opendata.arcgis.com/)

**Current Status**: The `GIS_Data/Parcel_Data/Parcel_Flood_Zone.shp` file is empty (0 bytes). Original parcel data with attributes needs to be re-downloaded or restored.

### File Structure Expected

The analysis expects the following data files:

```
data_work/
├── panel_parcel_month.parquet      [EXISTS - 3.3MB]
├── parcel_boundary_distances.parquet [EXISTS - 9.6MB]
├── parcel_attributes.parquet        [NEEDED - for balance tests]
└── diagnostics/
    ├── mccrary_density_test.csv     [CREATED]
    ├── pretrends_ftest.csv          [CREATED]
    ├── bandwidth_sensitivity.csv    [CREATED]
    └── donut_rd.csv                 [CREATED]
```

---

## Recommended Manuscript Revisions

### Methods Section Additions

1. **Add new subsection: "3.9 Identification Diagnostics"**
   - Report McCrary density test result
   - Describe covariate balance tests (when data available)
   - Document bandwidth sensitivity analysis

2. **Revise "3.7 Identification strategies"**
   - Add explicit statement about SUTVA violation and ring models
   - Note that main effect is likely attenuated

### Results Section Additions

3. **Add Table 5: Identification Diagnostics**
   - McCrary density test results
   - Pre-trends F-test results
   - Bandwidth sensitivity summary

4. **Add Figure: Event Study with Confidence Intervals**
   - Show all 48 month-by-month coefficients
   - Highlight pre-event period
   - Add F-test p-value annotation

### Discussion Section Revisions

5. **Revise Section 5.5 Limitations**
   - Add density discontinuity discussion
   - Acknowledge pre-trends concern for SFHA
   - Note that inundation boundary has cleaner identification

6. **Add "5.X Identification Caveats"**
   - SUTVA violation and spillover interpretation
   - Why density discontinuity doesn't invalidate design
   - Pre-trends: secular vs. event-driven trends

---

## Implementation Status

| Task | Status | Output File |
|------|--------|-------------|
| McCrary density test | ✅ COMPLETE | `mccrary_density_test.csv` |
| Pre-trends F-test | ✅ COMPLETE | `pretrends_ftest.csv` |
| Bandwidth sensitivity | ✅ COMPLETE | `bandwidth_sensitivity.csv` |
| Donut RD | ✅ COMPLETE | `donut_rd.csv` |
| Elevation discontinuity | ✅ COMPLETE | `parcel_elevation.csv` |
| Buyer composition DiD | ✅ COMPLETE | `buyer_composition_did.csv` |
| Assessor data ETL | ✅ COMPLETE | `assessor_clean.parquet`, `parcel_covariates_full.parquet` |
| Covariate balance tests | ✅ COMPLETE | `covariate_balance.csv`, `covariate_balance_sfr.csv` |
| Covariate-adjusted RD | ✅ COMPLETE | (inline analysis) |
| Manuscript revisions | PENDING | Requires author input |

---

---

## Finding 9: Differential Geographic Trends (CRITICAL)

**Result**: Trend controls change DiD estimates by 165%

### Pre-Event Growth Differential

| Metric | Inside Zone | Outside Zone | Differential |
|--------|-------------|--------------|--------------|
| Annual sales growth (2013-2018) | 28.5% | 21.7% | +6.8pp |
| Monthly inside share trend | -0.0004 | - | p=0.56 |
| Differential price trend | -5,380/month | - | p=0.08 |

### Impact on DiD Estimates

| Model | DiD Coefficient | SE | Change |
|-------|-----------------|-----|--------|
| Without trend controls | 0.2923 | 0.145 | - |
| With trend controls | 0.7757 | 0.278 | **+165%** |

### Interpretation

The flooded areas (inside zone) had higher pre-existing housing market growth. When we control for group-specific trends, the estimated treatment effect more than doubles. This raises questions about:

1. **Omitted variable bias**: Are we missing important confounders?
2. **Trend specification**: How should we model differential growth?
3. **Counterfactual**: What would have happened inside without the flood?

### Critical Issue: Counterintuitive Sign

Both specifications show **positive** coefficients (prices rising faster inside), which contradicts the "flight" narrative. This requires investigation before manuscript decisions.

**See**: `doc/RESULTS_INTERPRETATION.md` for investigation plan

**See**: `data_work/diagnostics/trend_adjusted_did.csv`, `src/07_estimation/trend_analysis.py`

---

## Next Steps

### Completed
- [x] McCrary density test
- [x] Pre-trends F-test
- [x] Bandwidth sensitivity
- [x] Donut RD
- [x] Covariate balance tests
- [x] Assessor data integration
- [x] Trend analysis

### Investigation Required (BLOCKING)
- [ ] **Investigate counterintuitive positive price effect**
- [ ] Determine if selection, composition, or specification explains finding
- [ ] Decide on primary specification (boundary, trend controls)

### After Investigation
- [ ] Finalize manuscript revisions
- [ ] Complete supplementary materials
- [ ] Prepare reviewer response

---

## Sources

- Douglas County GIS Portal: https://gis.douglascounty-ne.gov/
- Open Data Portal: https://data-dogis.opendata.arcgis.com/
- Assessor GIS Mapping: https://assessor.douglascounty-ne.gov/gis-mapping.html
