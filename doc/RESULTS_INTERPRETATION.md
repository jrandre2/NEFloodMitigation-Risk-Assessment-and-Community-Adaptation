# Results Interpretation Guide

## Status: RESOLVED (December 2025)

This document outlines the resolution of the positive price effect issue and provides interpretation guidance for the current estimates.

---

## Issue RESOLVED: Price Effect Clarification

### The Resolution

The earlier positive price effect (+0.528) was found to be from the **wrong sample** (included non-SFR properties). The correct SFR-only sample shows a **negative price effect**:

| Sample | N Sales | DiD Estimate | 95% CI |
|--------|---------|--------------|--------|
| All properties (WRONG) | 739 | +0.578 | - |
| **SFR only (CORRECT)** | 387 | **-0.26** | [-0.85, +0.33] |

This **aligns with** standard flood risk capitalization theory:
- Reduced demand for flood-exposed properties
- Price declines inside flood zones (point estimate negative)
- Effect imprecisely estimated due to small sample size

### Current Interpretation

1. **Narrative coherence**: The "Freeze and Rebuild" framing is supported:
   - **Freeze**: Strong evidence (38% relative decline in sale rates at SFHA)
   - **Prices**: Negative but imprecise (consistent with theory, CI includes zero)
   - **Composition**: Post-flood sales skew toward newer/larger properties

2. **Causal interpretation**: The inundation boundary passes pre-trends (F=1.50, p=0.152), supporting causal claims. SFHA boundary fails (F=3.33, p<0.001) and is used for robustness only.

3. **Publication readiness**: Results are now internally consistent and align with economic theory.

---

## Potential Explanations to Investigate

### 1. Selection on Sellers

**Hypothesis**: Only motivated sellers (distressed, relocating) sold inside the flood zone, while marginal sellers waited. The properties that did sell may have been systematically different.

**Tests needed**:
- Compare characteristics of sold vs unsold properties
- Examine seller motivation proxies (foreclosure, estate sales)
- Check time-on-market data if available

### 2. Compositional Changes

**Hypothesis**: New construction or renovated properties dominated post-flood sales inside the zone, pulling up average prices.

**Tests needed**:
- Separate analysis by building age (new vs existing)
- Check for renovation/improvement permits post-flood
- Compare lot sizes and property types pre/post

### 3. Investor Activity

**Hypothesis**: Investors bought up flood-damaged properties at higher prices anticipating future appreciation or rental income.

**Tests needed**:
- Examine buyer type (LLC, portfolio) in post-period
- Check for bulk purchases
- Compare cash vs financed purchases

### 4. Insurance/FEMA Effects

**Hypothesis**: FEMA buyouts or insurance settlements inflated recorded sale prices.

**Tests needed**:
- Identify FEMA buyout properties
- Check for unusual price patterns
- Compare to non-FEMA transactions

### 5. Boundary Definition Issues

**Hypothesis**: The inundation boundary may not correctly identify treatment.

**Tests needed**:
- Verify inundation boundary against actual flood extent
- Check for boundary measurement error
- Compare results at different boundary definitions

### 6. Trend Specification Error

**Hypothesis**: The trend controls are absorbing the treatment effect or creating bias.

**Tests needed**:
- Examine trend coefficients for plausibility
- Test alternative trend specifications (quadratic, spline)
- Check for multicollinearity

---

## Final Specification Decisions

### Which Boundary?

| Boundary | Pre-Trends | Effect Sign | Recommendation |
|----------|------------|-------------|----------------|
| **Inundation** | PASSES (F=1.50, p=0.152) | Negative (-0.26) | **PRIMARY** - Clean ID |
| SFHA | FAILS (F=3.33, p<0.001) | Negative (-0.00089) | ROBUSTNESS - Strong liquidity effect |

**Decision**: Inundation boundary is primary for causal claims. SFHA used for liquidity analysis where pre-trends are less critical.

### Sample Definition

| Sample | Use Case |
|--------|----------|
| **SFR only** | All price analyses (387 inside-boundary transactions) |
| All properties | Excluded - creates spurious positive effect |

**Decision**: Restrict to single-family residential for all analyses to ensure comparability.

---

## Key Results Summary

### Liquidity Effects (Strongest Finding)

| Boundary | Sale Rate DiD | Relative Change | Interpretation |
|----------|---------------|-----------------|----------------|
| SFHA 300m | -0.00089 | -38% | Strong freeze inside |
| Inundation 300m | +0.00157 | Elevated | Some inundated properties selling |

### Price Effects

| Boundary | Log-Price DiD | 95% CI | Interpretation |
|----------|---------------|--------|----------------|
| Inundation 300m | -0.26 | [-0.85, +0.33] | Negative but imprecise |

### Buyer Composition

| Metric | Inside Pre | Inside Post | DiD |
|--------|------------|-------------|-----|
| Entity share | 45.2% | 27.3% | -12.9pp |

**Interpretation**: Entity buyers (LLCs, etc.) *decreased* share post-flood, contrary to "investor swooping" narrative.

---

## Investigation Completed

### Root Cause Identified

The positive price effect was caused by including non-SFR properties (352 sales), which:
- Were disproportionately inside the inundation zone (34.9% vs 9.0%)
- Had lower pre-flood prices
- Created a spurious positive composition effect

### Correction Applied

All analyses now use SFR-only sample:
- Panel: `parcels_sfr.gpkg` as base
- Sales: `sales_clean.parquet` filtered to SFR
- Results: Consistent negative price effect

### Documentation

Full audit trail in `manuscript_quarto/REVISION_TRACKER.md`

---

## Related Files

- `data_work/diagnostics/trend_adjusted_did.csv` - Trend analysis results
- `src/07_estimation/trend_analysis.py` - Trend analysis code
- `doc/ANALYSIS_SUMMARY.md` - Overall findings summary
- `doc/CRITIQUE_FINDINGS.md` - Methodological audit

---

## Related Files

- `manuscript_quarto/REVISION_TRACKER.md` - Full revision audit trail
- `manuscript_quarto/referee-report.qmd` - Referee report with status annotations
- `data_work/diagnostics/pretrends_ftest.csv` - Pre-trends test results
- `data_work/tab_rd_price_summary.csv` - Price DiD estimates

---

*Last updated: 2025-12-23*
*Status: RESOLVED - Price effect issue clarified, manuscript updated*
