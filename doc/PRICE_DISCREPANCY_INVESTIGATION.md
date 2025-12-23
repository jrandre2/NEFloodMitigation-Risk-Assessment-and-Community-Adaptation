# Price Effect Discrepancy Investigation

**Date**: December 23, 2025
**Issue**: Manuscript claims +0.528 positive log-price DiD but pipeline outputs show -0.26

## Summary

The manuscript's claim of a "counterintuitive positive price effect" is **incorrect** for the SFR sample. After investigation, I found:

| Sample | N Sales | DiD Estimate | Sign |
|--------|---------|--------------|------|
| All properties (incl. non-SFR) | 739 | **+0.578** | Positive |
| SFR only (panel sample) | 387 | **-0.258** | Negative |

The +0.528 in the manuscript came from an earlier analysis that included non-SFR properties. The current pipeline correctly restricts to single-family residential (SFR) properties.

## Root Cause

1. **Panel construction** (`src/06_panels/build_panels.py`) uses `parcels_sfr.gpkg` as the base, which contains only SFR properties.

2. **352 non-SFR sales** in the inundation ±300m window are excluded from the panel but were included in earlier analyses.

3. These non-SFR properties are:
   - Disproportionately **inside** the inundation zone (34.9% vs 9.0% for SFR)
   - Have **lower prices** on average (log_price 12.36 vs 12.67)
   - Show large price increases post-flood

4. Including these non-SFR sales creates a positive DiD; excluding them (correctly) produces a negative DiD.

## Implications for the Manuscript

### The "Positive Price Effect" Narrative is Wrong

The manuscript's central claim about prices is:

> "contrary to standard predictions, observed transaction prices *increase* inside the flood zone"

This is **not true for SFR properties**. For the SFR sample:
- Inside prices decreased relative to outside
- This is **consistent with** standard flood risk capitalization theory
- The DiD is -0.26 log points (negative, though not statistically significant)

### What the Data Actually Shows

For SFR properties at the inundation boundary (±300m):

| Location | Pre-Flood Mean Log Price | Post-Flood Mean Log Price | N Pre | N Post |
|----------|-------------------------|--------------------------|-------|--------|
| Inside | 12.51 | 12.38 | 12 | 23 |
| Outside | 12.63 | 12.76 | 165 | 187 |

- Inside prices **fell** slightly (12.51 → 12.38)
- Outside prices **rose** (12.63 → 12.76)
- DiD = (12.38 - 12.51) - (12.76 - 12.63) = -0.13 - 0.13 = **-0.26**

### Revised Narrative Options

1. **Option A: Report the truth**
   - SFR prices show negative (though imprecise) effects
   - This is consistent with standard theory
   - "Freeze" still holds; "Rebuild" narrative needs revision

2. **Option B: Investigate non-SFR separately**
   - Report SFR as main result
   - Show non-SFR properties separately (may include commercial, multi-family)
   - Different dynamics for different property types

3. **Option C: Focus on composition effects**
   - The composition shift analysis still holds
   - Post-flood sales are systematically different
   - But the direction of the price effect changes the interpretation

## Recommended Actions

1. **Correct all price effect claims** in manuscript
   - Replace +0.528 with -0.26 (or current pipeline output)
   - Remove "counterintuitive positive price effect" language
   - The price effect is negative and imprecise, not positive

2. **Update the "Freeze and Rebuild" framing**
   - "Freeze" (liquidity decline) still holds
   - "Rebuild" narrative needs evidence beyond price effects
   - Consider focusing on building permits or assessor value changes

3. **Add sample definition clarity**
   - Explicitly state "SFR properties only"
   - Document why non-SFR excluded

4. **Consider heterogeneity by property type**
   - SFR vs non-SFR may have different dynamics
   - Could be interesting additional analysis

## Files Updated

- Current pipeline outputs in `data_work/tab_rd_price_summary.csv` show correct SFR-only estimates
- Manuscript needs comprehensive rewrite to reflect actual findings

## Verification Commands

```bash
# Verify SFR-only DiD
source .venv/bin/activate
python src/pipeline.py rd_summary -b inund -c 300

# Check output
cat data_work/tab_rd_price_summary.csv
```

Expected: DiD around -0.26 at inundation 300m boundary.
