# Manuscript Revision Checklist

## Status: BLOCKED - Awaiting Investigation Results

**Important**: Several checklist items are blocked pending investigation of the counterintuitive positive price effect finding. See `RESULTS_INTERPRETATION.md` for details.

---

## Pre-Revision Investigation Required

Before proceeding with manuscript revisions, the following must be resolved:

- [ ] **Investigate positive price effect**: Why did prices rise faster inside the flood zone?
- [ ] **Determine primary specification**: Inundation (clean ID) vs SFHA (expected sign)
- [ ] **Decide on trend controls**: Include in main spec or robustness only?
- [ ] **Reconcile narrative**: "Flight" implies price decline, but evidence shows increase

---

## Section 3: Methods

### 3.7 Identification Strategies
- [ ] Add explicit statement about SUTVA violation and ring models
- [ ] Note that main effect may be attenuated due to spillovers
- [ ] Reference donut RD as robustness check

### 3.9 Identification Diagnostics (NEW SUBSECTION)
- [ ] Report McCrary density test result and interpretation
- [ ] Document pre-trends F-test for both boundaries
- [ ] Describe bandwidth sensitivity analysis
- [ ] Add covariate balance discussion
- [ ] Reference supplementary materials for full results

### 3.X Trend Analysis (NEW SUBSECTION - if trends in main spec)
- [ ] Describe rationale for trend controls
- [ ] Specify trend-break model
- [ ] Document group-specific trend methodology

---

## Section 4: Results

### Tables to Add/Revise

- [ ] **Table X: Identification Diagnostics Summary**
  - McCrary test results (both boundaries)
  - Pre-trends F-test results
  - Bandwidth sensitivity summary
  - Covariate balance summary

- [ ] **Table X: Main DiD Results** (PENDING INVESTIGATION)
  - Which boundary to feature?
  - With or without trend controls?

- [ ] **Table X: Robustness Specifications**
  - Alternative boundaries
  - Alternative calipers
  - Trend-adjusted estimates
  - Donut RD results

### Figures to Add

- [ ] **Figure X: Event Study with Confidence Intervals**
  - Show all 48 month-by-month coefficients
  - Highlight pre-event period
  - Add F-test p-value annotation
  - Generate for primary boundary

- [ ] **Figure X: Trend Analysis** (if including trends)
  - Inside vs outside sales trends over time
  - Pre/post trend break visualization

---

## Section 5: Discussion

### 5.X Identification Caveats (NEW SUBSECTION)
- [ ] SUTVA violation and spillover interpretation
- [ ] Why density discontinuity doesn't invalidate design
- [ ] Pre-trends: secular vs. event-driven trends
- [ ] Discussion of differential geographic trends

### 5.5 Limitations (REVISE)
- [ ] Add density discontinuity discussion
- [ ] Acknowledge pre-trends concern for SFHA
- [ ] Note differential trend confounding potential
- [ ] Discuss counterintuitive findings (PENDING INVESTIGATION)

### 5.X Mechanisms (REVISE if needed)
- [ ] Insurance mandate evidence
- [ ] Credit constraints evidence
- [ ] Buyer composition findings (LLCs decreased, not increased)

---

## Supplementary Materials

### Appendix A: Data and Methods
- [ ] Full variable definitions
- [ ] Sample construction details
- [ ] Boundary definitions with maps

### Appendix B: Identification Diagnostics
- [ ] Complete McCrary test results and figures
- [ ] Full bandwidth sensitivity table
- [ ] Covariate balance tables
- [ ] Elevation discontinuity analysis

### Appendix C: Robustness Checks
- [ ] All caliper specifications
- [ ] Donut RD results
- [ ] Placebo test results (fake events, shifted boundaries)
- [ ] Permutation inference results
- [ ] Leave-one-out analysis

### Appendix D: Trend Analysis
- [ ] Differential pre-trends tests
- [ ] Trend-break model results
- [ ] Comparison with/without trend controls
- [ ] Long-run trend visualization

### Appendix E: Selection and Heterogeneity
- [ ] IPW results
- [ ] Lee bounds
- [ ] Quantile treatment effects
- [ ] Heterogeneity by property characteristics

### Appendix F: Mechanism Analysis
- [ ] Insurance channel (SFHA vs inundation)
- [ ] Credit constraints (by property value)
- [ ] Buyer composition analysis

---

## Response to Reviewers

### Key Points to Address

1. **Parallel Trends**:
   - SFHA fails pre-trends test (p=0.026)
   - Inundation passes (p=0.78)
   - Discuss implications for interpretation

2. **Identification Strategy**:
   - McCrary discontinuity reflects topography, not manipulation
   - Covariate imbalance is expected given boundary definition
   - Donut RD addresses spillover concerns

3. **Robustness**:
   - Results stable across bandwidths
   - Placebo tests support causal interpretation
   - Selection correction (Lee bounds) still meaningful

4. **Mechanism Evidence**:
   - Insurance mandate supported by SFHA-Inund comparison
   - Credit constraints supported by value heterogeneity
   - Buyer composition shifts documented

---

## Files to Reference

| Document | Location | Purpose |
|----------|----------|---------|
| ANALYSIS_SUMMARY.md | `doc/` | Current findings overview |
| CRITIQUE_FINDINGS.md | `doc/` | Detailed methodological audit |
| RESULTS_INTERPRETATION.md | `doc/` | Guidance on interpreting results |
| METHODOLOGY.md | `doc/` | Statistical methods |

---

## Timeline Notes

- **Journal**: Environment and Planning B (EPB-2025-0878)
- **Revision deadline**: March 16, 2026
- **Current status**: Analysis complete, investigation needed

---

*Last updated: 2025-12-23*
*Status: BLOCKED pending investigation of counterintuitive findings*
