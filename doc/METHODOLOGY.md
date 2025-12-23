# Methodology Documentation

This document describes the statistical methods and identification strategies used in the Freeze and Flight project.

---

## Research Design Overview

The study employs a **boundary regression discontinuity in panel (RD-in-panel)** design to estimate the causal effect of the March 2019 flood on housing market outcomes at regulatory flood boundaries.

### Key Features

1. **Spatial discontinuity**: SFHA boundary creates sharp regulatory threshold
2. **Temporal variation**: Pre/post flood comparison
3. **Local counterfactual**: Parcels just outside boundary serve as controls

---

## Boundary Definitions

### FEMA Special Flood Hazard Area (SFHA)

The SFHA boundary is derived from FEMA's National Flood Hazard Layer (NFHL).

**Included Zones**:
- Zone A: Areas subject to 1% annual chance flooding (no BFE determined)
- Zone AE: Areas subject to 1% annual chance flooding (BFE determined)
- Zone AO: Areas subject to 1% annual chance shallow flooding
- Zone AH: Areas subject to 1% annual chance shallow flooding (ponding)

**Assignment Rule**: Parcels are classified as "inside SFHA" using a majority-area rule (parcel centroid inside SFHA polygon). Sensitivity analysis uses 10% overlap threshold.

### 2019 Inundation Boundary

The realized flood extent is derived from Sentinel-2 satellite imagery captured during the March 2019 Missouri River flood event.

**Processing**:
1. Water detection using NDWI (Normalized Difference Water Index)
2. Binary mask creation at 10m resolution
3. Vectorization to polygon boundary
4. Assignment rule: ≥5% parcel overlap with inundation mask

---

## Identification Strategies

### 1. Boundary RD-in-Panel (Primary Design)

**Estimating Equation**:

For sale incidence (parcel-month panel):

```
Pr(Sale_it = 1) = Σ_{τ≠-1} β_τ [1{Inside_i} × 1{t=τ}] + α_i + γ_n(i),t + ε_it
```

For log prices (sale-level):

```
log(Price_it) = Σ_{τ≠-1} β_τ [1{Inside_i} × 1{t=τ}] + α_i + γ_n(i),t + ε_it
```

**Components**:
- `Inside_i`: Binary indicator for parcel i being inside the SFHA/inundation boundary
- `1{t=τ}`: Event-time indicator (τ = months relative to March 2019)
- `α_i`: Parcel fixed effects
- `γ_n(i),t`: Neighborhood × month fixed effects
- `β_τ`: Treatment effects by event time (τ = -1 is reference)

**Caliper Windows**:
- Primary: ±300m from boundary
- Sensitivity: ±150m from boundary

The narrow caliper ensures spatial comparability: parcels on either side of the boundary are similar in location-based attributes.

### 2. Near-but-Dry Ring Models (Substitution Test)

To test whether transactions reroute to parcels just outside the boundary, we estimate Poisson count models on microcell-level sales.

**Estimating Equation**:

```
log E[y_ct] = α_c + δ_t + θ_1 [1{Ring_0-250m} × 1{Post}] + θ_2 [1{Ring_250-300m} × 1{Post}]
```

**Components**:
- `y_ct`: Count of sales in microcell c at month t
- `α_c`: Cell fixed effects
- `δ_t`: Month fixed effects
- `Ring_0-250m`: Binary for cells 0-250m outside the SFHA line
- `Ring_250-300m`: Binary for cells 250-300m outside the SFHA line
- `Post`: Binary for post-flood period (March 2019 onward)

**Interpretation**: Positive θ coefficients indicate increased sales activity in near-but-dry rings after the flood, consistent with demand substitution away from inside the SFHA.

### 3. Composition Analysis

To examine changes in buyer characteristics, we estimate:

```
Y_it = β [Inside_i × Post_t] + α_i + γ_n(i),t + ε_it
```

Where Y is:
- Share of purchases by LLCs
- Share of purchases by multi-parcel owners
- Mean buyer-parcel distance
- Cash purchase share

---

## Fixed Effects Structure

### Parcel Fixed Effects (α_i)

Control for all time-invariant parcel characteristics:
- Location (latitude, longitude)
- Lot size
- Building characteristics
- Neighborhood amenities

### Neighborhood × Month Fixed Effects (γ_n(i),t)

Control for:
- County-wide housing market trends
- Neighborhood-specific seasonal patterns
- Local economic shocks

**Definition**: Neighborhoods are defined using assessor appraisal neighborhoods (administrative units used for property valuation).

---

## Standard Errors

### Primary: Neighborhood Clustering

Standard errors are clustered at the neighborhood level to account for:
- Spatial correlation in outcomes within neighborhoods
- Serial correlation in parcel outcomes over time

### Sensitivity: Conley Spatial HAC

For robustness, we compute Conley (1999) heteroskedasticity and autocorrelation consistent (HAC) standard errors that allow for arbitrary spatial correlation within a distance kernel.

**Parameters**:
- Distance cutoff: 1km (results robust to 500m and 2km)
- Kernel: Uniform (Bartlett kernel as sensitivity)

---

## Sample Restrictions

### Parcel Universe

- **Land use**: Single-family residential (SFR) parcels
- **Geography**: Douglas County, Nebraska
- **Time period**: March 2017 - March 2021 (±24 months from event)

### Sales Sample

Arms-length transaction filters:
1. Sale price ≥ $1,000 (exclude nominal transfers)
2. Exclude deed types: quitclaim, sheriff, foreclosure, gift
3. Exclude intra-family transfers (name matching)
4. Exclude intra-entity transfers (same buyer/seller)

### Boundary Windows

For RD analysis, restrict to parcels within caliper distance of boundary:
- ±300m window: ~1.8M parcel-months, ~5,000 sales
- ±150m window: ~820K parcel-months, ~2,200 sales

---

## Robustness Checks

1. **Caliper sensitivity**: ±150m vs. ±300m windows
2. **SFHA definition**: Majority-area vs. 10% overlap threshold
3. **Ring width**: 250m vs. 200m vs. 300m rings
4. **Placebo event**: Re-estimate with false event dates (March 2018, 2020)
5. **Repeat-sales residuals**: Price models using within-parcel variation only
6. **Spatial HAC**: Conley standard errors with varying distance cutoffs

---

## Key Assumptions

### 1. Parallel Trends

Parcels inside and outside the boundary would have followed similar trajectories absent the flood. Supported by:
- Pre-event coefficient estimates (β_τ for τ < 0) close to zero
- Similar pre-event sale rates in treatment and control groups

### 2. No Manipulation at Boundary

Parcels cannot sort to either side of the SFHA boundary in the short run. Supported by:
- SFHA boundaries set by federal mapping (not local choice)
- Parcel boundaries fixed

### 3. SUTVA

No spillovers across boundary. Potential violation if:
- Demand substitution affects "control" parcels just outside
- Addressed by explicitly modeling ring effects

### 4. Exclusion Restriction

The SFHA boundary affects outcomes only through flood risk salience, not through other channels. Note that:
- Insurance requirements apply inside SFHA (potential confound)
- Disclosure requirements may vary
