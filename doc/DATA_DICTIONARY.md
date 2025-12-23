# Data Dictionary

This document defines the key variables used in the Freeze and Flight analysis.

---

## Parcel-Level Variables

### Identifiers

| Variable | Type | Description |
|----------|------|-------------|
| `parcel_id` | string | Unique parcel identifier (APN) |
| `objectid` | integer | GIS feature ID |
| `neighborhood_id` | string | Assessor appraisal neighborhood code |

### Geographic Attributes

| Variable | Type | Unit | Description |
|----------|------|------|-------------|
| `centroid_x` | float | meters | Parcel centroid X coordinate (projected) |
| `centroid_y` | float | meters | Parcel centroid Y coordinate (projected) |
| `centroid_lat` | float | degrees | Parcel centroid latitude (WGS84) |
| `centroid_lon` | float | degrees | Parcel centroid longitude (WGS84) |
| `parcel_area_sqm` | float | m² | Parcel area in square meters |

### Property Attributes

| Variable | Type | Description |
|----------|------|-------------|
| `land_use` | string | Land use classification code |
| `is_sfr` | boolean | Single-family residential indicator |
| `building_age` | integer | Years since construction |
| `total_assessed_value` | float | Total assessed property value (USD) |
| `has_building` | boolean | Improved parcel indicator |

---

## Exposure Variables

### SFHA Exposure

| Variable | Type | Unit | Description |
|----------|------|------|-------------|
| `signed_dist_sfha_m` | float | meters | Signed Euclidean distance to nearest SFHA boundary. **Negative values = inside SFHA**, positive = outside |
| `inside_sfha` | boolean | - | Binary indicator: parcel centroid inside SFHA |
| `sfha_zone` | string | - | FEMA flood zone code (A, AE, AO, AH, X) |
| `sfha_overlap_pct` | float | % | Percent of parcel area overlapping SFHA |

### 2019 Inundation Exposure

| Variable | Type | Unit | Description |
|----------|------|------|-------------|
| `signed_dist_inund_m` | float | meters | Signed distance to 2019 inundation boundary. **Negative = inundated**, positive = not inundated |
| `inside_inund` | boolean | - | Binary indicator: parcel was inundated in March 2019 |
| `inund_overlap_pct` | float | % | Percent of parcel area inundated |

### RD Window Indicators

| Variable | Type | Description |
|----------|------|-------------|
| `rd_sfha_150` | boolean | Parcel within ±150m of SFHA boundary |
| `rd_sfha_300` | boolean | Parcel within ±300m of SFHA boundary |
| `rd_inund_150` | boolean | Parcel within ±150m of inundation boundary |
| `rd_inund_300` | boolean | Parcel within ±300m of inundation boundary |

### Ring Variables (for Poisson models)

| Variable | Type | Description |
|----------|------|-------------|
| `sfha_ring` | string | Distance ring category for parcels outside SFHA: `0_250m`, `250_500m`, `500_1000m`, `gt_1000m` |
| `inund_ring` | string | Distance ring category for parcels outside 2019 inundation: `0_250m`, `250_500m`, `500_1000m`, `gt_1000m` |

**Ring Categories**:
- `0_250m` - Near-but-dry: 0-250m outside boundary
- `250_500m` - 250-500m outside boundary
- `500_1000m` - 500-1000m outside boundary
- `gt_1000m` - Greater than 1000m outside boundary

---

## Sales Transaction Variables

### Transaction Identifiers

| Variable | Type | Description |
|----------|------|-------------|
| `sale_id` | string | Unique transaction identifier |
| `parcel_id` | string | Associated parcel identifier |
| `deed_book` | string | Recording book number |
| `deed_page` | string | Recording page number |

### Transaction Details

| Variable | Type | Unit | Description |
|----------|------|------|-------------|
| `sale_date` | date | - | Transaction recording date |
| `sale_price` | float | USD | Transaction price |
| `log_price` | float | log(USD) | Natural log of sale price |
| `deed_type` | string | - | Deed instrument type |
| `arms_length` | boolean | - | Arms-length transaction indicator |

### Party Information

| Variable | Type | Description |
|----------|------|-------------|
| `buyer_name` | string | Buyer name (cleaned) |
| `seller_name` | string | Seller name (cleaned) |
| `buyer_address` | string | Buyer mailing address |
| `buyer_zip` | string | Buyer mailing ZIP code |

---

## Owner Classification Variables

### Organizational Form

| Variable | Type | Description |
|----------|------|-------------|
| `owner_form` | string | Owner type classification |
| `owner_form_confidence` | float | Classifier confidence score (0-1) |

**Owner Form Categories**:
- `Individual` - Natural person(s)
- `LLC` - Limited Liability Company
- `Corporation` - Corporation or Inc.
- `Trust` - Trust or estate
- `Other` - Government, nonprofit, or other entity

### Portfolio Scale

| Variable | Type | Description |
|----------|------|-------------|
| `is_single_parcel` | boolean | Owner holds only one parcel (at time of transaction) |
| `is_multi_parcel` | boolean | Owner holds multiple parcels |
| `portfolio_size` | integer | Number of parcels held by owner |

### Owner Proximity

| Variable | Type | Unit | Description |
|----------|------|------|-------------|
| `owner_dist_km` | float | km | Great-circle distance from owner mailing ZIP centroid to parcel centroid |
| `log_owner_dist_km` | float | log(km) | Natural log of owner distance |
| `is_local_owner` | boolean | - | Owner mailing ZIP = parcel ZIP |
| `is_in_county` | boolean | - | Owner in Douglas County |
| `is_in_state` | boolean | - | Owner in Nebraska |
| `is_out_of_state` | boolean | - | Owner outside Nebraska |

**Distance Bands**:
| Variable | Description |
|----------|-------------|
| `dist_band_same_zip` | Owner in same ZIP as parcel |
| `dist_band_other_douglas` | Owner in different Douglas County ZIP |
| `dist_band_adjoining` | Owner in adjoining county |
| `dist_band_other_ne` | Owner in other Nebraska county |
| `dist_band_out_of_state` | Owner outside Nebraska |

---

## Panel Variables

### Time Variables

| Variable | Type | Description |
|----------|------|-------------|
| `ym` | string | Year-month in YYYY-MM format (e.g., "2019-03") |
| `year_month` | string | Alias for `ym` (YYYY-MM format) |
| `event_m` | integer | Months relative to March 2019 flood. Range: [-24, +24]. event_m=0 is March 2019. |
| `event_time` | integer | Alias for `event_m` |
| `post` | boolean | Post-flood period indicator (event_m ≥ 0) |

### Outcome Variables (Parcel-Month Panel)

| Variable | Type | Description |
|----------|------|-------------|
| `sold_this_month` | boolean | Sale recorded in this parcel-month (primary outcome) |
| `sale_occurred` | boolean | Alias for `sold_this_month` |
| `log_price` | float | Natural log of sale price (if sale occurred, otherwise NaN) |

### Outcome Variables (Microcell Panel)

| Variable | Type | Description |
|----------|------|-------------|
| `cell_id` | string | Microcell identifier |
| `monthly_sales` | integer | Count of sales in cell-month |

### Extended Panel Variables (`panel_parcel_month_extended.parquet`)

Variables from the extended panel construction (Stage 06, `extend_panel.py`).

| Variable | Type | Description |
|----------|------|-------------|
| `ym` | string | Year-month (YYYY-MM format), range: 2017-03 to 2022-12 |
| `event_time` | integer | Months relative to March 2019, range: [-24, +45] |
| `post` | int (0/1) | Post-flood indicator (event_time >= 0) |
| `sold_this_month` | int (0/1) | Sale recorded in this parcel-month |
| `n_sales` | int | Number of sales in parcel-month (typically 0 or 1) |
| `log_price` | float | Natural log of median sale price (NaN if no sale) |

**COVID-19 Period Indicators**:

| Variable | Type | Period | Description |
|----------|------|--------|-------------|
| `covid_lockdown` | int (0/1) | 2020-03 to 2020-06 | Initial COVID lockdown period |
| `covid_period` | int (0/1) | 2020-03 to 2021-12 | Broader pandemic period |

**Extended Panel Statistics**:
| Metric | Value |
|--------|-------|
| Date range | 2017-03 to 2022-12 |
| Event time range | t-24 to t+45 |
| Total months | 70 |
| Parcels (RD scope) | 38,085 |
| Panel observations | 2,665,950 |
| Total sales | 8,313 |
| Pre-flood sales (t<0) | 2,368 |
| Post-flood sales (t>=0) | 5,945 |
| COVID period sales | 3,185 |

---

## NFIP Variables

Variables from National Flood Insurance Program data integration (Stage 00, `load_nfip.py`).

### Claims Data (`nfip_claims_nebraska.parquet`)

| Variable | Type | Description |
|----------|------|-------------|
| `censusTract` | string | 11-digit census tract ID |
| `countyCode` | string | 5-digit FIPS county code |
| `dateOfLoss` | date | Date of flood loss event |
| `yearOfLoss` | int | Year extracted from dateOfLoss |
| `amountPaidOnBuildingClaim` | float | Payment for building damage (USD) |
| `amountPaidOnContentsClaim` | float | Payment for contents damage (USD) |
| `totalBuildingInsuranceCoverage` | float | Total building coverage limit (USD) |
| `totalContentsInsuranceCoverage` | float | Total contents coverage limit (USD) |
| `waterDepth` | float | Reported water depth (feet, often missing) |

**Note**: NFIP lat/lon coordinates are obfuscated for privacy; analysis uses census tract aggregation.

### Tract Summary (`tract_nfip_summary.parquet`)

| Variable | Type | Description |
|----------|------|-------------|
| `censusTract` | string | 11-digit census tract ID |
| `n_claims_total` | int | Total claims in tract (all years) |
| `n_claims_2019` | int | Claims in 2019 flood year |
| `total_payment_total` | float | Total payments all years (USD) |
| `total_payment_2019` | float | Total payments in 2019 (USD) |
| `mean_water_depth_2019` | float | Average reported water depth in 2019 (feet) |

**Key Statistics (Nebraska)**:
| Metric | Value |
|--------|-------|
| Total claims (1978-2023) | 6,062 |
| Claims in 2019 | 818 (15.7× historical average) |
| Total payments (1978-2023) | $92.3M |
| Payments in 2019 | $37.5M (40.7% of total) |
| Unique tracts with 2019 claims | 155 |

---

## Buyer Distance Variables

Variables from owner address geocoding (Stage 05, `geocode_buyer_addresses.py`).

### Geocoded Addresses (`owner_addresses_geocoded.parquet`)

| Variable | Type | Description |
|----------|------|-------------|
| `parcel_id` | string | Parcel identifier |
| `input_address` | string | Original owner mailing address |
| `match_status` | string | Geocoding result: "Match", "No_Match", "Tie" |
| `match_type` | string | Match quality: "Exact", "Non_Exact" |
| `matched_address` | string | Standardized matched address |
| `coordinates` | string | Geocoded coordinates (lon,lat) |
| `lat` | float | Latitude of owner address (WGS84) |
| `lon` | float | Longitude of owner address (WGS84) |

### Owner Distances (`owner_distances.parquet`)

| Variable | Type | Unit | Description |
|----------|------|------|-------------|
| `parcel_id` | string | - | Parcel identifier |
| `owner_lat` | float | degrees | Owner address latitude |
| `owner_lon` | float | degrees | Owner address longitude |
| `parcel_lat` | float | degrees | Parcel centroid latitude |
| `parcel_lon` | float | degrees | Parcel centroid longitude |
| `distance_miles` | float | miles | Great-circle (haversine) distance owner to parcel |
| `distance_band` | string | - | Categorical distance band |
| `out_of_state` | int (0/1) | - | Owner outside Nebraska |

**Distance Bands**:
| Band | Description |
|------|-------------|
| `<10mi` | Local owner (within 10 miles) |
| `10-50mi` | Regional owner (10-50 miles) |
| `50-100mi` | In-state distant (50-100 miles) |
| `100-500mi` | Multi-state regional (100-500 miles) |
| `>500mi` | Distant/national investor (>500 miles) |

**Technical Notes**:
- Uses Census Geocoder batch API (free, rate-limited)
- Typical match rate: ~81%
- Distance calculated using haversine formula (Earth radius = 3,959 miles)

---

## Estimation Output Variables

### RD Summary Statistics

| Variable | Type | Description |
|----------|------|-------------|
| `rate_inside_pre` | float | Sale rate inside boundary, pre-flood |
| `rate_inside_post` | float | Sale rate inside boundary, post-flood |
| `rate_outside_pre` | float | Sale rate outside boundary, pre-flood |
| `rate_outside_post` | float | Sale rate outside boundary, post-flood |
| `did_estimate` | float | Difference-in-differences estimate |
| `did_se` | float | Standard error of DiD estimate |
| `did_ci_lower` | float | 95% CI lower bound |
| `did_ci_upper` | float | 95% CI upper bound |

### Poisson Ring Model Output

| Variable | Type | Description |
|----------|------|-------------|
| `ring` | string | Ring identifier (e.g., "0-250m") |
| `rate_ratio` | float | Exponentiated coefficient (incidence rate ratio) |
| `rate_ratio_se` | float | Standard error of rate ratio |
| `pvalue` | float | P-value for H0: RR=1 |

---

## Data Sources

| Variable Group | Source |
|----------------|--------|
| Parcel attributes | Douglas County Assessor GIS |
| Sales transactions | Douglas County Register of Deeds |
| SFHA boundaries | FEMA National Flood Hazard Layer (NFHL) |
| 2019 inundation | Sentinel-2 satellite imagery (derived) |
| Building footprints | Microsoft Building Footprints |
| ZIP centroids | US Census ZCTA centroids |

---

## Missing Data Conventions

| Code | Meaning |
|------|---------|
| `NaN` / `None` | Missing value |
| `-999` | Not applicable |
| `""` (empty string) | Unknown/not recorded |

---

## Elevation Variables

| Variable | Type | Unit | Description |
|----------|------|------|-------------|
| `elevation_m` | float | meters | Ground elevation at parcel centroid (from USGS 3DEP DEM) |

## Building Footprint Variables

| Variable | Type | Unit | Description |
|----------|------|------|-------------|
| `footprint_sqm` | float | m² | Total building footprint area on parcel (from Microsoft Building Footprints) |

---

## Assessor Covariate Variables

Variables derived from the Nebraska statewide assessor geodatabase (Stage 05b).

### Building Characteristics

| Variable | Type | Unit | Description |
|----------|------|------|-------------|
| `year_built` | int | year | Year structure was built (parsed from BuildingYear string) |
| `building_age` | int | years | Age of structure as of 2019 (2019 - year_built) |
| `ImpSF` | float | sq ft | Building square footage (improvements) |
| `log_impsf` | float | log(sq ft) | log1p(ImpSF) |
| `QualImp` | string | - | Improvement quality grade |
| `CondImp` | string | - | Improvement condition grade |

### Assessed Values

| Variable | Type | Unit | Description |
|----------|------|------|-------------|
| `Total_Assessed_Value` | float | USD | County assessed total value |
| `log_assessed_value` | float | log(USD) | log1p(Total_Assessed_Value) |
| `Land_Value` | float | USD | Assessed land value only |
| `log_land_value` | float | log(USD) | log1p(Land_Value) |
| `Improvements_Value` | float | USD | Assessed improvements value |
| `log_improvement_value` | float | log(USD) | log1p(Improvements_Value) |

### Parcel Characteristics

| Variable | Type | Unit | Description |
|----------|------|------|-------------|
| `GIS_Acres` | float | acres | Parcel size from GIS calculation |
| `log_acres` | float | log(acres) | log1p(GIS_Acres) |
| `Property_Parcel_Type` | int | code | Property type code (1=SFR, 2=Multi-Family, etc.) |
| `property_type_label` | string | - | Human-readable property type |
| `Zoning` | string | code | Zoning classification code |
| `zoning_label` | string | - | Human-readable zoning label |
| `neighborhood` | string | code | Assessor neighborhood code |

### Derived Indicators

| Variable | Type | Description |
|----------|------|-------------|
| `is_sfr` | int (0/1) | Single-family residential (Property_Parcel_Type = 1) |
| `is_improved` | int (0/1) | Parcel has improvements (Improvements_Value > 0 or ImpSF > 0) |

**Property Type Codes**:
| Code | Label |
|------|-------|
| 1 | Single Family Residential |
| 2 | Multi-Family Residential |
| 3 | Commercial |
| 4 | Industrial |
| 5 | Agricultural |
| 6 | Vacant |
| 7 | Exempt |
| 8 | Other |

**Data Coverage** (Douglas County):
- Total parcels: 212,314
- year_built coverage: 88.1%
- Assessed value coverage: 100%
- SFR parcels: 170,143 (80.2%)

---

## Diagnostic Output Variables

Variables from identification diagnostic tests (Stage 07b).

### McCrary Density Test (`mccrary_density_test.csv`)

| Variable | Type | Description |
|----------|------|-------------|
| `boundary` | string | Boundary tested (sfha or inund) |
| `discontinuity` | float | Log-density discontinuity at boundary |
| `z_stat` | float | Z-statistic for discontinuity |
| `pval` | float | P-value (H0: no discontinuity) |
| `bandwidth` | float | Bandwidth used for local linear regression |

### Pre-Trends F-Test (`pretrends_ftest.csv`)

| Variable | Type | Description |
|----------|------|-------------|
| `boundary` | string | Boundary tested |
| `caliper` | int | RD window width in meters |
| `f_stat` | float | F-statistic for joint test of pre-event coefficients |
| `pval` | float | P-value (H0: all pre-event β = 0) |
| `n_pre_coefs` | int | Number of pre-event coefficients tested |

### Covariate Balance (`covariate_balance.csv`, `covariate_balance_sfr.csv`)

| Variable | Type | Description |
|----------|------|-------------|
| `caliper` | int | RD window width in meters |
| `covariate` | string | Covariate name (e.g., year_built) |
| `label` | string | Human-readable covariate label |
| `coef` | float | Difference: inside_mean - outside_mean |
| `se` | float | Standard error of difference |
| `tstat` | float | T-statistic |
| `pval` | float | P-value (H0: no difference) |
| `n_inside` | int | Sample size inside boundary |
| `n_outside` | int | Sample size outside boundary |
| `inside_mean` | float | Mean covariate value inside |
| `outside_mean` | float | Mean covariate value outside |
| `diff_pct` | float | Percent difference: 100 × coef / outside_mean |
| `balanced` | bool | True if pval > 0.05 |

### Buyer Composition DiD (`buyer_composition_did.csv`)

| Variable | Type | Description |
|----------|------|-------------|
| `outcome` | string | Buyer type outcome (LLC_share, Portfolio_share) |
| `inside_pre` | float | Mean outcome inside, pre-flood |
| `inside_post` | float | Mean outcome inside, post-flood |
| `outside_pre` | float | Mean outcome outside, pre-flood |
| `outside_post` | float | Mean outcome outside, post-flood |
| `did_estimate` | float | DiD estimate: (inside_post - inside_pre) - (outside_post - outside_pre) |
| `pval` | float | P-value for DiD estimate |

### Trend Analysis (`trend_analysis.csv`)

| Variable | Type | Description |
|----------|------|-------------|
| `caliper_m` | int | Caliper window in meters |
| `start_year` | int | Start year for analysis |
| `end_year` | int | End year for analysis |
| `n_sales` | int | Number of sales in sample |
| `pretrend_coef` | float | Inside × Trend coefficient (differential pre-trend) |
| `pretrend_se` | float | Standard error |
| `pretrend_pval` | float | P-value for differential pre-trend |
| `trend_break_pre` | float | Pre-event trend in inside share |
| `trend_break_post_shift` | float | Level shift at event date |
| `trend_break_change` | float | Change in trend post-event |

### Trend-Adjusted DiD (`trend_adjusted_did.csv`)

| Variable | Type | Description |
|----------|------|-------------|
| `model` | string | Specification: "no_trends" or "with_trends" |
| `did_coef` | float | DiD treatment effect coefficient |
| `did_se` | float | Robust standard error |
| `did_pval` | float | P-value |
| `did_ci_lo` | float | Lower 95% confidence interval |
| `did_ci_hi` | float | Upper 95% confidence interval |
| `inside_trend` | float | Inside group time trend (with_trends only) |
| `outside_trend` | float | Outside group time trend (with_trends only) |
| `n_obs` | int | Number of observations |
| `r_squared` | float | R-squared of regression |

**Key Finding**: DiD coefficient changes 165% when trend controls are added (0.29 → 0.78).

---

## Coordinate Reference Systems

| Data | CRS | EPSG |
|------|-----|------|
| Parcel centroids (output) | WGS84 | EPSG:4326 |
| Boundary distance calculations | UTM Zone 15N | EPSG:32615 |
| SFHA boundaries | NAD83 | EPSG:4269 |
| Sentinel-2 imagery | WGS84 / UTM Zone 14N | EPSG:32614 |
| DEM tiles | NAD83 (varies) | Varies by tile |
