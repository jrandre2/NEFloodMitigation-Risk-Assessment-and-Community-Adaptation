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
| `in_sfha_window_150m` | boolean | Parcel within ±150m of SFHA boundary |
| `in_sfha_window_300m` | boolean | Parcel within ±300m of SFHA boundary |
| `in_inund_window_150m` | boolean | Parcel within ±150m of inundation boundary |
| `in_inund_window_300m` | boolean | Parcel within ±300m of inundation boundary |

### Ring Variables (for Poisson models)

| Variable | Type | Description |
|----------|------|-------------|
| `ring_sfha_0_250m` | boolean | Parcel 0-250m outside SFHA (near-but-dry) |
| `ring_sfha_250_300m` | boolean | Parcel 250-300m outside SFHA |
| `ring_inund_0_250m` | boolean | Parcel 0-250m outside 2019 inundation |
| `ring_inund_250_300m` | boolean | Parcel 250-300m outside 2019 inundation |

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
| `year_month` | string | Year-month (YYYY-MM format) |
| `event_time` | integer | Months relative to March 2019 (event_time=0). Range: [-24, +24] |
| `post` | boolean | Post-flood period indicator (event_time ≥ 0) |

### Outcome Variables (Parcel-Month Panel)

| Variable | Type | Description |
|----------|------|-------------|
| `sale_occurred` | boolean | Sale recorded in this parcel-month |
| `sale_count` | integer | Number of sales in parcel-month (usually 0 or 1) |

### Outcome Variables (Microcell Panel)

| Variable | Type | Description |
|----------|------|-------------|
| `cell_id` | string | Microcell identifier |
| `monthly_sales` | integer | Count of sales in cell-month |

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

## Coordinate Reference Systems

| Data | CRS | EPSG |
|------|-----|------|
| Parcel centroids (projected) | NAD83 / Nebraska State Plane | EPSG:32104 |
| Parcel centroids (geographic) | WGS84 | EPSG:4326 |
| SFHA boundaries | NAD83 | EPSG:4269 |
| Sentinel-2 imagery | WGS84 / UTM Zone 14N | EPSG:32614 |
