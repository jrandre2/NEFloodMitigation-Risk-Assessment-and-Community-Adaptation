---
title: "Tables - Who Owns the Floodplain?"
geometry: margin=0.5in
fontsize: 9pt
header-includes:
  - \setlength{\LTleft}{0pt}
  - \setlength{\LTright}{0pt}
  - \setlength{\tabcolsep}{3pt}
  - \renewcommand{\arraystretch}{1.08}
  - \usepackage{pdflscape}
---

# Tables - Who Owns the Floodplain?

*Revised December 16, 2025*

## Table 1. Variables and transformations

| Variable | Definition & source | Unit | Transform | Notes |
|----------|---------------------|------|-----------|-------|
| SFHA exposure | Parcel majority area within Zones A/AE/AO/AH | Indicator | 0–1 | 10% overlap sensitivity checks boundary effects |
| Total value | Assessed parcel value | USD | Natural log | Core control |
| Acres | Assessor parcel size | Acres | Natural log | Core control |
| Improvement status | Building presence from assessor/footprints | Indicator | 0–1 | Used in descriptive splits & sensitivity |
| Owner type | Organizational form from classifier | Category | one-hot | Individual; LLC; Corporation; Trust; Gov/Nonprofit |
| Local portfolio footprint | SFR parcels per exact owner-of-record name (within Douglas County) | Count | Single vs multi | Exact match on owner-of-record name |
| Entity × footprint | Cross of owner type & footprint | Categories | one-hot | 10 mutually exclusive groups; Individual—Single is the reference |
| Neighborhood | Assessor appraisal neighborhood | Categories | one-hot | Fixed effects; 470 groups |
| Owner distance | Great-circle km: parcel centroid to owner ZIP centroid | Kilometers | log(1+km) | Secondary robustness control |

---

## Table 2. Organizational forms: functional definitions and interpretive role

| Category | Owner-of-record identification<br>(examples) | Interpretive role<br>in this study | Key caveats |
|----------|---------------------------------------------|------------------------------------|-------------|
| Individual | Personal names; joint names | Household/person owner-of-record | May be a landlord; name matching may split/merge households |
| LLC | Tokens such as "LLC", "L.L.C.",<br>"Limited Liability Company" | Business entity with limited liability; administratively legible as an entity | May be controlled by individuals, trusts, or corporations (beneficial owner not observed) |
| Corporation | "Inc", "Corp", "Co" (incorporated entities) | Business entity distinct from LLCs; may reflect corporate ownership strategies | Names can overlap with nonprofits or public entities; beneficial owner not observed |
| Trust | "Trust", "Trustee" and similar markers | Fiduciary/estate-planning or investment holding structure | Not a direct proxy for rental/tenure; beneficiaries not observed |
| Gov/NP | Public agencies; nonprofits; religious entities | Non-market or public-interest ownership relevant for administration | Heterogeneous category; may include quasi-public entities |

---

## Table 3. Local portfolio footprint distribution by organizational form (SFR)

| Entity type | Single-parcel | Single-parcel % | Multi-parcel | Multi-parcel % | Total |
|-------------|---------------|-----------------|--------------|----------------|-------|
| Individual | 131,436 | 86.32 | 20,826 | 13.68 | 152,260 |
| LLC | 4,725 | 23.04 | 15,782 | 76.96 | 20,507 |
| Corporation | 762 | 30.36 | 1,748 | 69.64 | 2,510 |
| Trust | 5,295 | 82.54 | 1,120 | 17.46 | 6,415 |
| Gov/Nonprofit | 803 | 30.43 | 1,836 | 69.57 | 2,639 |
| **Total** | **143,021** | **77.59** | **41,312** | **22.41** | **184,333** |

---

## Table 4. SFR owner-type distribution

| Owner type | Count | Percent |
|------------|-------|---------|
| Individual | 152,260 | 82.60 |
| LLC | 20,507 | 11.12 |
| Trust | 6,415 | 3.48 |
| Gov/Nonprofit | 2,639 | 1.43 |
| Corporation | 2,510 | 1.36 |
| **Total** | **184,333** | **100.00** |

---

::: {.landscape}

## Table 5. Model specifications

| Label | Specification | Distance term | Neighborhood FE | N parcels | FE groups |
|-------|--------------|---------------|-----------------|-----------|-----------|
| M0 | Entity × footprint indicators | — | No | 184,333 | — |
| M1 | M0 + log(value) + log(acres) | — | No | 184,333 | — |
| M2 (primary) | M1 + neighborhood FE | — | Yes | 183,313 | 470 |
| M3 | M1 + log(1+distance) | log(1+distance) | No | 184,333 | — |
| M4 | M1 + log(1+distance) + neighborhood FE | log(1+distance) | Yes | 183,313 | 470 |

*Note: N differs for fixed-effects models due to neighborhood assignment and required covariates; Table 7 reports estimates for the primary model (M2).*

---

:::

::: {.landscape}

## Table 6. Counts and unadjusted SFHA rates by organizational form and local footprint

| Entity & footprint | Parcels | % of SFR | SFHA count | % of SFHA | SFHA rate (%) | SFHA per 1,000 |
|--------------------|---------|----------|------------|-----------|---------------|----------------|
| Individual — Single | 131,436 | 71.30 | 1,065 | 52.5 | 0.81 | 8.1 |
| Individual — Multi | 20,826 | 11.30 | 260 | 12.8 | 1.25 | 12.5 |
| LLC — Single | 4,725 | 2.56 | 228 | 11.2 | 4.83 | 48.3 |
| LLC — Multi | 15,782 | 8.56 | 234 | 11.5 | 1.48 | 14.8 |
| Corporation — Single | 762 | 0.41 | 31 | 1.5 | 4.07 | 40.7 |
| Corporation — Multi | 1,748 | 0.95 | 51 | 2.5 | 2.92 | 29.2 |
| Trust — Single | 5,295 | 2.87 | 63 | 3.1 | 1.19 | 11.9 |
| Trust — Multi | 1,120 | 0.61 | 29 | 1.4 | 2.59 | 25.9 |
| Gov/NP — Single | 803 | 0.44 | 26 | 1.3 | 3.24 | 32.4 |
| Gov/NP — Multi | 1,836 | 1.00 | 40 | 2.0 | 2.18 | 21.8 |
| **Total** | **184,333** | **100.00** | **2,027** | **100.0** | **1.10** | **11.0** |

*Note: SFHA counts computed from parcel counts × SFHA rates; minor rounding differences may occur.*

---

:::

::: {.landscape}

## Table 7. Adjusted risk of SFHA location by organizational form and local footprint (primary model)

| Group | Risk ratio | 95% CI | p-value |
|-------|------------|--------|---------|
| Individual — Single | 1.000 | — | — |
| Individual — Multi | 1.209 | 1.038–1.407 | 0.015 |
| **LLC — Single** | **1.670** | **1.311–2.126** | **<0.001** |
| LLC — Multi | 1.379 | 1.104–1.722 | 0.005 |
| Corporation — Single | 0.989 | 0.626–1.563 | 0.963 |
| Corporation — Multi | 1.568 | 0.960–2.561 | 0.073 |
| Trust — Single | 0.844 | 0.612–1.165 | 0.303 |
| Trust — Multi | 1.459 | 1.043–2.041 | 0.028 |
| Gov/NP — Single | 0.994 | 0.480–2.059 | 0.987 |
| Gov/NP — Multi | 0.792 | 0.348–1.799 | 0.577 |

*Note: Modified Poisson GLM with log link; neighborhood fixed effects; controls for log(value) and log(acres). Standard errors clustered by neighborhood. N = 183,313.*

---

:::

::: {.landscape}

## Table 8. Within-form tests of equality between single- and multi-parcel owners (clustered)

| Entity type | Difference log RR (Single − Multi) | RR Single/Multi | Wald χ² | p-value | BH-FDR p-value |
|-------------|-----------------------------------|-----------------|---------|---------|----------------|
| LLC | 0.1914 | 1.2109 | 2.8347 | 0.0923 | 0.1230 |
| Corporation | -0.4606 | 0.6309 | 4.0041 | 0.0454 | 0.0908 |
| Trust | -0.5468 | 0.5788 | 4.6692 | 0.0307 | 0.0908 |
| Gov/NP | 0.2276 | 1.2556 | 0.8837 | 0.3472 | 0.3472 |

*Note: p-values reflect Wald tests on coefficient differences from the primary model; false discovery rate controlled using Benjamini & Hochberg (1995).*

---

:::

::: {.landscape}

## Table 9. Residual spatial dependence diagnostic for the primary model (M2)

| Moran's I (residuals) | Expected I | p-value (normal) | p-value (randomization) | k-nearest neighbors | N |
|-----------------------|------------|------------------|-------------------------|---------------------|-------|
| 0.392 | -0.000005 | <0.001 | <0.001 | 8 | 183,313 |

*Note: Moran's I computed on primary-model residuals using an 8-nearest-neighbor weights matrix defined on parcel centroids.*

:::
