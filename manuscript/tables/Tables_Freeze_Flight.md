# Tables - Freeze and Flight

*Liquidity and Spatial Sorting in Housing Markets After a Major Flood*

---

## Table 1. Parcel-month sale rates near hazard boundaries (boundary RD-in-panel)

*Rates are per parcel-month; DiD = [inside post - inside pre] - [outside post - outside pre].*

| Boundary | Caliper (m) | Inside pre | Inside post | Outside pre | Outside post | Delta Inside | Delta Outside | DiD | 95% CI | N (parcel-months) |
|----------|-------------|------------|-------------|-------------|--------------|--------------|---------------|-----|--------|-------------------|
| SFHA | 150 | 0.002376 | 0.002116 | 0.002410 | 0.002916 | -0.000260 | +0.000506 | **-0.000766** | [-0.001563, 0.000031] | 824,131 |
| SFHA | 300 | 0.002345 | 0.002035 | 0.002554 | 0.003134 | -0.000310 | +0.000580 | **-0.000890** | **[-0.001634, -0.000145]** | 1,798,986 |
| Inundation | 150 | 0.002070 | 0.003975 | 0.003535 | 0.004284 | +0.001905 | +0.000750 | +0.001155 | [-0.001468, 0.003778] | 65,072 |
| Inundation | 300 | 0.002222 | 0.004089 | 0.003395 | 0.003694 | +0.001867 | +0.000299 | +0.001568 | [-0.000647, 0.003783] | 110,250 |

*Source: `data_work/tab_rd_summary.csv`*

---

## Table 2. Log-price difference-in-differences near hazard boundaries (sale months only)

| Boundary | Caliper (m) | Inside median log-price (pre) | Inside median log-price (post) | Outside median log-price (pre) | Outside median log-price (post) | DiD (log) | 95% CI | N (sales) |
|----------|-------------|-------------------------------|--------------------------------|--------------------------------|---------------------------------|-----------|--------|-----------|
| SFHA | 150 | 11.857 | 12.029 | 12.193 | 12.296 | **-0.353** | [-0.829, 0.123] | 2,174 |
| SFHA | 300 | 11.857 | 12.044 | 12.211 | 12.315 | **-0.264** | [-0.728, 0.201] | 5,085 |
| Inundation | 150 | 12.492 | 12.388 | 12.846 | 12.972 | -0.282 | [-1.049, 0.484] | 248 |
| Inundation | 300 | 12.242 | 12.429 | 12.690 | 12.942 | -0.260 | [-0.851, 0.330] | 387 |

*Source: `data_work/tab_rd_price_summary.csv`. Approximate level effects are e^(DiD) - 1.*

---

## Table 3. Ring models for monthly sales counts just outside the line (Poisson, rate ratios)

| Boundary | Term (ring x post) | RR | 95% CI |
|----------|-------------------|-----|--------|
| SFHA | 0-250 m x Post | **1.441** | [0.976, 2.126] |
| SFHA | 250-300 m x Post | **1.310** | [0.874, 1.964] |
| Inundation | 0-250 m x Post | 0.614 | [0.317, 1.189] |
| Inundation | 250-300 m x Post | 0.454 | [0.189, 1.087] |

*Model includes cell and month fixed effects; full coefficient tables (including "post" main effects) in the export.*

*Source: `data_work/tab_poisson_ring_models.csv`*

---

## Table 4. SFHA share of boundary-window sales (+/-300 m around SFHA line)

| Period | Mean monthly share inside SFHA | N (months) |
|--------|-------------------------------|------------|
| Pre (2017-03 to 2019-02) | **3.35%** | 24 |
| Post (2019-04 to 2021-03) | **2.27%** | 24 |
| Change (Post - Pre) | **-1.08 pp** | --- |

*Computed from the monthly boundary-window series underlying fig_sfha_share_rd.png.*

*Source: `data_work/tab_sfha_share_rd.csv`*

---

## Data Sources

All tables are generated from the Freeze and Flight analysis pipeline:

- **Table 1**: `src/07_estimation/rd_summary.py` -> `data_work/tab_rd_summary.csv`
- **Table 2**: `src/07_estimation/rd_summary.py` -> `data_work/tab_rd_price_summary.csv`
- **Table 3**: `src/07_estimation/poisson_ring_models.py` -> `data_work/tab_poisson_ring_models.csv`
- **Table 4**: `src/08_figures/sfha_share_figs.py` -> `data_work/tab_sfha_share_rd.csv`

*Pipeline re-run and verified: December 22, 2025*

*All values confirmed to match manuscript (Freeze_and_Flight.md) exactly.*
