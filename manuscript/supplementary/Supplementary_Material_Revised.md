---
title: "Supplementary Material"
papersize: letter
geometry: margin=0.5in
---

# Supplementary Material

---

## Figures

![**Figure S1.** Classifier validation panels for the BERT model showing (a) normalized confusion matrix, (b) reliability calibration curve, (c) high-confidence coverage by class, and (d) ablation comparison.](figures_compressed/fig_B9_classifier_panels.png){width=7.5in}

\newpage

![**Figure S2.** Map of Douglas County Single Family Residential Parcels by Entity Type](figures_compressed/fig_A6_small_multiple_maps.png){width=7.5in}

\newpage

![**Figure S3.** Concentration of single family SFHA-located parcels by organizational form](figures_compressed/fig_A5_lorenz_small_multiples.png){width=7.5in}

\newpage

![**Figure S4.** Adjusted RRs for SFHA Locations by Ownership Structure and Portfolio Size](figures_compressed/fig_A1_forest_rr.png){width=7.5in}

\newpage

![**Figure S5.** Individual Owner Spatial Clustering by Parcel](figures_compressed/entity_LISA_is_individual.png){width=7.5in}

\newpage

![**Figure S6.** LLC Owners Spatial Clustering (LISA) by Parcel](figures_compressed/entity_LISA_is_llc.png){width=7.5in}

\newpage

![**Figure S7.** Corporate Owners Spatial Clustering (LISA) by Parcel](figures_compressed/entity_LISA_is_corporation.png){width=7.5in}

\newpage

![**Figure S8.** Government/Nonprofit Owners Spatial Clustering (LISA) by Parcel](figures_compressed/entity_LISA_is_other.png){width=7.5in}

\newpage

## Appendix Tables

### Table S1: Alignment between clusters and production labels

| Method | Clusters | Noise % | Weighted purity (overall) | Weighted purity (non-noise) | NMI | ARI |
|--------|----------|---------|---------------------------|------------------------------|-----|-----|
| Kmeans (K=2) | 2 | 0.00% | 95.1% | 95.1% | 0.806 | 0.955 |
| Kmeans (K=3) | 3 | 0.00% | 98.0% | 98.0% | 0.891 | 0.980 |
| Kmeans (K=4) | 4 | 0.00% | 98.1% | 98.1% | 0.522 | 0.281 |
| Kmeans (K=5) | 5 | 0.00% | 98.4% | 98.4% | 0.882 | 0.968 |
| HDBSCAN (leaf) | 44 | 46.16% | 95.2% | 99.0% | 0.162 | 0.018 |

---

### Table S2: Kmeans cluster composition by production label

| Cluster | Size | Individual | LLC | Trust | Corporation | Gov/Nonprofit | Dominant label | Purity |
|---------|------|------------|-----|-------|-------------|---------------|----------------|--------|
| 0 | 143,362 | 143,358 | 4 | 0 | 0 | 0 | Individual | 100.0% |
| 1 | 2,059 | 7 | 924 | 2 | 1,126 | 0 | Corporation | 54.7% |
| 2 | 7,790 | 0 | 7,790 | 0 | 0 | 0 | LLC | 100.0% |
| 3 | 5,482 | 688 | 3 | 4,790 | 0 | 1 | Trust | 87.4% |
| 4 | 1,985 | 92 | 83 | 264 | 1,052 | 494 | Corporation | 53.0% |

---

### Table S3. Ownership classifier configuration and training

| Component | Setting |
|-----------|---------|
| Base model | BERT base uncased with 5-way Softmax head |
| Labels emitted | Individual; LLC; Corporation; Trust; Gov/Nonprofit |
| Training data | 2,520 hand-labeled names (augmented to 5,040) |
| Loss & regularization | Focal loss; classification-head dropout = 0.20 |
| Model selection | 3-fold CV; early stopping on validation weighted F1 |
| Reliability checks | Probability calibration; stratified human audit |
| High-confidence rule | p-hat max >= 0.80 used for robustness subset |
| Deployment scope | 212,310 owner records countywide |

---

### Table S4. Production classification coverage and high-confidence shares (countywide)

| Owner type | Count | Percent | High-confidence count | High-confidence coverage % |
|------------|-------|---------|----------------------|---------------------------|
| Individual | 162,079 | 76.3 | 161,409 | 99.6 |
| LLC | 29,161 | 13.7 | 27,722 | 95.1 |
| Corporation | 5,588 | 2.6 | 4,169 | 74.6 |
| Trust | 7,006 | 3.3 | 6,844 | 97.7 |
| Gov/Nonprofit | 8,476 | 4.0 | 7,516 | 88.7 |
| **Total** | **212,310** | **100.0** | **207,660** | **97.8** |

---

## Additional Tables (To Be Added)

### Table S5. Robustness checks: Key contrasts under alternative specifications

Risk ratios (RR) and 95% confidence intervals for selected contrasts across model specifications. Reference group: Individual — Single in all models.

| Group | Primary (M2) | With Distance (M4) | Attenuation |
|-------|--------------|-------------------|-------------|
| LLC — Single | 1.670 [1.311–2.126]*** | 1.645 [1.293–2.093]*** | 1.5% |
| LLC — Multi | 1.379 [1.104–1.722]** | 1.358 [1.086–1.698]** | 1.5% |
| Individual — Multi | 1.209 [1.038–1.407]* | 1.201 [1.031–1.399]* | 0.7% |
| Trust — Multi | 1.459 [1.043–2.041]* | 1.446 [1.034–2.023]* | 0.9% |
| Corporation — Multi | 1.568 [0.960–2.561] | 1.548 [0.947–2.531] | 1.3% |

*Note: M2 = primary model (neighborhood FE, controls for log value and log acres); M4 = M2 + log(1+owner distance). Attenuation = percentage reduction in log(RR) from M2 to M4. Significance: \*\*\* p < 0.001, \*\* p < 0.01, \* p < 0.05. Standard errors clustered by neighborhood. N = 183,313 (M2, M4).*

**Additional robustness checks:**
- *High-confidence subset (p̂ max ≥ 0.80):* Key contrasts remain significant with directionally consistent estimates (n ≈ 179,000)
- *10% overlap SFHA rule:* Results stable when relaxing majority-area threshold to 10% parcel overlap
- *Alternative SE specifications:* Robust (HC3) and spatial HAC (Conley) standard errors yield consistent inference for primary contrasts

---

### Table S6. Owner-level SFHA intensity model

This table addresses the concern that multi-property owners are mechanically more likely to own SFHA parcels simply by holding more parcels. The model estimates SFHA intensity at the owner level using a Poisson regression with an offset for log(total SFR parcels owned), effectively controlling for portfolio size.

| Entity type | N owners | Mean SFHA parcels | IRR [95% CI] | p-value |
|-------------|----------|-------------------|--------------|---------|
| Individual | 139,154 | 0.010 | 1.000 (ref) | — |
| LLC | 7,646 | 0.060 | 2.577 [2.318–2.866] | <0.001 |
| Corporation | 1,076 | 0.076 | 3.746 [2.997–4.682] | <0.001 |
| Trust | 5,628 | 0.016 | 1.644 [1.331–2.031] | <0.001 |
| Gov/Nonprofit | 1,098 | 0.060 | 2.867 [2.239–3.672] | <0.001 |

*Note: Poisson GLM with log link; offset = log(total SFR parcels owned). N = 154,602 unique owners. IRR = incidence rate ratio. The offset specification means IRRs represent the rate of SFHA parcels per parcel owned, controlling for portfolio size.*

**Interpretation:** After accounting for the mechanical effect of portfolio size, entity-owned portfolios still exhibit significantly elevated SFHA intensity relative to individual-owned portfolios. Corporations show the highest IRR (3.75×), followed by Gov/Nonprofit entities (2.87×), LLCs (2.58×), and Trusts (1.64×). These results complement the parcel-level analysis (Table 7) by confirming that the elevated SFHA exposure among entity owners is not simply an artifact of larger portfolio sizes.

---

### Table S7. Eigenvector spatial filtering sensitivity analysis

This table presents results from a sensitivity analysis varying the number of eigenvector spatial filters (ESF) to assess whether ownership–SFHA associations are robust to explicit spatial modeling. Eigenvectors are extracted from a k=8 nearest-neighbor spatial weights matrix and included as controls in the Poisson GLM with neighborhood fixed effects (Griffith, 2003).

| N Filters | Moran's I | LLC RR | LLC p-value | Corp RR | Trust RR | Gov/NP RR |
|-----------|-----------|--------|-------------|---------|----------|-----------|
| 0 (FE only) | 0.392 | 1.379 | <0.001 | 1.162 | 0.935 | 0.804 |
| 15 | 0.260 | 1.386 | <0.001 | 1.096 | 0.936 | 0.782 |
| 30 | 0.312 | 1.285 | <0.001 | 0.986 | 0.886 | 0.712 |
| 50 | 0.146 | 1.316 | <0.001 | 0.991 | 0.879 | 0.622 |
| 75 | 0.146 | 1.316 | <0.001 | 0.995 | 0.859 | 0.701 |
| 100 | 0.198 | 1.270 | <0.001 | 0.946 | 0.878 | 0.619 |
| 150 | 0.126 | 1.225 | <0.001 | 0.903 | 0.904 | 0.690 |
| 200 | 0.003 | 1.235 | <0.001 | 1.012 | 0.916 | 0.665 |

*Note: N = 183,313 SFR parcels. Reference group: Individual owners. Moran's I computed on model residuals using an 8-nearest-neighbor weights matrix. RR = risk ratio for entity type (pooling single- and multi-parcel owners). With 200 spatial filters, residual Moran's I decreases from 0.392 to 0.003—a 99% reduction in residual spatial autocorrelation. The LLC risk ratio attenuates modestly from 1.38 to 1.24 (approximately 10% attenuation) but remains statistically significant (p < 0.001) across all specifications, indicating that the LLC–SFHA association is robust to spatial structure.*

---

## Model Card — Ownership-Form Classifier

### A1. Performance (evaluation set, n = 756)

**Overall:** Accuracy = 0.9934; Micro-F1 = 0.9934; Macro-F1 = 0.9800; Weighted-F1 = 0.9934.

**Per-class (Precision / Recall / F1 / Support):**
- Corporation: 1.000 / 0.9286 / 0.9630 / 42
- Individual: 0.9982 / 0.9982 / 0.9982 / 561
- LLC: 0.9878 / 1.0000 / 0.9939 / 81
- Government/Nonprofit: 0.9429 / 1.0000 / 0.9706 / 33
- Trust: 0.9744 / 0.9744 / 0.9744 / 39

### A6. Ablation & Baseline Comparisons

**Token-removal ablation:** From test strings, remove {LLC, LLP, LP, Inc, Corp, Ltd, Trust} (case-insensitive, punctuation-agnostic).

**Rule-based baseline:** Regex: CITY/COUNTY/STATE OF… → Government/Nonprofit; LLC/LLP/LP; TRUST/Trustee; INC/INCORPORATED/CORP/COMPANY/LTD; else Individual. Precedence: Government/Nonprofit → LLC → Trust → Corporation → Individual.

**Split hygiene:** Ablations evaluated on the held-out test set; augmented variants tied to the same base string remain in the same fold.

### Overall performance comparison

| Model | Accuracy | Macro F1 | Micro F1 | Weighted F1 |
|-------|----------|----------|----------|-------------|
| BERT — Original | 0.993386 | 0.979999 | 0.993386 | 0.993356 |
| BERT — Ablated | 0.953704 | 0.885115 | 0.953704 | 0.952532 |
| Rule-based Baseline | 0.841270 | 0.509683 | 0.841270 | 0.806666 |
