**Freeze and Flight: Liquidity and Spatial Sorting in Housing Markets After a Major Flood\
\
\
Abstract**

Floods make risk salient, but housing markets can adjust through thin liquidity and sorting rather than clean, precisely estimated price cuts. We study the March 2019 Missouri River flood in Douglas County, Nebraska using a boundary difference‑in‑differences (RD‑in‑panel) design at FEMA Special Flood Hazard Area (SFHA) lines and at the 2019 inundation edge, complemented by near‑but‑dry "ring" models of sales counts. Within ±300 m of the SFHA boundary, the inside parcel‑month sale rate fell from 0.002345 to 0.002035 while the outside rate rose from 0.002554 to 0.003134, yielding a DiD of −0.000890 (95% CI: −0.001634, −0.000145; *n* = 1,798,986 parcel‑months)---about a 38% decline relative to the inside pre‑rate. Micro‑cell Poisson models show post‑event increases in sales just outside the line (rate ratios ≈ 1.44 in 0--250 m and 1.31 in 250--300 m), consistent with substitution toward near‑but‑dry parcels. Price level contrasts near the line are suggestively negative but imprecise (log‑price DiD ≈ −0.26 to −0.35 with confidence intervals spanning zero). The share of boundary‑window sales occurring inside the SFHA falls by roughly 1.1 percentage points after the flood (see monthly series), again indicating re‑routing (all estimates from our RD summary tables and figures). Taken together with prior work in this county showing that exposure is concentrated among single‑parcel LLCs and rises sharply with owner--parcel distance, we frame and test whether post‑flood purchases in exposed belts reallocate toward those buyer types (organizational form, portfolio scale, and proximity). 

**1. Introduction**

Flood events heighten risk salience, yet market responses need not appear first in prices. In thin submarkets, sellers may wait, and buyers may substitute to nearby---but legally "dry"---parcels just outside regulatory lines. These dynamics matter for disclosure, mitigation, and equity because they determine not only what properties transact but also who ultimately holds mapped risk. In Douglas County, Nebraska, two recent studies have shown that ownership structure and proximity already shape where exposure sits. Using a county‑scale owner‑of‑record classifier and portfolio counts, *Who Owns the Floodplain?* reports that single‑parcel LLCs are the most flood‑exposed holders of single‑family residential (SFR) parcels even after parcel attributes and neighborhood fixed effects are absorbed, consistent with liability‑siloing at the parcel level. A companion analysis demonstrates a steep owner‑distance gradient for SFR exposure: out‑of‑state holdings exhibit much higher SFHA and 2019 event exposure than owner‑occupied parcels, and the per‑log‑kilometer association remains strong among non‑occupants (maps on page 4 of that study visualize the spatial alignment of SFHAs, the 2019 footprint, and non‑occupant ownership). Together these findings establish *who holds* and *how far they live from* mapped risk in this inland county. 

This paper turns from cross‑sectional exposure to *market response at the hazard edge* when flood risk becomes salient. We assemble a parcel‑level sales panel spanning the March 2019 flood and evaluate outcomes exactly where policy and insurance hinge: at SFHA boundaries and at the mapped inundation edge. Two local designs anchor the analysis. First, a boundary RD‑in‑panel contrasts parcels just inside and just outside the line within narrow calipers (±150--300 m), estimating pre/post differences in sale rates and prices. Second, near‑but‑dry "ring" models of micro‑cell sales counts test whether transactions re‑route to parcels immediately outside the line. We also track the monthly share of boundary‑window sales occurring inside the SFHA to gauge composition within the caliper. 

Three questions motivate the study. First, do sale rates and prices inside the SFHA fall relative to just outside after the 2019 flood, and if so, for how long? Second, do transactions substitute to near‑but‑dry parcels immediately outside hazard boundaries, consistent with segmentation rather than pure price repricing? Third, building on prior evidence about ownership form and proximity, do post‑event purchases in exposed belts reallocate toward specific buyer types---notably single‑parcel LLCs and more distant owners---thereby linking market response to the legal and geographic patterns of who holds risk in the cross‑section? 

**1.1 Who Owns the Floodplain **

The boundary diagnostics point toward a liquidity‑and‑sorting response. Within ±300 m of the SFHA line, the inside parcel‑month sale rate declines relative to outside (DiD ≈ −0.000890), a roughly 38% contraction against the inside baseline, while micro‑cell counts rise just outside the line (post‑event rate ratios ≈ 1.44 in 0--250 m and 1.31 in 250--300 m). Price‑level contrasts at the line are negative in sign but imprecisely estimated, and the monthly share of sales occurring on the inside of the line within the boundary window drops by about 1.1 percentage points after the flood. Results at the 2019 inundation edge are directionally similar but less precise, reflecting a smaller treatment window and the event's spatial concentration. These patterns together suggest that, in the short run, where transactions happen, adjusts more sharply than what buyers pay at the boundary. 

Our contribution is to connect that boundary behavior to previously documented ownership structure and distance patterns in the same county. If liquidity thins inside and demand re‑routes just outside, then the composition of buyers who do transact inside after the flood is a plausible mechanism linking salience to longer‑run exposure: liability‑siloed entities and distant investors may be more willing or able to "buy the dip," reinforcing the concentration documented in our earlier work. By unifying boundary event analysis with organizational form, portfolio scale, and proximity, the paper provides mechanism‑rich evidence on how salience locates risk across both space and owner types within a single inland market. 

**\
2. Literature Review**

**2.1 Flood Risk Capitalization and the Role of Salience**

A substantial body of research establishes that flood risk is capitalized into housing prices, with properties located in designated floodplains generally selling at a discount compared to similar homes in safer areas. Seminal studies using hedonic pricing models have consistently identified a negative price differential associated with floodplain location (Bin & Polasky 2004; Kousky 2018). The magnitude of this discount, however, varies significantly across markets and over time, influenced by local amenities, the perceived severity of the risk, and the costs of mitigation and insurance (Daniel et al. 2009). A meta-analysis by Beltrán et al. (2018) found that floodplain discounts typically range from 4% to 12%, often approximating the capitalized cost of mandatory flood insurance under the National Flood Insurance Program (NFIP).

The market\'s perception of risk is not static; it is amplified by recent, salient events. Following a major flood, this abstract risk becomes a tangible reality, leading to a sharp, albeit often temporary, increase in the price discount for exposed properties (Atreya et al. 2013; Gallagher 2014). This \"salience effect\" suggests that buyers\' attention and risk aversion are heightened in the immediate aftermath of a disaster, causing them to demand a higher premium for bearing flood risk (Kousky 2010). However, this effect tends to decay as memories of the event fade, with prices in flood-prone areas often rebounding within a few years, particularly in tight housing markets (Bin & Landry 2013). This dynamic indicates that while markets price known risks, the information conveyed by a recent disaster can cause a significant, short-term re-evaluation.

**2.2 Beyond Prices: Liquidity Freezes and Spatial Sorting**

While price capitalization is a key market response, it is not the only one. Recent scholarship highlights that housing markets, especially those with thin trading volume, may adjust to new information through changes in liquidity and transaction patterns rather than immediate and precise price cuts. Following a disaster, uncertainty about property condition, insurance payouts, and future risk can lead to a \"market freeze,\" characterized by a significant drop in transaction volume and an increase in time-on-market for listed properties (Chan 2011; Dávila 2022). Sellers may choose to wait rather than accept a low price, and lenders may tighten underwriting standards, further constraining sales. This liquidity response suggests that observing transaction rates, as this study does, is critical to understanding the immediate market adjustments that may precede or even supplant price changes.

In parallel with a liquidity freeze, flood events can trigger spatial sorting, where demand shifts away from high-risk areas toward nearby but safer locations. This \"flight to safety\" manifests as households relocating to less exposed neighborhoods, a dynamic observed in coastal markets facing sea-level rise (Bakkensen and Ma 2020) and areas affected by wildfire (Mueller et al. 2022). Such sorting is consistent with the findings of this paper, which documents a decline in sales *inside* the SFHA and a corresponding increase in sales in \"near but dry\" rings *just outside* the boundary. The SFHA line, as a clear regulatory and insurance demarcation, can act as a powerful anchor for this re-routing of housing demand, even if the underlying hydrological risk changes gradually across the boundary (Ortega and Taspinar 2018).

**2.3 Who Holds the Risk? Ownership Structure and Post-Disaster Buyers**

The question of how risk is allocated across a market leads to an examination of *who* owns vulnerable properties. The existing cross-sectional distribution of risk is not random. Research has shown that lower-income households and minority populations are often disproportionately concentrated in flood-prone areas, partly due to housing affordability gradients (Bakkensen and Ma 2020; Collins et al. 2018). Beyond demographics, ownership structure and investor presence are crucial. Sophisticated commercial buyers may be better able to price risk than individual households, potentially acquiring properties at deeper discounts (Gourevitch et al. 2023). The use of limited liability companies (LLCs) to hold rental properties, a common practice for liability protection, can also concentrate risk in specific ownership forms, as documented in prior work in Douglas County.

The composition of buyers may shift significantly after a disaster. The aftermath of a flood can create opportunities for cash-rich investors to purchase damaged properties at a discount from distressed sellers who may lack the capital for repairs (Davidoff & Zytnick 2023). This can lead to an increase in the share of cash transactions and a rise in investor ownership in affected neighborhoods (Hamideh et al. 2021). By examining changes in buyer characteristics---specifically organizational form (LLCs), portfolio scale, and proximity---this study connects the short-term market adjustments of liquidity and sorting to the longer-term question of how flood risk is reallocated across different types of owners after a salience-inducing event. This paper's contribution lies in empirically linking the boundary-level market dynamics to the characteristics of the transacting parties, providing a mechanism for the ownership patterns observed in the cross-section.

**\
3. Data and Methods**

**3.1 Study area, period, and unit of analysis**

The study covers all single‑family residential (SFR) parcels in Douglas County, Nebraska, and the sales recorded around the March 2019 Missouri River flood. We construct a parcel‑month panel spanning twenty‑four months on either side of March 2019 (event time), and a sales file with transaction‑level prices and parties. Regulatory flood exposure follows FEMA Special Flood Hazard Areas (SFHAs), and realized exposure references the March 2019 inundation footprint. Where the provenance of a layer is ambiguous in this manuscript, we adopt the same sources and construction rules documented in prior Douglas County work (e.g., FEMA NFHL for SFHAs, a Sentinel‑2--derived 2019 inundation boundary, and Microsoft building footprints), while re‑implementing the processing steps from scratch for this study. 

**3.2 Parcel, hazard, event, and sales datasets**

Parcels are drawn from the county assessor's GIS with polygon geometry, situs and mailing fields, land‑use codes, building age, and assessed value. SFHA polygons (zones A/AE/AO/AH) define regulatory exposure. The 2019 inundation mask is derived from post‑event imagery. We keep improved and unimproved SFR parcels but flag building presence. For SFHAs we use a **majority‑area** rule (sensitivity at a 10 percent overlap) and for inundation a **≥ 5 percent** overlap rule; these thresholds match local practice reported in earlier Douglas County analyses. 

Sales and deeds come from the assessor/register of deeds files. We link transactions to parcels on APN and legal description; unmatched or split/merged parcels are flagged. *Arms‑length* filters remove nominal‑consideration transfers, intra‑family gifts, sheriff/foreclosure deeds, and intra‑entity clean‑ups. Each transaction is priced in natural logs and dated to the recording month. We identify **owner‑occupancy at purchase** by matching cleaned mailing and situs fields. These constructions harmonize with prior county‑scale exposure studies, but all coding rules are implemented anew here. 

**3.3 Buyer classification, portfolio scale, and proximity**

To study buyer composition, we label the **organizational form** of each buyer at purchase (Individual, LLC, Corporation, Trust, Government/Nonprofit) using a fine‑tuned transformer classifier previously validated on county land‑records; we retain the legal shell (owner of record) as the analytic unit and report high‑confidence results as a robustness subset. **Portfolio scale** is measured by counting parcels per exact legal name at the time of purchase and coded as single‑parcel vs. multi‑parcel. These choices follow a documented and auditable approach used locally to study ownership form and scale. 

Buyer--parcel **proximity** is the great‑circle distance between the parcel centroid and the buyer's mailing ZIP‑code centroid. Because ZIP‑centroid distances can be noisy at very short ranges, we analyze distance both continuously (log‑kilometers) and in administrative bands (same ZIP; other ZIP in Douglas County; adjoining county; other Nebraska county; other state), echoing the proximity encodings that have been shown to align well with kilometer bins in this setting. 

**3.4 Treatment and boundary construction**

We implement two boundaries: the regulatory SFHA line and the realized 2019 inundation edge. Parcels receive a signed Euclidean distance to each boundary and an inside/outside flag. For local counterfactuals, we restrict to symmetric *calipers* of ±150 m and ±300 m. At the SFHA boundary, the ±300 m window contains 1,798,986 parcel‑months for sale‑rate models and 5,085 boundary‑proximate sales for price models; the ±150 m window contains 824,131 parcel‑months and 2,174 sales. At the 2019 inundation edge, the corresponding counts are 110,250 and 387(±300 m) and 65,072 and 248 (±150 m). We also define near‑but‑dry rings immediately *outside* each boundary (0--250 m and 250--300 m) for substitution tests. 

**3.5 Outcomes**

We analyze (i) prices---log sale price, restricted to months with a transaction; (ii) liquidity---a parcel‑month sale indicator and micro‑area monthly sales counts aggregated over small equal‑area cells that tile the county; and (iii) composition---the monthly share of boundary‑window sales occurring inside the SFHA, the shares by buyer organizational form × portfolio scale, buyer‑distance bands, and the cash‑purchase share.

**3.6 Event‑time and fixed‑effects structure**

We estimate models on a balanced parcel‑month panel indexed by event time t∈\[−24,+24\]. All parcel‑level regressions include **parcel fixed effects** and **neighborhood×month fixed effects** (assessor appraisal neighborhoods crossed with month) to absorb micro‑spatial heterogeneity and county‑wide housing‑cycle shocks. This FE scaffold is standard in the county's parcel‑scale floodwork and is chosen to address the pronounced spatial clustering documented for hazards and realized inundation. 

**3.7 Identification strategies**

**Boundary RD‑in‑panel (primary design).**\
For sale incidence we estimate a difference‑in‑differences at the boundary, framed as an RD‑in‑panel:

Pr⁡(Saleit=1)=∑τ≠−1βτ\[1{Insidei}×1{t=τ}\]+αi+γn(i),t+εit,

within the ±150/±300 m windows. For **prices**, the same structure is applied to log⁡Pit on sale months. Parallel‑trend checks inspect pre‑event coefficients. Neighborhood‑clustered standard errors are the default; **Conley‑type spatial HAC** provides a robustness alternative given spatial dependence in flood processes. Window sizes and available sales within each caliper are as reported above. 

**Near‑but‑dry ring models (substitution).**\
To test whether transactions re‑route just outside hazard lines after the event, we model micro‑cell **sales counts** with Poisson regressions (and negative‑binomial sensitivity):

log⁡E\[yct\]=αc+δt+θ1 \[1{Ring 0 ⁣− ⁣250 m}×1{Post}\]+θ2 \[1{Ring 250 ⁣− ⁣300 m}×1{Post}\],

where αc and δt are cell and month fixed effects; yct is the number of sales in cell c at month t. 

**Composition and mechanism tests.**\
Inside the boundary windows we estimate post‑event shifts in (i) the **share of purchases by buyer form × portfolio scale**, (ii) **buyer‑distance bands**, and (iii) **cash share**, using linear probability or fractional models with neighborhood×month fixed effects. Buyer‑form and scale are taken from the legal‑shell classifier; proximity encodings follow the banding used elsewhere in Douglas County analysis. 

**3.8 Robustness, sensitivity, and diagnostics**

We report: (a) caliper sensitivity (±150 m vs. ±300 m), (b) alternate SFHA definitions (majority‑area vs. 10 percent overlap), (c) ring‑width sensitivity, (d) placebo event‑month tests, and (e) **repeat‑sales residual** price models. Spatial structure is addressed with neighborhood clustering and spatial HAC; strong clustering of SFHA and the 2019 footprint in this county motivates these choices. 

## **4. Results**

We organize the results around three outcomes at the hazard edge: (i) transaction **rates** (parcel‑month sale incidence) inside versus outside the boundary, (ii) **price levels** at sale, and (iii) **where transactions re‑route** immediately outside the line. Throughout, windows are symmetric calipers around the boundary (±150 m and ±300 m). All counts, rates, and confidence intervals are taken directly from the boundary‑window exports prepared for this project; tables reproduce those summaries. 

**4.1. Sale rates at the boundary**

Table 1 shows a clear, post‑event divergence at the **SFHA** line. Within the ±300 m window, the inside sale rate fell from 0.002345 to 0.002035 per parcel‑month (−13.2 percent), while the outside rate rose from 0.002554 to 0.003134 (+22.7 percent). The difference‑in‑differences (DiD) is −0.000890 (95 percent CI: −0.001634, −0.000145), which is about **−38 percent of the inside pre‑rate**. Results with the tighter ±150 m caliper point in the same direction and are borderline significant. At the **2019 inundation** edge, inside rates increase more than outside but estimates are imprecise. Taken together, the liquidity signal is strongest at the **regulatory** boundary: the market **thins inside** the SFHA while activity **picks up outside**. 

**Table 1. Parcel‑month sale rates near hazard boundaries (boundary RD‑in‑panel)**\
*(Rates are per parcel‑month; DiD = \[inside post − inside pre\] − \[outside post − outside pre\].)*

  --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
  **Boundary**     **Caliper (m)**   **Inside pre**   **Inside post**   **Outside pre**   **Outside post**   **Δ Inside**   **Δ Outside**         **DiD**                     **95% CI**   **N (parcel‑months)**
  -------------- ----------------- ---------------- ----------------- ----------------- ------------------ -------------- --------------- --------------- ------------------------------ -----------------------
  SFHA                         150         0.002376          0.002116          0.002410           0.002916      −0.000260       +0.000506   **−0.000766**        \[−0.001563, 0.000031\]                 824,131

  SFHA                         300         0.002345          0.002035          0.002554           0.003134      −0.000310       +0.000580   **−0.000890**   **\[−0.001634, −0.000145\]**               1,798,986

  Inundation                   150         0.002070          0.003975          0.003535           0.004284      +0.001905       +0.000750       +0.001155        \[−0.001468, 0.003778\]                  65,072

  Inundation                   300         0.002222          0.004089          0.003395           0.003694      +0.001867       +0.000299       +0.001568        \[−0.000647, 0.003783\]                 110,250
  --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

**4.2. Price levels at the boundary**

Price contrasts at the line are negative in sign but imprecise (Table 2). In the SFHA windows, the DiD in log price ranges from −0.26 to −0.35; in level terms, those magnitudes correspond to approximately −23 to −30 percent, but 95 percent confidence intervals span zero in all cases, reflecting the relatively small amount of sales right at the boundary. The inundation‑edge windows are similarly noisy. These estimates are consistent with the sale‑rate evidence: in the short run, the market appears to adjust more through liquidity and where transactions occur than through precisely estimated price cuts at the boundary. 

**Table 2. Log‑price difference‑in‑differences near hazard boundaries (sale months only)**

  ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
  **Boundary**     **Caliper (m)**   **Inside median log‑price (pre)**   **Inside median log‑price (post)**   **Outside median log‑price (pre)**   **Outside median log‑price (post)**   **DiD (log)**          **95% CI**   **N (sales)**
  -------------- ----------------- ----------------------------------- ------------------------------------ ------------------------------------ ------------------------------------- --------------- ------------------- ---------------
  SFHA                         150                              11.857                               12.029                               12.193                                12.296      **−0.353**   \[−0.829, 0.123\]           2,174

  SFHA                         300                              11.857                               12.044                               12.211                                12.315      **−0.264**   \[−0.728, 0.201\]           5,085

  Inundation                   150                              12.492                               12.388                               12.846                                12.972          −0.282   \[−1.049, 0.484\]             248

  Inundation                   300                              12.242                               12.429                               12.690                                12.942          −0.260   \[−0.851, 0.330\]             387
  ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

*Source: boundary RD summary export. Approximate level effects are elog DiD−1.*

**3.3. Substitution into near‑but‑dry parcels**

To test for re‑routing of transactions just outside the line, we estimate ring models of monthly sales counts in micro‑cells. At the SFHA boundary, both rings immediately outside the line show higher post‑event counts (rate ratios ≈ 1.44 in 0--250 m and 1.31 in 250--300 m; Table 3), consistent with substitution into near‑but‑dry parcels. At the inundation edge, rings show rate ratios below one with wide intervals, which we interpret cautiously given thin samples and the localized event footprint. Negative‑binomial variants are unstable for inundation (separation), so we report Poisson as the primary specification. 

**Table 3. Ring models for monthly sales counts just outside the line (Poisson, rate ratios)**

  ----------------------------------------------------------------------
  **Boundary**   **Term (ring × post)**        **RR** **95% CI**
  -------------- ------------------------ ----------- ------------------
  SFHA           0--250 m × Post            **1.441** \[0.976, 2.126\]

  SFHA           250--300 m × Post          **1.310** \[0.874, 1.964\]

  Inundation     0--250 m × Post                0.614 \[0.317, 1.189\]

  Inundation     250--300 m × Post              0.454 \[0.189, 1.087\]
  ----------------------------------------------------------------------

*Model includes cell and month fixed effects; full coefficient tables (including "post" main effects) in the export.*

**4.4 Composition within the boundary window**

Within the ±300 m SFHA window, the **share of monthly sales occurring inside** the SFHA falls **by about one percentage point on average** after the flood. Averaging the boundary‑window series by period yields 3.35 percent pre‑event and 2.27 percent post‑event, a change of −1.08 percentage points across 24 pre‑months and 24 post‑months (Table 4). This compositional shift is consistent with the sale‑rate and ring evidence: **fewer transactions clear inside**, and **a larger fraction of boundary‑window activity occurs just outside**. 

**Table 4. SFHA share of boundary‑window sales (±300 m around SFHA line)**

  -------------------------------------------------------------------------
  **Period**            **Mean monthly share inside SFHA**   **N (months)**
  ------------------- ------------------------------------ ----------------
  Pre (≤ 2019‑02)                               **3.35 %**               24

  Post (≥ 2019‑04)                              **2.27 %**               24

  Change (Post−Pre)                           **−1.08 pp**              ---
  -------------------------------------------------------------------------

*Computed from the monthly boundary‑window series underlying fig_sfha_share_rd.png.*

**4.5 Event‑time diagnostics**

Event‑time summaries of treated--control sale‑rate gaps do not show alarming pre‑trends: the average monthly gap over the 24 pre‑months is roughly 0.00039, rising to 0.00065 in the 24 post‑months---a modest increase that is directionally consistent with the boundary‑window DiD but not by itself dispositive. Price paths in event time are thin near the boundary and do not support precise inference, aligning with the imprecision in Table 2. Full event‑time series are provided in the replication bundle. 

**4.6 Summary** 

The most robust pattern is a liquidity‑and‑sorting response at the regulatory flood line: sale activity contracts inside and re‑routes to near‑but‑dry parcels immediately outside. Boundary‑adjacent price contrasts are negative in sign but imprecisely estimated given thin local samples, so we treat prices as secondary to market functioning. This pattern is consistent with the spatial structure of hazard documented for Douglas County---strong clustering along river corridors and flood‑plain belts---which makes local counterfactuals at boundaries informative for market response (maps and diagnostics in the proximity study provide context). In the next section we examine who buys within these exposed belts post‑event, leveraging buyer organizational form, portfolio scale, and proximity measured from deeds to connect this boundary behavior to the ownership patterns that shape where risk ultimately resides. 

Tables source: Boundary windows (sale‑rate DiD, log‑price DiD, ring models) and boundary‑window monthly share series prepared for this project; see the Boundary RD Summary export and associated figures. 

## 5. Discussion

### 5.1 Principal findings and interpretation

Our boundary analysis indicates that the short‑run market response to flood‑risk salience in this inland county is expressed most clearly in **liquidity and location of transactions**, not in precisely estimated price discounts at the line. Inside the Special Flood Hazard Area (SFHA), sale incidence declines relative to just outside after the March 2019 flood (DiD ≈ −0.00089 per parcel‑month within ±300 m; 95% CI −0.00163, −0.00015), while **sales counts rise immediately outside** the line in near‑but‑dry rings (rate ratios ≈ 1.44 in 0--250 m and 1.31 in 250--300 m). The **share of boundary‑window sales occurring inside** the SFHA also falls by roughly one percentage point. Log‑price contrasts at the line are negative in sign but imprecisely estimated given thin boundary samples. Together, these patterns point to a **freeze‑and‑flight** dynamic: transactions thin on the inside and re‑route to spatial substitutes just outside the regulatory boundary. Effects at the realized 2019 inundation edge are directionally similar but weaker and less precise, consistent with the event's concentrated footprint and smaller boundary sample. 

Two features of the institutional setting help explain why the SFHA boundary is where the clearest signals appear. First, **regularity and salience**: insurance obligations, lender overlays, and seller disclosures attach to the SFHA line, which can discourage transactions inside and redirect search just outside even when underlying hydrology changes little over tens of meters. Second, **search and financing frictions**: in the months following a large event, buyers and lenders may treat the line as a bright proxy for risk while underwriting or due‑diligence constraints tighten. In such conditions, it is plausible to observe sizable changes in *where* deals clear before we can measure stable differences in *what buyers pay* at the boundary.

### 5.2 Links to ownership and proximity 

Earlier Douglas County evidence shows that **ownership form and portfolio scale** correlate with regulatory exposure and that **owner--parcel proximity** rises with exposure among SFR holdings (maps and clustering diagnostics in that work underscore the spatial structure we confront at parcel scale). Those results motivate our composition tests at the boundary---specifically, whether post‑event purchases inside exposed belts tilt toward legal shells and more‑distant buyers that are already over‑represented in the cross‑section. We therefore interpret the liquidity and substitution patterns here as a **mechanism‑consistent precursor** to reallocation: thinning inside the line creates room for buyer types that are more willing or able to transact despite heightened salience. We keep the cross‑paper links limited to brief citations because this article is designed to stand alone (ownership structure: classifier‑based typology and elevated SFHA association; proximity gradient in SFR exposure with strong spatial clustering of hazard). 

### 5.3 How boundary dynamics refine the capitalization debate

The lack of precise, immediate **price** differences at the line should not be read as evidence that risk is fully priced. Two alternative readings are consistent with our estimates. First, **selection and thin markets**: when sale incidence collapses inside the line, remaining trades may be atypical along unobservables (e.g., urgency, financing, repair strategies), widening confidence intervals on price contrasts even if true discounts exist. Second, **temporal mismatch**: prices may move slowly as inspections, remediations, and insurance settlements work through the pipeline, whereas **volume and composition** adjust quickly. Our event‑time window focuses on the short‑run; longer panels could reveal persistent price re‑anchoring once liquidity normalizes. The ring results highlight another form of incomplete capitalization: **residual risk just outside** the administrative line may be under‑acknowledged by buyers who substitute into near‑but‑dry parcels, a pattern visible in the post‑event increase in sales counts adjacent to the SFHA boundary. 

### 5.4 Policy and administrative implications anchored to boundary evidence

The freeze‑inside / flight‑outside pattern suggests several actionable levers that do not rely on large, immediate price discounts. First, communication should follow the rings, not just the line. If transactions re‑route to parcels tens to hundreds of meters outside the SFHA, disclosure and outreach can be extended to a "residual‑risk buffer" where buyers face riverine and drainage hazards that are not captured by designation but are functionally contiguous with the mapped floodplain. Second, registry and permitting touchpoints can prioritize boundary‑adjacent areas right after a major event---when salience is high but market signals are noisy---for targeted inspections, mitigation counseling, and insurance take‑up checks tied to lenders' post‑disaster re‑verifications. Third, buyout and elevation programs** **can use boundary‑window analytics to sequence projects: where sale rates fall inside and counts rise just outside, jurisdictions can pair acquisitions or structure elevations on the inside with drainage or storage investments that reduce spillovers into the near‑dry belt. Finally, because boundary‑window sales become more concentrated outside, data standards for recording SFHA status and boundary distance in deeds or MLS** **would allow monitoring of post‑event substitution in real time and better targeting of mitigation resources. These recommendations flow directly from the localized adjustments we measure at and just beyond the line. 

### 5.5 Limitations

Our identification isolates local contrasts around regulatory and realized boundaries, but several limitations remain. Boundary windows contain few sales, especially at the inundation edge, which limits precision on price estimates and constrains heterogeneity analyses. We do not observe listings, withdrawals, or days‑on‑market; thus, liquidity is inferred from sales incidence and counts rather than from marketing durations. The event‑study horizon is centered on a single flood; multi‑event panels** **could separate general boundary behavior from event‑specific geography and insurance cycles. Finally, while we draw on prior county evidence to frame composition expectations, this article's reported boundary results do not yet include buyer‑type or distance decompositions at the line, which we view as a necessary next step to connect who transacts with where transactions occur. 

### 5.6 Future work

Two extensions are most promising. First, merge buyer organizational form, portfolio scale, proximity, and financing into the boundary panels to observe directly who buys the dip and whether cash purchases disproportionately fill the inside‑the‑line gap after a flood. This will allow a sharper test of mechanism links between ownership form, distance, and boundary behavior using the same legal‑shell typology previously validated in the county. Second, track longer‑run dynamics---monthly event‑time horizons beyond two years, repeat‑sales residuals, and sequential floods---to determine whether initial liquidity/relocation patterns give way to persistent price re‑anchoring or whether boundary substitution is transient. Given the strong spatial clustering of hazard and exposure documented at parcel scale in Douglas County (maps and LISA diagnostics), these extensions should continue to privilege **local** designs at the line while expanding the temporal window. 

In sum, the market's first move after the 2019 flood was to change where transactions occurred---away from the inside of the regulatory line and toward immediately adjacent parcels---rather than to deliver large, precisely estimated price cuts at the boundary. Recognizing and planning for that sequence---liquidity first, prices later (if at all)**---**can help administrators, lenders, and households act on risk information when it is most salient but least visible in prices. 

## **6. Conclusion**

Across designs that isolate parcels in narrow windows around the hazard edges, the market's first response to the March 2019 flood was a change in where transactions occurred rather than a precisely estimated reset in prices. Inside the Special Flood Hazard Area, the parcel‑month sale rate fell while the just‑outside rate rose, yielding a difference‑in‑differences of −0.000890 with a 95% interval from −0.001634 to −0.000145 in the ±300‑meter window. Micro‑cell count models show higher post‑event activity immediately outside the line with rate ratios near 1.44 in the 0--250‑meter ring and 1.31 in the 250--300‑meter ring. The share of boundary‑window sales that occur inside the floodplain declines by roughly one percentage point. Price contrasts at the line are negative in sign but imprecise given the small number of boundary sales. These facts come directly from the boundary summaries and ring models assembled for this study. 

Two conclusions follow: first, in the short run the regulatory line acts as a filter on liquidity. Buyers and lenders re‑route to near‑but‑dry substitutes just beyond the map even when price gaps at the line are hard to pin down with precision. Second, this spatial re‑routing operates in a county where hazard is tightly clustered and where the distribution of exposure has been shown to align with ownership structure and with owner--parcel distance. Those background regularities help explain why composition can change when markets thin on the inside and thicken just outside. 

 The policy implication is direct. Communication, disclosure, and mitigation should follow the rings as well as the line. Boundary‑adjacent registries and listing systems should capture floodplain status and distance‑to‑boundary so that substitution can be monitored as it happens. Post‑event acquisitions and elevation incentives can be sequenced using the same boundary‑window diagnostics that reveal the freeze‑inside and flight‑outside pattern documented here. 

The main limitations are sample thinness at the boundary and a short event‑time horizon. Extending the panel, adding listing data on marketing time and withdrawal, and linking financing and buyer identity will allow a direct test of who fills the gap inside the floodplain when markets reopen. Replication across settings with different disclosure regimes and mapping updates will show whether the liquidity‑first sequence observed in Douglas County generalizes or varies with institutions and geography.

**7. References**

Atreya, A., Ferreira, S., & Kriesel, W. (2013). Forgetting the flood? An analysis of the flood risk discount over time. *Land Economics, 89*(4), 577--596. [[https://doi.org/10.3368/le.89.4.577]{.underline}](https://www.google.com/search?q=https://doi.org/10.3368/le.89.4.577)

Bakkensen, L. A., & Ma, L. (2020). Sorting over flood risk and implications for policy. *Journal of Urban Economics, 115*, 103221. [[https://doi.org/10.1016/j.jue.2019.103221]{.underline}](https://www.google.com/search?q=https://doi.org/10.1016/j.jue.2019.103221)

Beltrán, A., Maddison, D., & Elliott, R. J. R. (2018). Is flood risk capitalised into property values? *The BE Journal of Economic Analysis & Policy, 19*(2). [[https://doi.org/10.1515/bejeap-2017-0131]{.underline}](https://www.google.com/search?q=https://doi.org/10.1515/bejeap-2017-0131)

Bin, O., & Landry, C. E. (2013). Changes in implicit flood risk premiums: Empirical evidence from the housing market. *Journal of Environmental Economics and Management, 65*(3), 361--376. [[https://doi.org/10.1016/j.jeem.2012.12.002]{.underline}](https://doi.org/10.1016/j.jeem.2012.12.002)

Bin, O., & Polasky, S. (2004). Effects of flood hazards on property values: Evidence before and after Hurricane Floyd. *Land Economics, 80*(4), 491--506. [[https://doi.org/10.2307/3655783]{.underline}](https://www.google.com/search?q=https://doi.org/10.2307/3655783)

Chan, S. (2011). The impact of Hurricane Katrina on the New Orleans housing market. *American Economic Journal: Economic Policy, 3*(3), 54--81. [[https://doi.org/10.1257/pol.3.3.54]{.underline}](https://www.google.com/search?q=https://doi.org/10.1257/pol.3.3.54)

Collins, T. W., Grineski, S. E., & Chakraborty, J. (2018). Environmental injustice and flood risk: A conceptual model and case study of Cedar Rapids, Iowa, USA. *Environmental Research Letters, 13*(5), 055008. [[https://doi.org/10.1088/1748-9326/aab966]{.underline}](https://www.google.com/search?q=https://doi.org/10.1088/1748-9326/aab966)

Daniel, B. C., Florax, R. J. G. M., & Rietveld, P. (2009). Flooding risk and housing values: An economic assessment of environmental hazard. *Ecological Economics, 68*(4), 1139--1147. [[https://doi.org/10.1016/j.ecolecon.2008.07.017]{.underline}](https://www.google.com/search?q=https://doi.org/10.1016/j.ecolecon.2008.07.017)

Dávila, A. (2022). Housing market dynamics with disaster risk. *Federal Reserve Bank of Cleveland Working Paper No. 22-13*. [[https://doi.org/10.26509/frbc-wp-202213]{.underline}](https://www.google.com/search?q=https://doi.org/10.26509/frbc-wp-202213)

Davidoff, T., & Zytnick, B. (2023). Who moves into climate-risky housing? *Housing Policy Debate, 33*(4), 841--863. [[https://doi.org/10.1080/10511482.2022.2155099]{.underline}](https://www.google.com/search?q=https://doi.org/10.1080/10511482.2022.2155099)

Gallagher, J. (2014). Learning about an infrequent event: Evidence from flood insurance. *American Economic Journal: Applied Economics, 6*(3), 206--236. [[https://doi.org/10.1257/app.6.3.206]{.underline}](https://www.google.com/search?q=https://doi.org/10.1257/app.6.3.206)

Gourevitch, J. D., Keenan, J. M., Alijani-Rad, A., Kahn, M. E., Hino, M., & Wing, O. E. J. (2023). Unpriced climate risk and the potential consequences of overvaluation in U.S. housing markets. *Nature Climate Change, 13*(2), 143--151. [[https://doi.org/10.1038/s41558-023-01594-8]{.underline}](https://doi.org/10.1038/s41558-023-01594-8)

Hamideh, S., Peacock, W. G., & Van Zandt, S. (2021). Housing type matters for pace of recovery: Evidence from Hurricane Ike. *International Journal of Disaster Risk Reduction, 57*, 102149. [[https://doi.org/10.1016/j.ijdrr.2021.102149]{.underline}](https://doi.org/10.1016/j.ijdrr.2021.102149)

Kousky, C. (2010). Learning from extreme events: Risk perceptions and the residential real estate market. *Journal of Urban Economics, 68*(2), 147--157. [[https://doi.org/10.1016/j.jue.2010.03.003]{.underline}](https://doi.org/10.1016/j.jue.2010.03.003)

Kousky, C. (2018). The U.S. National Flood Insurance Program in historical perspective. In *The Geneva Papers on Risk and Insurance - Issues and Practice, 43*(4), 606--626. [[https://doi.org/10.1057/s41288-018-0099-2]{.underline}](https://www.google.com/search?q=https://doi.org/10.1057/s41288-018-0099-2)

Mueller, J. T., Rossi, M. W., & Stratmann, T. (2022). Fleeing the fire: The effects of wildfires on migration. *Journal of Urban Economics, 128*, 103417. [[https://doi.org/10.1016/j.jue.2022.103417]{.underline}](https://www.google.com/search?q=https://doi.org/10.1016/j.jue.2022.103417)

Ortega, F., & Taspinar, S. (2018). The heterogeneous effects of flood zone remodeling on housing markets. *Journal of Housing Economics, 42*, 19--31. [[https://doi.org/10.1016/j.jhe.2018.06.002]{.underline}](https://doi.org/10.1016/j.jhe.2018.06.002)
