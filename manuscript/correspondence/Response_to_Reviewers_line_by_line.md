---
title: "Response to Reviewers (Line-by-line) — EPB-2025-0878"
geometry: margin=1in
mainfont: "Times New Roman"
---
# Response to Reviewers (Line-by-line) — EPB-2025-0878
This document responds line-by-line to each reviewer critique. For every critique, the Changes made section lists the specific manuscript edits made in the revised version (with direct quotes of revised text where helpful).
If a critique has not been addressed in the provided revision, its Changes made section is left blank (as requested).


## Referee 1

### 1) "The authors should buttress the argument about why ownership patterns are important for flood risk…"

**Reviewer critique:**
> *The authors should buttress the argument about why ownership patterns are important for flood risk. This is not done in a convincing manner in the first paragraph of manuscript which references studies on valuation and uncertainty, but does not explain or frame how these relate to different tenure structures or institutional patterns of ownership.*

**Changes made**
- Rewrote the opening framing to foreground a governance/implementation rationale for "who holds mapped exposure," rather than relying on pricing/valuation alone. Added to Abstract and Introduction:
  > "Regulatory flood maps identify where hazard is located but leave a parallel governance question under-specified: who holds that mapped exposure."
  > "This ownership blind spot matters because the implementation of floodplain governance is mediated through owners. Many of the key levers available to cities and counties—enforcement of floodplain development standards, targeted outreach and mitigation assistance, participation in buyouts or property elevations, and post-disaster repair and recovery coordination—ultimately depend on identifying, contacting, and negotiating with the entities that own the exposed housing stock."
  > "In other words, hazard is spatial, but governance is organizational: regulatory exposure is administered through property owners and the legal forms through which they hold assets."

- Added a second argument showing ownership remains governance-relevant even under full risk pricing (to avoid hinging motivation on any single capitalization mechanism). Added to Introduction:
  > "Even if flood risk were perfectly understood and priced, owners would still differ in their capacity and willingness to manage, insure, mitigate, or exit flood-exposed property. These differences are institutional: legal organizational form shapes liability, financing and underwriting pathways, administrative legibility, and compliance strategies, while portfolio footprint shapes operational capacity, diversification, and bargaining power with contractors, insurers, and regulators."


### 1a) Clarify why the "five mutually exclusive organizational forms" are relevant and connect to broader literature

**Reviewer critique:**
> *The authors need to clarify why the "five mutually exclusive organizational forms" are relevant and connect this to some greater literature in terms of a) institutional structure and ownership behavior, and b) institutional structure and ownership behavior related to hazards.*

**Changes made**
- Added a clearer conceptual bridge from legal form and portfolio footprint to governance-relevant behavior and capacity in Introduction:
  > "Even if flood risk were perfectly understood and priced, owners would still differ in their capacity and willingness to manage, insure, mitigate, or exit flood-exposed property. These differences are institutional: legal organizational form shapes liability, financing and underwriting pathways, administrative legibility, and compliance strategies, while portfolio footprint shapes operational capacity, diversification, and bargaining power…"

- Added hazard-governance linkage (why ownership matters for implementation and the "unit of engagement") in Literature Review 2.1:
  > "However, maps and parcel overlays primarily describe where regulatory exposure lies; they do not, by themselves, describe how that exposure is organized across owners. This distinction is consequential because most governance interventions are implemented through property owners rather than through parcels in the abstract."

- Added explicit acknowledgement that categories are not a proxy for rental/occupancy status, and that any form can include both types (to support interpretation) in Methods 3.3:
  > "Table 2 summarizes how these categories are identified from owner-of-record strings, the interpretive role they play in this study, and key caveats, including the fact that any category may contain both owner-occupants and landlords."


### 1a(i)) "More justification and development from 140–154… rationale of the ownership typology… literature review of other projects/classification"

**Reviewer critique:**
> *There should also be more justification and development from 140-154 as to the rational of the ownership typology. And a literature review of other similar projects and classification.*

**Changes made**
- Expanded and reorganized the ownership-typology justification in Introduction ("ownership blind spot," limits of absentee proxies, why legal form + footprint) including:
  > "Despite its importance, ownership is often represented using coarse proxies that blur analytically meaningful distinctions… What is needed is a tractable, transferable way to make ownership structure legible at parcel scale…"

- Added an explicit measurement/typology section in the Literature Review 2.4 situating the approach relative to ownership-scale and obscurity work:
  > "Most empirical housing work that studies 'investor' or 'institutional' ownership relies on simplified typologies… definitions and thresholds vary substantially, complicating cross-study comparisons…"
  > "In response, an emerging measurement literature has emphasized methods to identify ownership scale and reduce obscurity by linking entities across shell structures and networks…"


### 1b) Complement vs competing definitions; "Who Owns America" and robustness to other classifications

**Reviewer critique:**
> *Does this work complement or provide confusing competing definitions with efforts like: Who Owns America…*

**Changes made**
- Added positioning that the study complements rather than competes with beneficial-owner/scale-consolidation efforts in Literature Review 2.4:
  > "This study is designed to complement—rather than compete with—beneficial-owner consolidation approaches. We retain the owner-of-record legal shell as a meaningful administrative unit because it is the entity appearing in land records and can be consequential for governance and liability."


### 1b(ii)) "If different can the analysis be run with other classifications to ensure robustness."

**Reviewer critique:**
> *If different can the analysis be run with other classifications to ensure robustness.*

**Changes made**

We address classification robustness in two ways: (1) testing sensitivity to classification uncertainty within our legal-form approach, and (2) clarifying why our legal-form categories would be captured by alternative methodologies.

*Classification uncertainty robustness:*

- Re-estimated the primary models on a high-confidence classification subset (predicted probability ≥ 0.80, n ≈ 179,000) to test robustness to classification uncertainty. Key contrasts remain significant with directionally consistent estimates. Added to Methods 3.8:
  > "Finally, we assess robustness of key estimates to classification uncertainty by re-estimating models on the high-confidence label subset…"

- Reported robustness results in Section 4.7:
  > "Estimates are also robust to restricting the sample to high-confidence classifier labels (predicted probability ≥ 0.80)…"

- Validated the BERT classifier against a rule-based baseline and ablation variants in the Supplementary Material (Model Card). The rule-based classifier achieves Macro F1 = 0.51, while BERT achieves 0.98, and ablated BERT (with legal-form tokens removed) achieves 0.89. These comparisons demonstrate that the BERT classifier substantially outperforms a simple alternative, while the ablation shows the classifier relies on contextual patterns beyond explicit legal-form tokens. Reported in Supplementary Material:
  > "Overall performance comparison: BERT — Original: Accuracy 0.993, Macro F1 0.980; BERT — Ablated: Accuracy 0.954, Macro F1 0.885; Rule-based Baseline: Accuracy 0.841, Macro F1 0.510."

*Compatibility with alternative ownership typologies:*

- The legal-form categories used here (Individual, LLC, Corporation, Trust, Gov/Nonprofit) are standard administrative distinctions encoded in owner-of-record name strings. Any classification approach—whether rule-based, ML-based, or beneficial-ownership linkage methods like "Who Owns America"—would necessarily identify these same legal shells as an intermediate step before consolidating to beneficial owners. The key finding that LLC-owned parcels exhibit elevated SFHA exposure would therefore be captured by alternative approaches that retain legal-form information.

- The portfolio-footprint dimension (single- vs. multi-parcel) is conceptually aligned with scale-based approaches: our within-county footprint measure captures a local version of the ownership-scale construct central to "Who Owns America." While beneficial-ownership consolidation might reclassify some single-shell LLCs as part of larger portfolios (if controlled by the same beneficial owner), this would if anything strengthen the finding that entity-owned portfolios are over-represented in SFHAs relative to individual households. Added to Literature Review 2.4:
  > "This study is designed to complement—rather than compete with—beneficial-owner consolidation approaches. We retain the owner-of-record legal shell as a meaningful administrative unit…"

- We acknowledge as a limitation that beneficial-ownership linkage is not available in current data, but note that this would be a productive extension. Added to Limitations 5.4:
  > "Future work can combine shell-level typologies with beneficial ownership linkage methods to assess how much the organization of exposure changes when entities are consolidated into networks or ultimate owners."


### 1c) "Address the question of unclear title and how this fits into the problem."

**Reviewer critique:**
> *The authors also should address the question of unclear title and how this fits into the problem.*

**Changes made**
- Added explicit clarification early that the unit is owner-of-record (not beneficial owner), and why that still answers a governance-relevant version of "who owns." Added to Introduction:
  > "At the same time, we acknowledge an important limitation: owner-of-record entities are not the same as ultimate beneficial owners. LLCs may be owned by other entities or individuals, and consolidating beneficial ownership may be desirable for some research and enforcement purposes. Here we retain the owner-of-record unit because it is the entity directly appearing in administrative land records and because the legal shell itself can be meaningful for governance, enforcement, and liability structures…"


### 1d) "Clarity about both organizational form and portfolio scale… and how those lead to hypotheses…"

**Reviewer critique:**
> *This begins to be fleshed out on line 59, but really needs much more attention. Clarity about both organizational form and portfolio scale, and how those lead to hypotheses are necessary in introducing the paper.*

**Changes made**
- Added a dedicated explanation of the two dimensions and why each matters in Introduction:
  > "Within this governance-oriented frame, legal form and local portfolio footprint provide two observable dimensions that plausibly organize exposure in different ways… Meanwhile, within-form contrasts by local footprint can be especially revealing because they distinguish small, single-asset entities from owners operating multiple properties."

- Added explicit research questions and hypotheses with stated motivation in Introduction:
  > "We organize the analysis around three linked questions: (1) How do unadjusted SFHA rates differ… (2) Do these differences persist… (3) Within organizational forms, do single-parcel and multi-parcel entities exhibit distinct patterns…?"
  > "H1 (Organizational form)… H2 (Within-LLC footprint)… RQ3 (Within-form heterogeneity beyond LLCs)…"


### 1e) "Many different ideas… need to be more systematically presented in relationship to the hypotheses."

**Reviewer critique:**
> *In sum, there are many different ideas being presented and they need to be more systematically presented in relationship to the hypotheses.*

**Changes made**
- Re-structured the Introduction to move from: motivation → measurement gap → study design → questions → hypotheses → contributions, and explicitly labeled H1/H2/RQ3.
- Clarified interpretive stance (observational associations; governance-forward) to keep mechanisms separate from what is tested. Added to Introduction:
  > "Our framing is deliberately governance-forward and observational… we treat ownership as a correlate of selection into, retention of, and organization of regulatory exposure."


## 2) Single county particularities

### 2) "Provide discussion/background of how much the single county's particularities influence this analysis."

**Reviewer critique:**
> *The authors should provide some discussion and background of how much the single county's particularities influence this analysis.*

**Changes made**
- Added explicit statements that the analysis is county-specific and uses a within-county footprint, plus limitations and next steps for multi-county/linked designs. Added to Methods 3.3 and Limitations 5.4:
  > "The local portfolio footprint measure captures the number of SFR parcels held under the same owner-of-record name within Douglas County. This jurisdiction-bounded measure reflects the footprint most directly relevant for local administration, while not observing holdings outside the county or holdings distributed across multiple legal shells."
  > "Second, the analysis relies on owner-of-record entities and within-county footprint… but does not observe holdings outside Douglas County… Future work can combine shell-level typologies with beneficial ownership linkage methods…"


### 2a) "Portfolio scale only at county level is problematic…"

**Reviewer critique:**
> *The measure of portfolio scale at only the county level is problematic. What if an owner has a huge portfolio in the region or national, but a light footprint locally, what if they have tons of property in an abutting count?*

**Changes made**
- Added explicit limitation language regarding footprint measurement in Methods 3.3 and Limitations 5.4:
  > "This jurisdiction-bounded measure reflects the footprint most directly relevant for local administration, while not observing holdings outside the county or holdings distributed across multiple legal shells."
  > "Future work… [should] address how scale operates in regional housing markets that cross county boundaries."


### 2b) "Describe the county, the amount of land in floodplain… in each category… industrial and commercial."

**Reviewer critique:**
> *Also, describe the county, the amount of land in floodplain and the amount in each of the different categories, and the amount in industrial and commercial.*

**Changes made**
- Added explicit county-wide SFHA prevalence for single-family residential parcels in Methods 3.2:
  > "Under this majority-area rule, 2,029 of 184,333 SFR parcels (≈1.10%) are SFHA-exposed in the full county roll."

- Added descriptive breakdowns of ownership composition and SFHA prevalence across the ten owner-form × footprint groups in Results 4.1–4.2 (e.g., ownership shares and unadjusted SFHA rates), including:
  > "Individuals hold 82.60% of SFR parcels, followed by LLCs (11.12%), trusts (3.48%), government/nonprofit entities (1.43%), and corporations (1.36%)…"
  > "Two groups have especially high raw SFHA rates: single-parcel LLCs (4.83%) and single-parcel corporations (4.07%)…"

- Added explicit limitation that the scope is single-family residential and does not cover commercial/industrial/multifamily in Limitations 5.4:
  > "Additionally, the analysis is restricted to single-family residential parcels; commercial, industrial, and multifamily properties in SFHAs may exhibit different ownership patterns and are not examined here."


### 2b(i)) "Amount in industrial and commercial."

**Reviewer critique:**
> *…and the amount in industrial and commercial.*

**Changes made**

- The analysis is restricted to single-family residential (SFR) parcels by design. This scope is stated explicitly in Limitations 5.4:
  > "Additionally, the analysis is restricted to single-family residential parcels; commercial, industrial, and multifamily properties in SFHAs may exhibit different ownership patterns and are not examined here."

- We do not report industrial/commercial SFHA prevalence because the ownership classifier and study design were developed specifically for SFR parcels. The BERT classifier was fine-tuned on hand-labeled owner-of-record names drawn from the SFR housing stock, where naming conventions (e.g., personal names, LLC suffixes, trust markers) follow patterns common to residential ownership. Commercial and industrial parcels exhibit different naming conventions—including corporate subsidiaries, holding companies, and institutional investors with more complex entity structures—that would require separate training data and model validation to classify reliably. The current study focuses on making ownership structure legible within the single-family housing stock, where flood governance intersects most directly with household-level vulnerability and residential policy levers.


## 3) Regulatory floodplain imperfections; insurance pricing; Risk Rating 2.0; temporal strategies; acquisition vs cross-section

### 3) "Imperfections of regulatory floodplain… (no mention of Risk Rating 2.0…)… NPV of average annual losses."

**Reviewer critique:**
> *The imperfections of the regulatory floodplain are one issue to consider and are different actors interacting with risk differently within and outside of it in terms of insurance prices (no mention of Risk Rating 2.0, which should influence behavior post and pre 2022, so when a parcel is acquired should matter) or considerations of the net present value of average annual losses.*

**Changes made**
- Clarified that SFHA is a regulatory boundary, not comprehensive hazard, and that the outcome is mapped exposure rather than losses. Added to Methods 3.2:
  > "SFHAs are the mapped regulatory zones… making them a policy-relevant boundary for governance even though they do not represent the full set of flood hazards… Accordingly, the outcome is interpreted as mapped regulatory exposure rather than a direct measure of expected flood losses."

- Added a sensitivity check on the SFHA boundary definition (majority-area vs ≥10% overlap) in Methods 3.2:
  > "Because some parcels intersect SFHA boundaries only in small slivers, we assess sensitivity… using an alternative exposure rule that classifies a parcel as exposed if at least 10% of its area overlaps an SFHA…"

- Added explicit discussion of Risk Rating 2.0 timing in Limitations 5.4:
  > "FEMA's Risk Rating 2.0, implemented in 2021–2022, fundamentally restructured flood insurance pricing… the 2022 snapshot analyzed here cannot distinguish ownership patterns that predate versus postdate this regime change."


### 3a) "Institutionally different actors may have different time strategies…"

**Reviewer critique:**
> *To wit, institutionally different actors may have different time strategies which could influence risk behavior beyond the hypotheses offered.*

**Changes made**
- Added explicit acknowledgment that cross-sectional data cannot identify acquisition vs retention vs restructuring, and that timing strategies require longitudinal designs. Added to Limitations 5.4:
  > "Third, the design is cross-sectional. A single assessment roll cannot distinguish between acquisition, retention, and restructuring processes…"


### 3b) "Hypothesis might be better explored with acquisition models…"

**Reviewer critique:**
> *I think the author's hypothesis might be better explored with acquisition models not a cross-sectional ownership model.*

**Changes made**
- Added explicit recommendation for transaction/acquisition designs in Limitations 5.4:
  > "Transaction-based designs linking deeds and sale dates to buyer legal form would allow tests of selection into SFHAs…"


## 4) Classification vs analytical methods balance; spatial modeling; development patterns; single snapshot; limitations

### 4) "Classification algorithms should be de-prioritized… and analytical methods justified…"

**Reviewer critique:**
> *This is a paper that is based on hypotheses. Therefore, the classification algorithms should be de-prioritized and included in replicable appendices, and the analytical methods justified in greater detail.*

**Changes made**
- Moved classifier configuration/training/coverage detail to Supplementary Material and shortened main-text classifier discussion. Added to Methods 3.4:
  > "The classifier produces probabilistic outputs by class; model configuration, training, and coverage are documented in the Supplementary Material…"

- Expanded and clarified the statistical estimand and modeling rationale in Methods 3.6:
  > "We estimate associations using a modified Poisson generalized linear model with a log link for a binary outcome (Zou, 2004). This approach yields coefficients that can be exponentiated and interpreted as risk ratios…"


### 4a) "Why not a spatial Poisson model? Does the neighborhood FE model get analyzed for spatial clustering? Any what is the neighborhood scale here?"

**Reviewer critique:**
> *Why not a spatial poison model? Does the neighborhood fixed effects model get analyzed for spatial clustering? Any what is the neighborhood scale here?*

**Changes made**
- Defined the neighborhood fixed-effect unit explicitly in Methods 3.6–3.7:
  > "α_n denotes neighborhood fixed effects defined by assessor appraisal neighborhoods."

- Added explicit statements that fixed effects do not eliminate all spatial dependence and added residual spatial diagnostics in Methods 3.7:
  > "Fixed effects are not assumed to remove all spatial dependence in the residuals; they are used to strengthen within-area comparisons at a policy-relevant micro-geographic unit."
  > "Because spatial dependence can persist at scales smaller than neighborhoods, we also assess residual spatial structure by computing Moran's I on primary-model residuals using an 8-nearest-neighbor spatial weights matrix…"

- Added spatial HAC (Conley) standard error robustness as an additional spatial inference check in Methods 3.7:
  > "As an additional robustness check, we re-estimate key contrasts using spatial HAC (Conley) standard errors that allow for distance-decay correlation across parcels…"

- NEW: Implemented eigenvector spatial filtering (ESF) as a direct spatial Poisson robustness check. Added to Methods 3.7:
  > "To directly address whether a spatial Poisson specification would alter the main findings, we implement eigenvector spatial filtering (ESF) as a robustness analysis (Griffith, 2003). ESF extracts eigenvectors from a row-standardized k-nearest-neighbor spatial weights matrix (k = 8) and includes them as additional controls in the Poisson GLM. These eigenvectors capture spatially structured variation at multiple scales; including progressively more eigenvectors absorbs increasingly fine-grained spatial dependence. We conduct a sensitivity analysis varying the number of spatial filters from 15 to 200 and report the resulting coefficient estimates and residual Moran's I statistics (Table S7)."

- NEW: Reported ESF sensitivity analysis results in Section 4.6:
  > "In a sensitivity analysis varying the number of spatial filters from 15 to 200 (Table S7), residual Moran's I decreases from 0.39 (neighborhood FE only) to effectively zero (0.003) with 200 filters—a 99% reduction in residual spatial autocorrelation. Across all specifications, the LLC risk ratio remains stable and statistically significant: it attenuates modestly from 1.38 (FE only) to 1.24 (FE + 200 spatial filters), with p < 0.001 in all cases. This 10% attenuation indicates that approximately one-tenth of the unadjusted LLC–SFHA association reflects spatial confounding absorbed by the filters, while the remaining association is robust to spatial structure."


### 4b) "How sure are we that these results are not reflective of development patterns and flood plain of this county."

**Reviewer critique:**
> *How sure are we that these results are not reflective of the development patterns and flood plain of this county.*

**Changes made**
- Strengthened the non-causal interpretation and explicitly framed results as conditional associations, not causal siting effects, in Introduction:
  > "We do not argue that organizational form causes parcels to be in floodplains… Instead, we treat ownership as a correlate of selection into, retention of, and organization of regulatory exposure."

- Added explicit reporting of remaining residual spatial dependence after fixed effects in Results 4.6:
  > "Moran's I computed on the primary-model residuals… is 0.392 (p < 0.001)… This remaining residual clustering indicates that SFHA exposure and/or ownership patterns still exhibit spatial structure not fully captured by neighborhood fixed effects and parcel covariates."

- NEW: ESF analysis provides strongest evidence that results are not spatial artifacts: Even when residual Moran's I is reduced to near zero (0.003), the LLC risk ratio remains significant (RR = 1.24, p < 0.001).


### 4c) "Concern about single jurisdiction / regional housing market…"

**Reviewer critique:**
> *The Gourvetic study is national, and I have the concern about the Immergluck and Law study that the focus was too much on a single jurisdiction and it overlooked the regional housing market.*

**Changes made**
- Added limitation language about county-bounded footprint and the need for cross-county/region designs in Limitations 5.4:
  > "This choice… does not observe holdings outside Douglas County… Such linkage would also help address how scale operates in regional housing markets that cross county boundaries."


### 4d) "Single snapshot… any way to look at trends in ownership over time?"

**Reviewer critique:**
> *We should be very careful about this as a single snapshot. Is there anyway to look at trends in ownership overtime?*

**Changes made**
- Added explicit cross-sectional limitation and proposed multi-year assessor panels and transaction histories in Limitations 5.4:
  > "Third, the design is cross-sectional… Transaction-based designs… and multi-year assessor panels could reveal trends in ownership composition and concentration of regulatory exposure."


### 4e) "Justify the study more in terms of major limitations…"

**Reviewer critique:**
> *The study need to justify the study more in terms of the major limitations it has.*

**Changes made**
- Expanded a dedicated Limitations and directions for future research section (Discussion 5.4) covering: (i) SFHA as regulatory boundary, (ii) owner-of-record vs beneficial owner, (iii) within-county footprint, (iv) cross-sectional design, (v) residual spatial dependence, and (vi) extensions.


## 5) Paper focus

### 5) "Decide whether hypothesis-driven, county dynamics, or methods paper…"

**Reviewer critique:**
> *The authors need to decide whether this is a hypothesis driven work about ownership and its effects, or the dynamics in one particular county, or a methods paper about classifying ownership. It is very interesting and potentially valuable, but it does not currently do enough on either one of these three fronts.*

**Changes made**
- Added explicit statement of the paper's orientation and contributions in Introduction:
  > "Our framing is deliberately governance-forward and observational."
  > "The paper contributes in three ways. Substantively… Conceptually… Methodologically…"


## Referee 2

### 1) Risk sorting mechanism clarification (discount vs overvaluation; assumptions)

**Reviewer critique:**
> *The risk sorting mechanism proposed in the manuscript doesn't make sense as currently written… Gourevitch et al. (2023) find that floodplain properties are overvalued… Why would profit-seeking floodplain owners then believe that "expected returns compensate for expected losses"… The authors should elaborate on their assumptions around floodplain discount and overvaluation…*

**Changes made**
- Removed/avoided the strong "returns compensate for losses" framing and replaced with a more cautious statement emphasizing imperfect pricing signals without assuming floodplain parcels are "good deals." Added to Introduction:
  > "A growing literature shows that flood risk is not always fully reflected in housing market signals… Even if flood risk were perfectly understood and priced, owners would still differ…"

- Added explicit clarification that the paper does not rely on a single pricing mechanism and does not make strong profit/return claims in Discussion 5.2:
  > "However, this interpretation does not require the strong claim that flood-exposed parcels offer superior risk-adjusted returns on average. The more modest implication is that a mix of constraints and strategies… can shape who holds SFHA-located parcels…"

- Added (general) language acknowledging financing/insurance channels as plausible correlates of ownership patterns in Introduction:
  > "Business entities (LLCs and corporations) may hold relatively more SFHA-designated property… they may face different financing channels and insurance constraints…"


### 1 (additional mechanism suggestion: cash purchases / avoiding insurance mandate)

**Reviewer critique:**
> *Separately, I think LLCs/corporations may have a financial advantage for buying in floodplains, since they can often purchase with cash and avoid the flood insurance mandate…*

**Changes made**

- We appreciate this suggestion, which identifies a plausible financing-related mechanism. Testing this mechanism directly is not possible with the current data—assessor records do not include transaction-level financing information (cash vs. mortgage) or insurance take-up data.

- The manuscript already accommodates this and related mechanisms by (1) framing results as conditional associations rather than causal claims, and (2) listing financing and insurance channels among the ways entity owners may differ from individuals. The relevant passage in Discussion 5.2 (which predates this revision) states:
  > "…they may face different financing channels and insurance constraints, have greater capacity to manage compliance and repair, diversify risk across a larger set of holdings, or pursue acquisition strategies that differ systematically from individual households."

- The Limitations section (5.4) already notes the need for transaction-level data to test acquisition-related mechanisms:
  > "Transaction-based designs linking deeds and sale dates to buyer legal form would allow tests of selection into SFHAs…"


### 2) Provide functional/legal definitions of the five forms and interpretive meaning; acknowledge beneficial ownership

**Reviewer critique:**
> *The authors need to provide functional and legal definitions of the five organizational forms, as well as what they intend each organizational form to represent… Moreover… LLCs are often owned by corporations and trusts. This should be acknowledged in the text.*

**Changes made**
- Added an explicit pointer that definitions and interpretive caveats are consolidated in a dedicated table. Added to Methods 3.3:
  > "Table 2 summarizes how these categories are identified from owner-of-record strings, the interpretive role they play in this study, and key caveats, including the fact that any category may contain both owner-occupants and landlords."

- Added explicit acknowledgment that owner-of-record is not beneficial owner, and why shell-level owner-of-record remains meaningful for governance and liability. Added to Introduction:
  > "Owner-of-record entities are not the same as ultimate beneficial owners… Here we retain the owner-of-record unit because it is the entity directly appearing in administrative land records and because the legal shell itself can be meaningful…"


### 3) Correlation vs causation; "isolate the effect" language; direction of causality

**Reviewer critique:**
> *The authors need to be careful not to conflate correlation with causation. One example is "The model is designed to isolate the effect of ownership…" … Organizational forms do not make parcels more or less likely to be in SFHAs…*

**Changes made**
- Removed the "isolate the effect" framing and replaced with explicit "conditional association / distribution" language. Added to Introduction:
  > "We do not argue that organizational form causes parcels to be in floodplains… Instead, we treat ownership as a correlate of selection into, retention of, and organization of regulatory exposure."

- Revised the modeling/estimand framing in Methods 3.6:
  > "The analysis does not treat organizational form as causing parcels to be located in floodplains. Instead, the regression framework estimates how mapped regulatory exposure is distributed across owner-of-record structures…"


### 4) Discussion alignment; renters/disclosures speculation; SFR vs rentals mapping

**Reviewer critique:**
> *Much of the discussion about renters and disclosures seem to be several degrees removed… There is a lot of discussion of renters, but the categories of ownership do not map neatly onto rentals… revise the discussion and the introduction…*

**Changes made**
- Added a clear statement that forms do not map cleanly to owner-occupancy vs rental, and treated that as a caveat for interpretation in Methods 3.3:
  > "…key caveats, including the fact that any category may contain both owner-occupants and landlords."

- Reframed policy relevance away from owner-type-specific standards and toward implementation capacity (concentration/tractability), in Discussion 5.3:
  > "The principal implication of these findings is not that floodplain standards, disclosure requirements, or insurance recommendations should vary by owner type. Rather, the implication is that the implementation of flood governance often depends on the organizational units that hold exposure and on how concentrated that exposure is across owners."


### 5) Within-form contrasts; mechanical probability of multi-property owners owning at least one SFHA parcel; units

**Reviewer critique:**
> *For within-form contrasts: is there any accounting for the fact that multi-property owners are more likely to own SFHA homes purely based on the fact that they own multiple properties? … I wasn't clear on the units of the hypothesis testing.*

**Changes made**
- Added an explicit note distinguishing parcel-level prevalence from owner-level "at least one SFHA parcel" mechanics, in Methods 3.6:
  > "The parcel-level prevalence estimated here is the governance-relevant quantity… Multi-parcel owners are mechanically more likely to hold at least one SFHA parcel simply by owning more parcels, but the parcel-level model estimates the share of each group's holdings that fall within SFHAs…"

- Added an owner-level intensity model (Poisson with offset) as a supplementary check, in Methods 3.6:
  > "As a direct test of whether entity-type associations persist after controlling for portfolio size, the Supplementary Material reports an owner-level Poisson model with an offset for total parcels owned…"

- Added concentration analysis framing (Lorenz/Gini) to provide an owner-level distribution perspective, in Methods 3.8:
  > "To capture the administrative tractability of governance interventions, we quantify how SFHA-exposed parcels are distributed across owners within each organizational form… construct Lorenz curves and compute Gini coefficients…"


## Referee 2 — Minor comments

### Minor 1) Fixed effects and spatial autocorrelation

**Reviewer critique:**
> *Line 50 "Spatial autocorrelation motivates the use of fixed effects": this is inaccurate…*

**Changes made**
- Replaced/clarified the fixed-effect rationale and explicitly stated fixed effects do not eliminate residual spatial dependence; added clustered inference and residual spatial diagnostics. Added to Methods 3.6–3.7:
  > "Neighborhood fixed effects are included to account for neighborhood-level mean differences…"
  > "Fixed effects are not assumed to remove all spatial dependence in the residuals…"
  > "we also assess residual spatial structure by computing Moran's I on primary-model residuals…"


### Minor 2) Conflation of single-family residential parcels with single-family rentals; SFR abbreviation

**Reviewer critique:**
> *The authors occasionally conflate single family residential parcels with single family rentals…*

**Changes made**
- Revised the framing to consistently define the analytic sample as single-family residential parcels (SFR parcels) and added caveats that legal-form categories can contain both owner-occupied and rental properties. Example additions include:
  > "Using the complete set of single-family residential parcels…" (Introduction)
  > "…key caveats, including the fact that any category may contain both owner-occupants and landlords." (Methods 3.3)


### Minor 3) Dense sentences; unpack the "remote owner / information friction" logic

**Reviewer critique:**
> *There are some dense sentences throughout the background section… e.g., lines 64-65…*

**Changes made**
- Rewrote proximity/absentee discussion into a more explicit "proximity is a secondary correlate" framing in Literature Review 2.2.2:
  > "Because proximity can correlate with both owner type and management strategy, it is best treated as a secondary correlate that may moderate or mediate observed ownership–exposure associations rather than as a complete substitute for organizational measures."

- Operationalized proximity explicitly as a robustness check in Methods 3.5:
  > "As a secondary robustness control, we construct an owner–parcel distance proxy… included as log(1 + distance)…"


### Minor 4) Clarify policy relevance if rules apply uniformly

**Reviewer critique:**
> *I don't follow the argument that you need to know who owns floodplain properties for "disclosure, mitigation, insurance take-up, and recovery policy"…*

**Changes made**
- Reframed policy implications explicitly as implementation/delivery issues rather than owner-type-specific rule differences in Discussion 5.3:
  > "The principal implication of these findings is not that floodplain standards, disclosure requirements, or insurance recommendations should vary by owner type."
  > "From an administrative perspective, two places with identical SFHA parcel counts can pose very different governance challenges if one has exposure dispersed among many single-parcel households and the other has exposure concentrated in a smaller number of portfolios."
