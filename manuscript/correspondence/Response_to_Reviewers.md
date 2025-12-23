# Response to Reviewers

**Manuscript:** EPB-2025-0878
**Title:** "Who Owns the Floodplain? Organizational Form, Portfolio Scale, and Regulatory Flood Exposure in Douglas County, Nebraska"
**Journal:** Environment and Planning B: Urban Analytics and City Science
**Date:** December 16, 2025

---

Dear Dr. Wolf and Reviewers,

Thank you for the thoughtful and constructive reviews of our manuscript. We have carefully considered each comment and made substantial revisions to address the concerns raised. Below we provide a point-by-point response to each comment, with references to specific sections where changes were made.

---

## Response to Referee 1

### Comment 1: Strengthen Argument for Ownership Patterns and Flood Risk

**R1.1a. Clarify Relevance of Five Organizational Forms**

> *The authors need to clarify why the "five mutually exclusive organizational forms" are relevant...*

**Response:** We have substantially revised the Introduction (Section 1) to articulate a governance-forward framing that explains why ownership structure matters for flood governance implementation. We now explicitly frame ownership as the "implementation interface" through which floodplain management operates, rather than making claims about market mechanisms. We added **Table 2** (new) providing functional definitions of each organizational form, their identification from owner-of-record strings, interpretive roles, and key caveats. This table directly addresses the request for clearer justification of the typology.

**Location:** Introduction paragraphs 1-3; new Table 2 (Section 3.3)

---

**R1.1b. Comparison with Existing Classification Efforts (Who Owns America)**

> *Does this work complement or provide confusing competing definitions with efforts like "Who Owns America"?*

**Response:** We now explicitly position our approach as complementary to beneficial-ownership consolidation methods. Section 2.4 discusses the "Who Owns America" methodology (An et al., 2024) and explains that we intentionally retain the owner-of-record legal shell because it is (a) the unit appearing in administrative land records and (b) consequential for governance and liability even when beneficial ownership is layered. We acknowledge that beneficial-owner consolidation can be layered as an extension. This framing is consistent throughout the revised manuscript.

**Location:** Section 2.4; Section 5.4 (Limitations)

---

**R1.1c. Unclear Title Issue**

> *The authors also should address the question of unclear title and how this fits into the problem.*

**Response:** We address this in the revised Introduction and in Section 2.4, noting that owner-of-record entities are not the same as ultimate beneficial owners and that LLCs may be controlled by individuals, trusts, or corporations. We treat the legal shell as meaningful for governance purposes while acknowledging this limitation explicitly.

**Location:** Introduction paragraph 6; Table 2 caveats column; Section 5.4

---

**R1.1d-e. Need for Clarity on Form and Scale; Systematic Presentation**

> *Clarity about both organizational form and portfolio scale, and how those lead to hypotheses are necessary...*

**Response:** The revised Introduction now systematically presents: (1) the governance motivation, (2) the two observable dimensions (organizational form and local portfolio footprint), and (3) three linked research questions with two pre-specified hypotheses (H1, H2) and one exploratory research question (RQ3). The hypotheses are grounded in specific literature and clearly stated before the methods section.

**Location:** Introduction (end of Section 1); Section 2.5 (Synthesis and Hypotheses)

---

### Comment 2: Single County Particularities

**R1.2a. Portfolio Scale Measurement Limitations**

> *The measure of portfolio scale at only the county level is problematic...*

**Response:** We acknowledge this limitation explicitly in Section 5.4 (Limitations). We now use the term "local portfolio footprint" (replacing "portfolio scale") throughout to emphasize that this is a within-county measure that does not observe holdings outside Douglas County or across multiple legal shells. We note that beneficial-ownership linkage methods could address this in future work.

**Location:** Section 3.3 (renamed "local portfolio footprint"); Section 5.4 (Limitations)

---

**R1.2b. County Description Needed**

> *Describe the county, the amount of land in floodplain and the amount in each of the different categories...*

**Response:** Section 3.1 has been substantially expanded to provide geographic and hydrological context for the study area. The revised text now describes Douglas County's location (eastern Nebraska, anchoring the Omaha metro region), population (≈601,000 residents), and land area (326 mi²). We describe the Papillion Creek watershed (≈402 mi²) that organizes much of the county's urban drainage, note its history of damaging floods and structural flood-control interventions, and cite recent watershed documentation (Strauch & Hoefer, 2025). The section also provides key sample statistics: 184,333 SFR parcels, with ≈1.1% falling within mapped SFHAs under our majority-area overlay rule. Table 6 now includes SFHA count and share columns showing how SFHA parcels are distributed across ownership groups, and Supplementary Figure S2 provides a spatial visualization. We note that analysis is restricted to SFR parcels; commercial/industrial properties are noted as out of scope in Section 5.4.

**Location:** Section 3.1 (expanded study area description); revised Table 6 (with SFHA count/share columns); Section 5.4

---

### Comment 3: Regulatory Floodplain Imperfections and Risk Rating 2.0

> *No mention of Risk Rating 2.0, which should influence behavior post and pre 2022...*

**Response:** We have added explicit acknowledgment of FEMA's Risk Rating 2.0 in Section 5.4 (Limitations): "In particular, FEMA's Risk Rating 2.0, implemented in 2021–2022, fundamentally restructured flood insurance pricing to reflect property-specific risk factors; the 2022 snapshot analyzed here cannot distinguish ownership patterns that predate versus postdate this regime change."

**Location:** Section 5.4 (Limitations, third paragraph)

---

**R1.3a-b. Temporal Strategies and Acquisition Models**

> *Institutionally different actors may have different time strategies... hypothesis might be better explored with acquisition models...*

**Response:** We acknowledge this limitation and reframe our contribution as documenting conditional associations in a cross-sectional snapshot, not as testing causal mechanisms. Section 5.4 now explicitly notes that "transaction-based designs linking deeds and sale dates to buyer legal form would allow tests of selection into SFHAs." We position our governance-forward approach as complementary to acquisition-based designs rather than as a substitute.

**Location:** Section 5.2 (Discussion); Section 5.4 (Limitations)

---

### Comment 4: Classification vs. Analytical Methods Balance

**R1.4 (intro). De-prioritize Classification, Justify Analytical Methods**

> *This is a paper that is based on hypotheses. Therefore, the classification algorithms should be de-prioritized and included in replicable appendices, and the analytical methods justified in greater detail.*

**Response:** We have restructured the manuscript to de-prioritize the classification workflow in the main text and move detailed documentation to the Supplementary Material. The main-text classification section (now Section 3.4) has been condensed to a single paragraph that briefly describes the approach and directs readers to the appendix for details. The Supplementary Material now contains: (a) Table S3 documenting classifier configuration and training, (b) Table S4 reporting production classification coverage and high-confidence shares, (c) Figure S1 showing the confusion matrix, and (d) a full Model Card with performance metrics, per-class precision/recall/F1, and ablation comparisons against a rule-based baseline. This structure keeps replicability information accessible while foregrounding the hypothesis tests and analytical methods in the main text.

Regarding analytical methods justification: we have substantially expanded Sections 3.6–3.7 to justify the modified Poisson GLM approach, explain the role of neighborhood fixed effects, document residual spatial diagnostics (Moran's I), and describe the eigenvector spatial filtering robustness check. These additions respond directly to the concern that analytical methods required greater justification.

**Location:** Section 3.4 (condensed); Supplementary Material (Tables S3–S4, Figure S1, Model Card); Sections 3.6–3.7 (expanded analytical methods)

---

**R1.4a. Spatial Modeling Questions**

> *Why not a spatial Poisson model? Does the neighborhood fixed effects model get analyzed for spatial clustering? What is the neighborhood scale here?*

**Response:**

We have substantially expanded the spatial analysis to directly address this concern. In addition to the previously reported diagnostics (residual Moran's I, Conley standard errors), we now implement **eigenvector spatial filtering (ESF)** as an explicit spatial Poisson robustness check (Griffith, 2003). Key additions:

1. **Residual Moran's I diagnostic**: We report Moran's I = 0.392 on primary-model residuals (Table 9, Section 4.6), documenting significant remaining spatial dependence after neighborhood fixed effects.

2. **Eigenvector spatial filtering**: We extract eigenvectors from a k=8 nearest-neighbor spatial weights matrix and include them as controls in the Poisson GLM. This directly models spatial structure rather than relying solely on inference corrections.

3. **Sensitivity analysis**: We vary the number of spatial filters from 15 to 200 (Table S7). With 200 filters, residual Moran's I decreases from 0.39 to effectively zero (0.003)—a **99% reduction** in residual spatial autocorrelation.

4. **Coefficient robustness**: The LLC risk ratio attenuates modestly from 1.38 (FE only) to 1.24 (FE + 200 spatial filters) but **remains statistically significant (p < 0.001) across all specifications**. This 10% attenuation indicates that approximately one-tenth of the association reflects spatial confounding, while the core finding is robust to spatial structure.

5. **Neighborhood definition**: Neighborhoods are assessor appraisal neighborhoods (470 groups in the analytic sample), and we use clustered standard errors at this level.

This analysis demonstrates that the primary findings are not artifacts of unmodeled spatial dependence—even when residual Moran's I is driven to near zero, the LLC–SFHA association persists.

**Location:** Section 3.7 (methods); Section 4.6 (results); Section 4.7 (robustness); Table 9; Table S5; **new Table S7**

---

**R1.4b-d. Development Patterns Concern; Single Jurisdiction; Temporal Trends**

**Response:** We acknowledge these limitations throughout Section 5.4. The cross-sectional design and single-county scope are noted as motivations for future multi-city replication and panel designs. We interpret results as conditional associations rather than causal effects.

**Location:** Section 5.4 (Limitations)

---

### Comment 5: Paper Focus

> *The authors need to decide whether this is a hypothesis-driven work, the dynamics in one particular county, or a methods paper...*

**Response:** We have clarified that this is primarily a **governance-oriented empirical paper** that documents how regulatory flood exposure is organized across ownership structures, with a secondary methodological contribution (the replicable classification workflow). The revised framing emphasizes governance relevance throughout: ownership structure provides information about the implementation interface for floodplain management. We de-emphasize causal claims and mechanism testing in favor of documenting conditional associations that are decision-relevant for administration.

**Location:** Introduction; Section 5.3 (Implications); Section 6 (Conclusion)

---

## Response to Referee 2

### Comment 1: Risk Sorting Mechanism Clarification

> *The risk sorting mechanism proposed in the manuscript doesn't make sense as currently written...*

**Response:** We have substantially revised the framing to remove claims about "risk-return sorting" as a primary mechanism. The revised manuscript takes a governance-forward, observational approach: we document conditional associations without claiming that ownership structures cause parcels to be in floodplains or that flood exposure is "profitable." We acknowledge multiple pathways (acquisition strategies, financing constraints, operational specialization) without privileging any single mechanism. The Discussion (Section 5.2) now presents alternative interpretations as plausible possibilities rather than as confirmed mechanisms.

**Location:** Introduction (reframed); Section 5.2 (Interpreting patterns without conflating association with mechanism)

---

### Comment 2: Organizational Form Definitions Needed

> *The authors need to provide functional and legal definitions of the five organizational forms...*

**Response:** We added **Table 2** providing functional definitions, identification criteria, interpretive roles, and key caveats for each organizational form. The table explicitly notes that:
- Any category may contain both owner-occupants and landlords
- LLCs may be controlled by individuals, trusts, or corporations (beneficial owner not observed)
- The distinction between single-parcel and multi-parcel LLCs is analytically salient for liability-siloing patterns

We acknowledge that LLCs can be owned by corporations and trusts, which complicates direct comparisons; this is noted in the caveats and in Section 5.4.

**Location:** New Table 2; Section 3.3

---

**R2.2 (Key Questions). Specific Questions on Organizational Forms**

> *Key Questions:*
> - *Why is single-parcel liability siloing significant for LLCs but not for corporations?*
> - *Why is it important to isolate differences between trusts and corporations – why choose these five groups?*
> - *What do we learn from the fact that single-parcel LLCs are over-represented in terms of owning SFHA property?*

**Response:** We address each question in turn:

1. **Single-parcel liability siloing and LLCs vs. corporations:** We do not claim that liability siloing is significant for LLCs but not corporations. Rather, we hypothesize (H2) that single-parcel LLCs may reflect asset-level structuring because the LLC form is particularly amenable to single-asset liability compartmentalization at low cost (Travis, 2019). Corporations can also silo assets, but the LLC form has become the predominant vehicle for this strategy in residential real estate. The data show that single-parcel LLCs are indeed overrepresented in SFHAs (RR = 1.67), while single-parcel corporations are not (RR ≈ 1.0)—an empirical difference we document without claiming to explain causally.

2. **Why five groups (trusts vs. corporations):** Trusts and corporations are distinct legal structures that are legible in owner-of-record data and serve different functions. Trusts are fiduciary/estate-planning vehicles; corporations are business entities with different formation, governance, and liability characteristics. We distinguish them because (a) they are empirically distinguishable in assessor records, (b) they may have different relationships to flood exposure, and (c) collapsing them would obscure potentially meaningful variation. The results confirm this: trusts and corporations exhibit different SFHA prevalence patterns (Table 7). The five-category typology reflects the major legal forms identifiable in administrative data; we do not claim these are the only possible categories, but they provide a governance-relevant decomposition that is replicable across jurisdictions.

3. **What we learn from single-parcel LLC overrepresentation:** The finding that single-parcel LLCs are disproportionately located in SFHAs (adjusted RR = 1.67) suggests that SFHA exposure is not uniformly distributed across ownership structures. This pattern is governance-relevant because it identifies a subset of the ownership landscape—administratively legible as LLCs with single-asset footprints—where regulatory flood exposure is concentrated. We interpret this as a descriptive pattern warranting attention, not as evidence of a specific causal mechanism.

**Location:** Section 3.3; Table 2; Section 5.2

---

### Comment 3: Correlation vs. Causation

> *The authors need to be careful not to conflate correlation with causation...*

**Response:** We have thoroughly revised the manuscript to use cautious, observational language throughout. Key changes:
- Removed "isolate the effect of ownership" language
- Replaced with "estimate conditional associations" and "conditional prevalence"
- Section 5.2 is titled "Interpreting patterns without conflating association with mechanism"
- We explicitly state: "The results document conditional prevalence differences: they do not identify a causal effect of organizational form on floodplain siting. Floodplains are generated by hydrology and mapped through regulatory processes, and organizational form does not determine floodplain location."

**Location:** Throughout; especially Section 3.6, Section 4.6, Section 5.2

---

### Comment 4: Discussion Alignment with Findings

> *Much of the discussion about renters and disclosures seem to be several degrees removed from the actual content of the findings...*

**Response:** We have revised the Discussion to focus on governance implications that follow directly from the findings: (1) ownership structure provides information about implementation interfaces, and (2) concentration patterns affect administrative tractability. We removed or softened speculation about renter disclosure and tenant outcomes, instead framing these as directions for future research. Policy implications (Section 5.3) now focus on implementation questions (portfolio-aware administration) rather than rule differentiation.

**Location:** Section 5.3 (revised); Section 5.4 (future research)

---

### Comment 5: Within-Form Contrasts Statistical Question

> *Is there any accounting for the fact that multi-property owners are more likely to own SFHA homes purely based on the fact that they own multiple properties?*

**Response:** We added a new paragraph in Section 3.6 ("A note on estimands") that directly addresses this concern:

> "The parcel-level prevalence estimated here is the governance-relevant quantity because floodplain regulation, permitting, and program delivery are administered at the parcel. Multi-parcel owners are mechanically more likely to hold at least one SFHA parcel simply by owning more parcels, but the parcel-level model estimates the share of each group's holdings that fall within SFHAs—a quantity that is comparable across owners regardless of portfolio size. To complement this parcel-level view, the concentration analysis (Section 3.8) provides an owner-level perspective by documenting how SFHA holdings are distributed across owners within each organizational form."

We also note in Supplementary Material that an owner-level model (Table S6) could be developed to address this directly.

**Location:** Section 3.6 (new paragraph); Supplementary Material (Table S6 placeholder)

---

### Minor Comments

**M1. Spatial Autocorrelation and Fixed Effects**

> *Fixed effects do not fix the residual dependence from spatial autocorrelation...*

**Response:** Corrected. We now state that fixed effects absorb neighborhood-level mean differences but do not remove residual spatial dependence. We report residual Moran's I (0.392) and use clustered standard errors at the neighborhood level. We also report Conley spatial HAC SEs as a robustness check.

**Location:** Section 3.7; Table S5

---

**M2. SFR Abbreviation Confusion**

> *The authors occasionally conflate single family residential parcels with single family rentals. For instance, the abbreviation SFR first refers to all SF parcels, then SF rentals in lines 138-140 before going back to all SF parcels.*

**Response:** We have clarified that SFR refers to "single-family residential" parcels throughout and removed any conflation with "single-family rentals." The analysis does not directly observe tenure status.

**Location:** Throughout; Table 2 caveats

---

**M3. Dense Sentences in Background**

> *There are some dense sentences throughout the background section that could benefit from unpacking. For instance, lines 64-65: "information frictions can disadvantage remote owners with proximity shaping both acquisition and ongoing management."*

**Response:** We have revised dense sentences in the literature review and introduction to improve clarity.

**Location:** Sections 1-2

---

**M4. Policy Implications Clarification**

> *I don't follow the argument that you need to know who owns floodplain properties for disclosure, mitigation, insurance take-up, and recovery policy...*

**Response:** We have revised Section 5.3 to focus on implementation rather than rule differentiation. The key insight is that administrative tractability differs depending on whether exposure is concentrated (fewer larger portfolios) or dispersed (many small owners)—even when uniform standards apply. We explicitly state: "The principal implication of these findings is not that floodplain standards, disclosure requirements, or insurance recommendations should vary by owner type."

**Location:** Section 5.3

---

## Summary of Major Revisions

| Area | Key Changes |
|------|-------------|
| **Framing** | Governance-forward, observational; removed causal language |
| **Terminology** | "Local portfolio footprint" (not "scale"); "conditional associations" |
| **New Table 2** | Functional definitions of organizational forms |
| **New Table 9** | Residual spatial dependence diagnostic |
| **New Table S7** | Eigenvector spatial filtering sensitivity analysis (spatial Poisson robustness) |
| **Section 4.6** | Expanded spatial analysis with ESF results; Moran's I reduction from 0.39 to 0.003 |
| **New Section 4.7** | Robustness checks with Table S5 and S7 references |
| **Section 3.6** | Estimand clarity paragraph addressing mechanical probability |
| **Section 3.7** | Spatial inference justification; Conley SEs; ESF methods description |
| **Section 5.4** | Risk Rating 2.0 acknowledgment; non-SFR limitation |
| **Table 6** | Added SFHA count and share columns |
| **Abstract** | Updated with governance framing, correct statistics |

We believe these revisions address the substantive concerns raised by both reviewers while strengthening the manuscript's contribution to urban flood governance research.

Sincerely,

[Authors]
