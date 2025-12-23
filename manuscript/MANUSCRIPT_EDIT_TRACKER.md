# Manuscript Edit Tracker

**Authoritative Manuscript:** `/Users/jesseandrews/Library/CloudStorage/OneDrive-TexasTechUniversity/Who Owns the Floodplain - JA Revisons.docx`

**Extracted Version:** `manuscript/drafts/JA_Revisions_extracted.md`

**Date:** December 16, 2025

---

## IMPORTANT: Two Manuscripts Exist

| Version | Location | Status |
|---------|----------|--------|
| **NEW (Authoritative)** | `OneDrive/Who Owns the Floodplain - JA Revisons.docx` | Active revision |
| **Extracted copy** | `manuscript/drafts/JA_Revisions_extracted.md` | Editable markdown version |
| OLD (superseded) | `manuscript/drafts/floodplain_doc.txt` | Do not use |

---

## VERIFIED: NEW Manuscript Statistics Are Correct

The NEW manuscript already reports correct values matching Tables_Main_Revised.md:

| Group | Reported in Manuscript (line #) | Matches Table 7? |
|-------|--------------------------------|------------------|
| LLC-Single | RR = 1.670 [1.311–2.126], p < 0.001 (L164) | ✓ Yes |
| LLC-Multi | RR = 1.379 [1.104–1.722], p = 0.005 (L164) | ✓ Yes |
| Individual-Multi | RR = 1.209 [1.038–1.407], p = 0.015 (L166) | ✓ Yes |
| Trust-Multi | RR = 1.459 [1.043–2.041], p = 0.028 (L166) | ✓ Yes |
| Corporation-Multi | RR = 1.568 [0.960–2.561], p = 0.073 (L168) | ✓ Yes (marginal) |

**Note:** NEW manuscript reports RRs directly, not percentages like "67% higher."

---

## Completed Supporting Materials

### 1. Revised Abstract (P0.3)
- **File:** `manuscript/drafts/REVISED_ABSTRACT.md`
- **Status:** Ready for insertion into NEW manuscript (which has no abstract)
- Uses "local portfolio footprint" terminology
- Governance-forward framing

### 2. Table S5 - Robustness Checks (P0.2/P2.2)
- **File:** `manuscript/supplementary/Supplementary_Material_Revised.md`
- **Status:** Complete
- **Action needed:** NEW manuscript doesn't reference Table S5 yet

### 3. Revised Tables (Tables_Main_Revised.md)
- **File:** `manuscript/tables/Tables_Main_Revised.md`
- **Status:** Complete

---

## Pending Edits for NEW Manuscript

### P0 - Must Do:
- [x] Verify statistics - DONE (all correct)
- [x] **Insert abstract** - DONE (added to JA_Revisions_extracted.md)
- [x] **Add Table S5 reference** - DONE (new Section 4.7 "Robustness checks" added)

### P1 - Recommended:
- [ ] P1.1: Expand study area description + define neighborhoods (SKIPPED per user)
- [x] P1.2: Add Risk Rating 2.0 limitation sentence - DONE (Section 5.4)
- [x] P1.3: Add estimand clarity paragraph (parcel vs owner) - DONE (Section 3.6)
- [x] P1.4: Add spatial inference robustness check - DONE (Section 3.7)

### P2 - Optional:
- [x] P2.1: Add SFHA count/share columns to Table 6 - DONE (Tables_Main_Revised.md)
- [x] P2.3: Create owner-level rate model (Table S6) - DONE (Supplementary_Material_Revised.md)
  - Analysis script: `scripts/owner_level_sfha_model.py`
  - Results: LLC IRR=2.58, Corporation IRR=3.75, Trust IRR=1.64, Gov/Nonprofit IRR=2.87 (all p<0.001)
  - Reference added to Section 3.6 (estimand paragraph)
- [x] P2.4: Add non-SFR SFHA descriptive context - DONE (Section 5.4)

### P3 - Packaging:
- [x] P3.1: Write response-to-reviewers letter - DONE (manuscript/correspondence/Response_to_Reviewers.md)
- [ ] P3.2: Create tracked changes + clean copy versions - REQUIRES WORD (user to complete)
- [x] P3.3: Verify Supplementary completeness - DONE (all referenced tables present)

---

## Key Files Summary

| File | Purpose |
|------|---------|
| `JA_Revisions_extracted.md` | Full manuscript in markdown (editable) |
| `REVISED_ABSTRACT.md` | Abstract to insert |
| `Tables_Main_Revised.md` | Authoritative tables |
| `Supplementary_Material_Revised.md` | Supplementary with Table S5 |

---

## Workflow

1. Edits made to `JA_Revisions_extracted.md` (markdown)
2. You incorporate changes into the Word document
3. Supporting materials already in markdown format
