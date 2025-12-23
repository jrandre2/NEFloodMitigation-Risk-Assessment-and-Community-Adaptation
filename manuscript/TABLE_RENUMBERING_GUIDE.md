# Table Renumbering Guide

*Created December 16, 2025 for manuscript revision*

This document tracks the table renumbering changes required to align the main manuscript with the revised tables.

---

## Table Number Mapping

| OLD Number | OLD Title | NEW Number | NEW Location |
|------------|-----------|------------|--------------|
| Table 1 | Variables and transformations | Table 1 | Main (unchanged) |
| Table 2 | Portfolio size distribution by entity type | Table 3 | Main |
| Table 3 | Ownership classifier configuration | Table S3 | **Supplementary** |
| Table 4 | Production classification coverage | Table S4 | **Supplementary** |
| Table 5 | SFR owner-type distribution | Table 4 | Main |
| Table 6 | Model specifications | Table 5 | Main |
| Table 7 | Counts and unadjusted SFHA rates | Table 6 | Main |
| Table 8 | Adjusted risk of SFHA location | Table 7 | Main |
| Table 9 | Tests of equality (within-form) | Table 8 | Main |
| — | (new) Organizational forms: functional definitions | Table 2 | Main (NEW) |
| — | (new) Residual spatial dependence diagnostic | Table 9 | Main (NEW) |

---

## Required Cross-Reference Updates in Manuscript

### Section 3.1 (Study Area)
- "Table 1" → **No change** (stays Table 1)

### Section 3.2 (Classifying Owner Type & Scale)
- "Tables 3 and 4" → **"Tables S3 and S4 in the Supplementary Material"**
- "Table 5" → **"Table 4"**
- "Table 2" → **"Table 3"**

### Section 3.3 (Variables)
- "Table 1" → **No change** (stays Table 1)

### Section 3.5 (Model Specifications)
- "Table 6" → **"Table 5"**

### Section 4.1 (Results - Unadjusted SFHA Rates)
- "Table 7" → **"Table 6"**

### Section 4.3 (Results - Primary Model Estimates)
- "Table 8" → **"Table 7"**

### Section 4.4 (Results - Within-form Contrasts)
- "Table 9" → **"Table 8"**

### Section 5.1 (Discussion - Interpreting Ownership Patterns)
- "(Table 8)" → **"(Table 7)"**

---

## Tables to Remove from Main Manuscript

The following tables should be removed from the main text and replaced with references to the Supplementary Material:

1. **Table 3. Ownership classifier configuration and training** → Now Table S3
2. **Table 4. Production classification coverage and high-confidence shares** → Now Table S4

---

## New Tables to Add

1. **Table 2. Organizational forms: functional definitions and interpretive role**
   - Insert after Table 1, before the former Table 2 (now Table 3)
   - Content: Owner type categories with identification examples, interpretive roles, and key caveats

2. **Table 9. Residual spatial dependence diagnostic for the primary model (M2)**
   - Insert after Table 8 (within-form tests)
   - Content: Moran's I on residuals, k-nearest neighbors, N

---

## Summary of Section 5/6 Changes (P0.1)

**Discussion Section 5.1:**
- Change "(Table 8)" to "(Table 7)" — this refers to adjusted risk ratios

**Conclusion Section 6:**
- No explicit table references found requiring update

---

## Status (Updated December 16, 2025)

**Completed in floodplain_doc.txt:**
- [x] Update Table 2 reference in Section 3.2 → Table 3
- [x] Update Tables 3 and 4 reference in Section 3.2 → Tables S3 and S4
- [x] Update Table 5 reference in Section 3.2 → Table 4
- [x] Update Table 6 reference in Section 3.5 → Table 5
- [x] Update Table 7 reference in Section 4.1 → Table 6
- [x] Update Table 8 reference in Section 4.3 → Table 7
- [x] Update Table 9 reference in Section 4.4 → Table 8
- [x] Update Table 8 reference in Section 5.1 → Table 7 (Discussion section)
- [x] Update table headers to new numbering

**Still needed in .docx file:**
- [ ] Make same changes in Who_Owns_the_Floodplain_20250909.docx
- [ ] Insert new Table 2 (Organizational forms)
- [ ] Insert new Table 9 (Moran's I diagnostic)
- [ ] Move old Tables 3, 4 content to Supplementary
