# Nebraska Flood Mitigation: Risk Assessment and Community Adaptation

**Disaster Risk Awareness and Housing Resilience Planning for HUD-MID Areas in Nebraska**

This repository supports a research project examining flood risk and community adaptation in
Nebraska's Most Impacted and Distressed (HUD-MID) areas — communities identified by the U.S.
Department of Housing and Urban Development (HUD) as severely affected by recent flood disasters.
The work combines analysis of Federal Emergency Management Agency (FEMA) National Flood Insurance
Program (NFIP) claims data with American Community Survey (ACS) socioeconomic indicators and
high-resolution geospatial data to answer: *who is most exposed to flood risk, what does the
damage look like by structure type, and how can local planning improve resilience?*

Project period: January 2024 – December 2025.
Led by researchers at the University of Nebraska-Lincoln.

---

## Key Output

**[Flood Damage Analysis by Structure Type (PDF)](Data/Reports/Damage_by_structure_type.pdf)**
— The primary report summarizing NFIP claim patterns, damage distributions by building type,
and implications for Nebraska's HUD-MID communities.

---

## Project Timeline

![Project Timeline](https://raw.githubusercontent.com/jrandre2/NEFloodMitigation-Risk-Assessment-and-Community-Adaptation/main/TimeLine.svg)

---

## Objectives

1. Conduct comprehensive flood risk assessments for Nebraska HUD-MID communities
2. Develop risk awareness and housing resiliency plans
3. Enhance community engagement and education on flood risks
4. Evaluate and improve integration of flood resilience in local and regional planning

---

## Repository Map

```
NEFloodMitigation/
├── Data/
│   ├── NE_FEMA_Claims.csv               # Nebraska NFIP claims (~6,000 records; see dictionary)
│   ├── ACS_Tract_Data_Dictionary.json   # Field definitions for ACS tract-level variables
│   ├── ACS_Variables_Defined.json       # ACS variable reference
│   ├── FEMA_Claims_Data_Dictionary.JSON # FEMA NFIP field-level data dictionary
│   ├── FEMA_Claims_Data_Themes.svg      # Visual overview of claims data themes
│   ├── Geospatial-Scripts/              # Python analysis and modeling scripts
│   │   ├── ACS_Spatial_Regression_Models.py
│   │   ├── Bootstrap_Parameter_Testing_Pipeline.py
│   │   ├── CenPy_ACS_to_Shapefile.py
│   │   ├── DEM_to_Points.py
│   │   ├── NFIPPolicyDescriptivesBootstrap.py
│   │   ├── Owner_distance.py
│   │   ├── Points_to_Tracts.py
│   │   ├── [+ 8 additional bootstrap/LiDAR scripts]
│   │   └── DouglasOwners/               # Douglas County property owner classification
│   └── Reports/
│       ├── Damage_by_structure_type.pdf # Primary output report (3 pp.)
│       └── Report_Template.JSON         # Report structure template
├── docs/
│   └── TECHNICAL_NOTES.md              # Script dependencies, workflows, environment setup
├── TimeLine.svg                         # Project timeline graphic
└── LICENSE
```

**Large/raw data** (shapefiles, GeoTIFFs, geodatabases) are excluded from version control
via `.gitignore` and must be obtained separately. The FEMA claims CSV (`NE_FEMA_Claims.csv`)
includes geospatial identifier fields (county FIPS, census tract, coordinates); see
`Data/FEMA_Claims_Data_Dictionary.JSON` for full field definitions.

---

## What the Analysis Does

**Bootstrap spatial disaggregation** — NFIP claims arrive with county-level geography. The
bootstrap pipeline probabilistically assigns claims to individual census tracts using building
footprints, flood zone codes, and elevation stratification, producing tract-level risk estimates
with uncertainty bounds.

**ACS spatial regression** — Spatial lag and spatial error models link ACS socioeconomic
variables (income, poverty, disability, race/ethnicity, broadband access) to flood claim rates
at the census-tract level, identifying which communities face compounded social and physical
flood vulnerability.

**Property owner classification** — Scripts classify Douglas County residential properties
as owner-occupied or absentee/investor-owned using name-matching and owner-address distance
metrics, informing questions about who bears flood risk and who makes mitigation decisions.

**LiDAR and DEM processing** — Multithreaded scripts convert LiDAR point clouds and Digital
Elevation Model (DEM) data to raster and point formats for elevation analysis.

For dependency requirements, script-by-script workflow details, and environment setup, see
**[docs/TECHNICAL_NOTES.md](docs/TECHNICAL_NOTES.md)**.

---

## Data

| File | Description |
|---|---|
| `Data/NE_FEMA_Claims.csv` | ~6,000 Nebraska NFIP flood insurance claims (building/contents damage, policy info, geospatial identifiers) — sourced from FEMA OpenFEMA |
| `Data/FEMA_Claims_Data_Dictionary.JSON` | Field-level definitions for the NFIP claims file |
| `Data/ACS_Tract_Data_Dictionary.json` | Definitions for ACS tract-level socioeconomic variables |

---

## Team

- **Dr. Zhenghong Tang** (Principal Investigator): [ztang2@unl.edu](mailto:ztang2@unl.edu)
- **Dr. Yunwoo Nam** (Co-Investigator): [ynam2@unl.edu](mailto:ynam2@unl.edu)
- **Dr. Jesse Andrews** (Researcher): [jandrews30@unl.edu](mailto:jandrews30@unl.edu)
- **Dr. Jiyoung Lee** (Researcher): [jlee142@unl.edu](mailto:jlee142@unl.edu)

---

## Acknowledgments

Supported by the U.S. Department of Housing and Urban Development (HUD) as part of its
disaster recovery and resilience program for the Most Impacted and Distressed (MID) areas of
Nebraska. Partners include the Nebraska Department of Economic Development, Nebraska Department
of Natural Resources, FEMA, and local community organizations.
