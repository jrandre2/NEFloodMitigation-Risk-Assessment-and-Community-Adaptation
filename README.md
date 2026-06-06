# Nebraska Flood Resilience Project

**Disaster Risk Awareness and Housing Resilience Planning for HUD-MID Areas in Nebraska**

A research repository containing geospatial analysis scripts, FEMA NFIP claims data, and supporting
documentation for flood risk assessment in Nebraska's Most Impacted and Distressed (HUD-MID) areas.
Project period: January 2024 – December 2025. Led by researchers at the University of Nebraska-Lincoln.

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

## Repository Contents

```
NEFloodMitigation/
├── Data/
│   ├── NE_FEMA_Claims.csv              # Nebraska NFIP flood insurance claims (~6,000 records)
│   ├── ACS_Tract_Data_Dictionary.json  # Variable definitions for ACS tract-level data
│   ├── ACS_Variables_Defined.json      # ACS variable reference
│   ├── FEMA_Claims_Data_Dictionary.JSON # FEMA NFIP field-level data dictionary
│   ├── FEMA_Claims_Data_Themes.svg     # Visual overview of FEMA claims data themes
│   ├── Geospatial-Scripts/             # Python analysis and modeling scripts
│   │   ├── ACS_FLDZONE_by_Area         # ACS data by flood zone (arcpy)
│   │   ├── ACS_Spatial_Regression_Models.py
│   │   ├── Bootstrap_Parameter_Testing_Pipeline.py
│   │   ├── Bootstrap_NFIP_Probalistic_Metrics
│   │   ├── Bootstrap_OmniScript        # Multi-scenario bootstrap pipeline (Dodge Co.)
│   │   ├── BootstrapMetricsBorderPerturbations
│   │   ├── BootstrapMetricsScript
│   │   ├── CenPy_ACS_to_Shapefile.py
│   │   ├── DEM_to_Points.py
│   │   ├── Multithread_LiDAR_to_GeoTIFF
│   │   ├── NFIPPolicyDescriptivesBootstrap.py
│   │   ├── NFIP_Bootstrap
│   │   ├── NFIP_Bootstrap_Parameter_Sampling
│   │   ├── Owner_distance.py
│   │   ├── Points_to_Tracts.py
│   │   └── DouglasOwners/              # Douglas County property owner classification
│   │       ├── DefiningOwners.py
│   │       ├── Owner-Distance_Single-Family.py
│   │       └── ResidentialOwnerNameClassification.py
│   └── Reports/
│       ├── Damage_by_structure_type.pdf  # Flood damage analysis by structure type (3 pp.)
│       ├── Report_Template.JSON          # Detailed report template with section guidance
│       └── ReportTemplate.JSON           # Compact report metadata template
├── TimeLine.svg                          # Project timeline graphic
├── LICENSE
└── README.md
```

### Script Notes

Scripts in `Data/Geospatial-Scripts/` without a `.py` extension are Python scripts committed
without an extension (likely from direct GitHub uploads). Several depend on **ArcPy** (ArcGIS Pro)
for geospatial operations; others use open-source libraries (geopandas, gdal, cenpy).

---

## Data

| File | Description | Records |
|------|-------------|---------|
| `Data/NE_FEMA_Claims.csv` | Nebraska NFIP flood insurance claims with building/contents damage, policy, and geospatial fields | ~6,000 |

The FEMA NFIP claims data is sourced from FEMA's OpenFEMA dataset. Field definitions are in
`Data/FEMA_Claims_Data_Dictionary.JSON`.

---

## Key Analyses

- **Bootstrap spatial disaggregation** — Ensemble-based probabilistic assignment of NFIP claims
  to census tracts using stratified sampling and elevation/flood-zone filters
- **ACS spatial regression** — Linking American Community Survey socioeconomic variables to
  flood risk zones at the census-tract level
- **Property owner classification** — Identifying absentee/investor vs. owner-occupant properties
  in Douglas County using name-matching and distance metrics
- **LiDAR processing** — Multithreaded conversion of LiDAR point clouds to GeoTIFF rasters
  for elevation analysis

---

## Contact

- **Dr. Zhenghong Tang** (Principal Investigator): [ztang2@unl.edu](mailto:ztang2@unl.edu)
- **Dr. Yunwoo Nam** (Co-Investigator): [ynam2@unl.edu](mailto:ynam2@unl.edu)
- **Dr. Jesse Andrews** (Researcher): [jandrews30@unl.edu](mailto:jandrews30@unl.edu)
- **Dr. Jiyoung Lee** (Researcher): [jlee142@unl.edu](mailto:jlee142@unl.edu)

---

## Acknowledgments

Supported by the U.S. Department of Housing and Urban Development (HUD) as part of its efforts
to address disaster recovery and resilience in the Most Impacted and Distressed (MID) areas of
Nebraska. Partners include the Nebraska Department of Economic Development, Nebraska Department
of Natural Resources, FEMA, and local community organizations.
