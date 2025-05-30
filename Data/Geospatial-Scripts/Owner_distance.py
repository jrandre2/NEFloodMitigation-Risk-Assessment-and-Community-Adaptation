#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Owner-Distance / Flood-Risk — Spatial + Full Stats (2025-05-30, no-impute)
=======================================================================
*   Descriptives, ANOVA, non-parametric tests, Cliff’s δ.
*   GLM / OLS suites for absenteeism, SFHA, and observed inundation.
*   Spatial diagnostics: Moran’s I, sparse GMM SAR & SEM (16-NN).


All results are written to
`<project>/diagnostics/owner_distance_spatial_full_<timestamp>.csv`.
"""

from __future__ import annotations

# ─── imports ─────────────────────────────────────────────────────────────
import os, csv, time, traceback, warnings
from datetime import datetime
from typing import List, Any, Dict

import arcpy
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.anova import anova_lm

# Spatial stack (optional) ----------------------------------------------
try:
    import libpysal as ps
    from spreg import GM_Lag, GM_Error
    from esda.moran import Moran
    HAS_SPATIAL = True
except ImportError:
    HAS_SPATIAL = False

# ─── configuration ─────────────────────────────────────────────────────
PROJECT_DIR = r"C:\Mac\Home\Documents\ArcGIS\Projects\OwnerDistanceProject"
GDB         = os.path.join(PROJECT_DIR, "OwnerDistanceProject.gdb")
PARCELS_FC  = os.path.join(GDB, "Parcels")
FIRM_FC     = os.path.join(GDB, "FIRM")
INUND_POLY  = os.path.join(GDB, "InundationPolygon")

# --- core fields --------------------------------------------------------
PID = "Parcel_ID"
DIST = "OwnerDist_km"
ZONE_FIRM = "FLD_ZONE"
LU_CODE = "Property_P"
DIM_CODE = "Zoning"
PLOT_SIZE = "SHAPE_Area"
INUND_F = "ActualInundated"
STRUCT_VAL = "Total_Asse"
YR_BUILT = "BuildingYe"
PAR_LON, PAR_LAT = "ParcelLon", "ParcelLat"

CALC_AGE = "CalculatedBuildingAge"

SFHA_CODES = {"A", "AE", "AO", "AH", "AR", "A99", "V", "VE"}
NEIGHBORS_K = 16

# --- land-use mapping (Agriculture, Exempt & Other → Industrial/Mixed) -
LU_MAP: Dict[int, str] = {
    1: "Residential",
    2: "Industrial/Mixed Use",
    3: "Commercial",
    4: "Industrial/Mixed Use",
    6: "Industrial/Mixed Use",
    9: "Industrial/Mixed Use",
}
DIM_MAP = {1: "Acreage", 2: "RuralLarge", 3: "RuralSmall", 4: "UrbanRegular", 5: "UrbanSmall"}

# ─── helpers ------------------------------------------------------------
log = lambda m: (arcpy.AddMessage(m), print(m))

ts = lambda: datetime.now().strftime("%Y%m%d_%H%M%S")

def write_csv(path: str, header: List[str], rows: List[List[Any]]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows([header, *rows])
    log(f"✓ Wrote {os.path.basename(path)}")

# Odds-ratio safe exp ----------------------------------------------------

def safe_exp(v: float):
    if pd.isna(v):
        return np.nan
    return np.exp(np.clip(v, -20, 20))

# Cliff’s δ --------------------------------------------------------------

def cliffs_d(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) == 0 or len(y) == 0:
        return np.nan
    m, n = len(x), len(y)
    gt = np.sum(x[:, None] > y)
    lt = np.sum(x[:, None] < y)
    return (gt - lt) / (m * n)

# ─── spatial helpers ----------------------------------------------------
if HAS_SPATIAL:
    def build_weights(lonlat: np.ndarray):
        W = ps.weights.KNN(np.radians(lonlat), k=NEIGHBORS_K)
        W.transform = "R"
        if W.n_components > 1:
            log(f"⚠ spatial weights have {W.n_components} disconnected components")
        return W

    def spatial_models(df: pd.DataFrame, W, outcome: str, out: List[List[Any]]):
        y = df[outcome].astype(float).values.reshape(-1, 1)
        reg_cols = ["log_BuildYear", "log_Total_Asse"]
        if "log_PlotSize" in df.columns:
            reg_cols.append("log_PlotSize")
        Xraw = df[reg_cols].dropna().copy()
        if Xraw.empty:  # nothing to model
            out.append([f"Spatial_{outcome}", "Status", None, "Error", "No predictors"])
            return
        keep_idx = Xraw.index
        y = y[keep_idx]
        X = (Xraw - Xraw.mean()) / Xraw.std().replace(0, 1)
        try:
            sar = GM_Lag(y, X.values, w=W, name_y=outcome)
            out.append([f"Spatial_SAR_{outcome}", "rho", None, "coef", float(sar.rho)])
            out.append([f"Spatial_SAR_{outcome}", "rho", None, "p", float(getattr(sar, 'pr2', np.nan))])
        except Exception as e:
            out.append([f"Spatial_SAR_{outcome}", "Status", None, "Error", str(e)])
        try:
            sem = GM_Error(y, X.values, w=W, name_y=outcome)
            out.append([f"Spatial_SEM_{outcome}", "lambda", None, "coef", float(sem.lam)])
            out.append([f"Spatial_SEM_{outcome}", "lambda", None, "p", float(getattr(sem, 'pr_lambda', np.nan))])
        except Exception as e:
            out.append([f"Spatial_SEM_{outcome}", "Status", None, "Error", str(e)])
else:
    def build_weights(*_):
        log("Spatial libraries unavailable – spatial models skipped.")
    def spatial_models(*_):
        pass

# ─── stats helpers (descriptives, pairwise) -----------------------------

def add_descriptives(df, group_col, val_col, label, rows):
    numeric = pd.to_numeric(df[val_col], errors='coerce')
    groups = [g.dropna() for _, g in numeric.groupby(df[group_col]) if not g.empty]
    if len(groups) > 1:
        h, p = stats.kruskal(*groups)
        rows += [[label, None, None, "Kruskal_H", h], [label, None, None, "Kruskal_p", p]]
    for gname, subset in df.groupby(group_col):
        s = pd.to_numeric(subset[val_col], errors='coerce').dropna()
        rows.append([label, str(gname), None, "mean", s.mean() if not s.empty else np.nan])
        rows.append([label, str(gname), None, "median", s.median() if not s.empty else np.nan])


def pairwise_mwu_holm(df, group_col, val_col, label, rows):
    groups = df[group_col].dropna().unique()
    combos, pvals = [], []
    for i, g1 in enumerate(groups):
        for g2 in groups[i+1:]:
            x = df.loc[df[group_col]==g1, val_col].dropna(); y = df.loc[df[group_col]==g2, val_col].dropna()
            if x.empty or y.empty: continue
            stat, p = stats.mannwhitneyu(x, y)
            combos.append((g1, g2)); pvals.append(p)
    if pvals:
        rej, padj, _, _ = multipletests(pvals, method='holm')
        for (g1, g2), pa, r in zip(combos, padj, rej):
            rows += [[f"Pairwise_{label}", g1, g2, "p_adj", pa], [f"Pairwise_{label}", g1, g2, "reject_H0", r]]

# regression -------------------------------------------------------------

def run_and_log_regression(df, formula, fam, label, rows):
    df_m = df.dropna(subset=[c for c in df.columns if c in formula])
    if len(df_m) < 20:
        log(f"Skipping {label}: N<20"); return
    model = (smf.glm(formula, data=df_m, family=fam).fit() if fam else smf.ols(formula, df_m).fit())
    ci = model.conf_int()
    if fam:
        pseudo_r2 = 1 - model.llf/model.llnull if model.llnull else np.nan
        rows.append([label, "ModelFit", None, "Pseudo_R2", pseudo_r2])
    else:
        rows.append([label, "ModelFit", None, "R2", model.rsquared])
    for term in model.params.index:
        coef, p = model.params[term], model.pvalues[term]
        lo, hi = ci.loc[term]
        base = [label, term, None]
        if fam and isinstance(fam.link, sm.families.links.logit):
            rows += [base+["OR", safe_exp(coef)], base+["OR_CI_low", safe_exp(lo)], base+["OR_CI_high", safe_exp(hi)]]
        else:
            rows += [base+["Coef", coef], base+["CI_low", lo], base+["CI_high", hi]]
        rows.append(base+["p", p])

# ─── main ----------------------------------------------------------------

def main():
    t0 = time.time(); arcpy.env.overwriteOutput = True
    if not (arcpy.Exists(PARCELS_FC) and arcpy.Exists(FIRM_FC)):
        log("✖ Required layers missing"); return

    diag_dir = os.path.join(PROJECT_DIR, "diagnostics"); os.makedirs(diag_dir, exist_ok=True)

    fields = [PID, ZONE_FIRM, DIST, LU_CODE, DIM_CODE, INUND_F, STRUCT_VAL, YR_BUILT,
              PLOT_SIZE, PAR_LON, PAR_LAT]
    avail = [f.name for f in arcpy.ListFields(PARCELS_FC)]
    fields = [f for f in fields if f in avail]
    arr = arcpy.da.TableToNumPyArray(PARCELS_FC, fields, skip_nulls=False, null_value=np.nan)
    df = pd.DataFrame(arr)

    # Derived vars ------------------------------------------------------
    df[DIST] = pd.to_numeric(df[DIST], errors='coerce')
    df['SFHA'] = df[ZONE_FIRM].isin(SFHA_CODES).astype(int)
    df['SFHA_status'] = np.where(df['SFHA']==1, 'InSFHA', 'NotInSFHA')
    df['LandUse'] = df[LU_CODE].apply(lambda v: LU_MAP.get(int(v), 'Industrial/Mixed Use') if pd.notna(v) else 'Industrial/Mixed Use')
    df['LotClass'] = df[DIM_CODE].apply(lambda v: DIM_MAP.get(int(v), 'Unknown_LC') if pd.notna(v) else 'Unknown_LC')
    df['Absentee'] = (df[DIST] > 0).astype(int)
    df['log_dist'] = np.log1p(df[DIST].fillna(0))

    # Basic logs of valuation/year -------------------------------------
    if STRUCT_VAL in df.columns:
        df['log_Total_Asse'] = np.log1p(pd.to_numeric(df[STRUCT_VAL], errors='coerce'))
    if YR_BUILT in df.columns:
        df['log_BuildYear'] = np.log1p(pd.to_numeric(df[YR_BUILT], errors='coerce'))

    if PLOT_SIZE in df.columns:
        df['log_PlotSize'] = np.log1p(pd.to_numeric(df[PLOT_SIZE], errors='coerce'))

    # Results container -------------------------------------------------
    rows: List[List[Any]] = []

    add_descriptives(df, 'LandUse', DIST, 'LandUse_vs_Dist', rows)
    pairwise_mwu_holm(df, 'LandUse', DIST, 'LandUse_Dist', rows)

    # Regression: absentee ~ SFHA + LandUse ----------------------------
    df['LandUse_cat'] = pd.Categorical(df['LandUse'], categories=list(set(LU_MAP.values())))
    run_and_log_regression(df, 'Absentee ~ SFHA + C(LandUse_cat)', sm.families.Binomial(), 'Logit_Absentee', rows)

    # Spatial suite ----------------------------------------------------
    if HAS_SPATIAL and all(c in df.columns for c in [PAR_LON, PAR_LAT]):
        coords = df[[PAR_LON, PAR_LAT]].apply(pd.to_numeric, errors='coerce').dropna()
        W = build_weights(coords.values)
        s_df = df.loc[coords.index].copy()
        spatial_models(s_df, W, 'SFHA', rows)
        if INUND_F in s_df.columns and s_df[INUND_F].nunique() > 1:
            spatial_models(s_df, W, INUND_F, rows)
        try:
            mi = Moran(s_df['SFHA'].values, W)
            rows += [["Moran_I", "SFHA", None, "I", mi.I], ["Moran_I", "SFHA", None, "p", mi.p_sim]]
        except Exception:
            log("Moran's I failed")
    else:
        log("Skipping spatial models – prerequisites missing")

    out_csv = os.path.join(diag_dir, f"owner_distance_spatial_full_{ts()}.csv")
    write_csv(out_csv, ["Metric", "Sub1", "Sub2", "Stat", "Value"], rows)
    log(f"✓ Finished in {(time.time()-t0)/60:.1f} min → {out_csv}")

# ─── entry point --------------------------------------------------------
if __name__ == '__main__':
    if not arcpy.Exists(PARCELS_FC):
        log(f"✖ Parcels not found at {PARCELS_FC}")
    elif not arcpy.Exists(FIRM_FC):
        log(f"✖ FIRM not found at {FIRM_FC}")
    else:
        log("✓ Starting analysis (no-impute, updated LU mapping)…")
        main()
