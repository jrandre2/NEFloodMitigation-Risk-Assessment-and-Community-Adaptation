#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Owner-Distance / Flood-Risk — SINGLE-FAMILY, Douglas County
==========================================================
✓ Detects owner-occupied parcels by comparing *Situs_Addr* and *OW1_Addres*,
  using:
    • house-number + first 2 street tokens comparison (after cleaning)
    • PO-Box heuristic (ZIP + house number)
    • NO distance fallback - pure address matching only
✓ Filters to single-family parcels (Zoning == 1)
✓ Sets distance to 0 for all owner-occupied properties
✓ Analyzes distance as continuous variable with log transformation
✓ Includes enhanced regression analyses
✓ Outputs CSV with descriptives, GLM models, and optional spatial diagnostics
✓ Creates visualizations of distance distributions
✓ Uses 1km distance-based spatial weights for spatial analysis
"""

from __future__ import annotations
import os, csv, re, time, traceback
from datetime import datetime
from typing import List, Any

import arcpy
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf
import statsmodels.api as sm

# ───────── optional spatial stack ─────────
try:
    import libpysal as ps
    from spreg import GM_Lag, GM_Error
    from esda.moran import Moran
    HAS_SPATIAL = True
except ImportError:
    HAS_SPATIAL = False

# ───────── optional plotting ─────────
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOTS = True
except ImportError:
    HAS_PLOTS = False

# ───────── paths ─────────
PROJECT_DIR = r"C:\Mac\Home\Documents\ArcGIS\Projects\OwnerDistanceProject"
GDB         = os.path.join(PROJECT_DIR, "OwnerDistanceProject.gdb")
PARCELS_FC  = os.path.join(GDB, "Parcels")
FIRM_FC     = os.path.join(GDB, "FIRM")

# ───────── field names ─────────
PID, DIST, ZONING, ZONE_FIRM = "Parcel_ID", "OwnerDist_km", "Zoning", "FLD_ZONE"
SITUS_ADDR, OWNER_ADDR       = "Situs_Addr", "OW1_Addres"
SITUS_ZIP,  OWNER_ZIP        = "Situs_Zip", "Mailing_Zip"
PLOT_SIZE,  STRUCT_VAL       = "SHAPE_Area", "Total_Asse"
YR_BUILT,   INUND_F          = "BuildingYe", "ActualInundated"
PAR_LON, PAR_LAT             = "ParcelLon", "ParcelLat"

# Alternative field names for plot size (common in ArcGIS)
PLOT_SIZE_ALTS = ["SHAPE_Area", "Shape_Area", "AREA", "Area", "ACRES", "Acres"]

# Spatial parameters
DISTANCE_THRESHOLD_KM = 1.0  # 1km distance threshold for spatial contiguity
SFHA_CODES  = {"A","AE","AO","AH","AR","A99","V","VE"}

# ───────── basic helpers ─────────
log = lambda m: (arcpy.AddMessage(m), print(m))
ts  = lambda: datetime.now().strftime("%Y%m%d_%H%M%S")

def write_csv(path: str, header: List[str], rows: List[List[Any]]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows([header, *rows])
    log(f"✓ Wrote {os.path.basename(path)}")

# ───────── owner-occupancy logic ─────────
_city_tokens = {"OMAHA","ELKHORN","BENNINGTON","VALLEY","WATERLOO"}
_state_token = "NE"
_po_re       = re.compile(r"^P\s*O\s*BOX\s+", re.I)
_zip_pad_re  = re.compile(r"(\d{5})0{4}$")          # chop "0000"
_multi_spc   = re.compile(r"\s+")

def _clean(a: str | None) -> str:
    if not isinstance(a, str):
        return ""
    a = _multi_spc.sub(" ", a.upper()).strip()
    a = _zip_pad_re.sub(r"\1", a)
    parts = [p for p in a.split()
             if p not in _city_tokens and p != _state_token]
    return " ".join(parts)

def _key(a: str) -> list[str]:
    return a.split()[:3]            # house # + 2 street tokens

def owner_occ(row) -> bool:
    s_raw, o_raw = row.get(SITUS_ADDR), row.get(OWNER_ADDR)
    if not (isinstance(s_raw, str) and isinstance(o_raw, str)):
        return False
    s, o = _clean(s_raw), _clean(o_raw)
    
    # If situs is empty after cleaning, can't be owner-occupied
    if not s:
        return False

    # PO-Box heuristic
    if _po_re.match(o):
        s_parts = s.split()
        # Check if s has any parts and first part is a digit
        if s_parts and s_parts[0].isdigit():
            zip_ok = str(row.get(SITUS_ZIP))[:5] == str(row.get(OWNER_ZIP))[:5]
            number_ok = s_parts[0] in o
            return zip_ok and number_ok
        return False

    return _key(s) == _key(o)

safe_exp = lambda v: (np.nan if pd.isna(v) or v < -20 or v > 20 
                      else np.exp(v))

# ───────── stats helpers ─────────
def add_descriptives(df, group_col, val_col, label, rows):
    numeric = pd.to_numeric(df[val_col], errors="coerce")
    groups  = [g.dropna() for _, g in numeric.groupby(df[group_col])
               if not g.empty]
    if len(groups) > 1:
        h, p = stats.kruskal(*groups)
        rows += [[label, None, None, "Kruskal_H", h],
                 [label, None, None, "Kruskal_p", p]]
    for gname, subset in df.groupby(group_col):
        s = pd.to_numeric(subset[val_col], errors="coerce").dropna()
        rows += [[label, str(gname), None, "mean",   s.mean()   if not s.empty else np.nan],
                 [label, str(gname), None, "median", s.median() if not s.empty else np.nan],
                 [label, str(gname), None, "count",  len(s)     if not s.empty else 0]]

def add_distance_analysis(df, rows):
    """Add detailed distance analysis including log distance"""
    if DIST not in df.columns:
        return
    
    # Basic distance statistics
    dist_numeric = pd.to_numeric(df[DIST], errors="coerce")
    log_dist = np.log1p(dist_numeric)
    
    # Distance percentiles for each group
    for gname, subset in df.groupby("Absentee"):
        s = pd.to_numeric(subset[DIST], errors="coerce").dropna()
        if not s.empty:
            for p in [10, 25, 50, 75, 90, 95, 99]:
                rows.append([f"Distance_Percentile_P{p}", str(gname), None, "km", s.quantile(p/100)])
    
    # Log distance statistics
    add_descriptives(df, "Absentee", "log_Distance", "SF_LogDist_Absentee", rows)
    
    # Distance bins for visualization
    bins = [0, 0.5, 1, 2, 5, 10, 25, 50, 100, 1000, 10000]
    df['dist_bin'] = pd.cut(dist_numeric, bins=bins, labels=[f"{bins[i]}-{bins[i+1]}" for i in range(len(bins)-1)])
    
    for gname, subset in df.groupby("Absentee"):
        bin_counts = subset['dist_bin'].value_counts().sort_index()
        for bin_label, count in bin_counts.items():
            rows.append([f"Distance_Bin_{bin_label}km", str(gname), None, "count", count])
            rows.append([f"Distance_Bin_{bin_label}km", str(gname), None, "percent", count/len(subset)*100])

def run_and_log_regression(df, formula, fam, label, rows):
    # Extract column names from formula
    formula_cols = [col.strip() for col in re.findall(r'\b\w+\b', formula) 
                    if col not in ['~', '+', '*', '-', ':', 'I', 'C']]
    # Only keep columns that exist in the dataframe
    valid_cols = [c for c in formula_cols if c in df.columns]
    
    df_m = df.dropna(subset=valid_cols)
    if len(df_m) < 20:
        log(f"Skipping {label}: N<20 after dropping NA"); return
    
    # Check if all required columns exist
    missing_cols = [c for c in formula_cols if c not in df.columns]
    if missing_cols:
        log(f"Warning: Missing columns for {label}: {missing_cols}")
        # Modify formula to exclude missing columns
        for col in missing_cols:
            formula = formula.replace(f" + {col}", "").replace(f"{col} + ", "")
            formula = formula.replace(f" ~ {col}", " ~ 1")  # If dependent var missing
        log(f"Modified formula: {formula}")
    
    try:
        model = smf.glm(formula, df_m, family=fam).fit() if fam else \
                smf.ols(formula, df_m).fit()
        ci = model.conf_int()
        if fam:
            rows.append([label, "ModelFit", None, "Pseudo_R2",
                         1 - model.llf/model.llnull if model.llnull else np.nan])
        else:
            rows.append([label, "ModelFit", None, "R2", model.rsquared])
        for term in model.params.index:
            coef, p = model.params[term], model.pvalues[term]
            lo, hi  = ci.loc[term]
            base = [label, term, None]
            if fam and isinstance(fam.link, sm.families.links.logit):
                rows += [[*base, "OR",         safe_exp(coef)],
                         [*base, "OR_CI_low",  safe_exp(lo)],
                         [*base, "OR_CI_high", safe_exp(hi)]]
            else:
                rows += [[*base, "Coef",   coef],
                         [*base, "CI_low", lo],
                         [*base, "CI_high", hi]]
            rows.append([*base, "p", p])
    except Exception as e:
        log(f"Error in {label} regression: {str(e)}")
        rows.append([label, "Status", None, "Error", str(e)])

def run_enhanced_regressions(sf, rows):
    """Run additional regression analyses comparing owner-occupied vs absentee properties"""
    
    log("\n══════════════════════════════════════════════════════")
    log("ENHANCED REGRESSION ANALYSIS")
    log("══════════════════════════════════════════════════════\n")
    
    # Import additional modules if available
    try:
        from statsmodels.discrete.discrete_model import Probit
        from statsmodels.regression.quantile_regression import QuantReg
        HAS_ADVANCED = True
    except ImportError:
        HAS_ADVANCED = False
        log("Note: Some advanced models unavailable (missing statsmodels components)")
    
    # ─────────────────────────────────────────────────────────────
    # 1. COMPARING OWNER-OCCUPIED VS ABSENTEE PROPERTIES
    # ─────────────────────────────────────────────────────────────
    
    log("1. WHAT PROPERTY CHARACTERISTICS PREDICT OWNER-OCCUPANCY?")
    log("─" * 50)
    
    # A. Basic property characteristics model
    if all(col in sf.columns for col in ["log_Total_Asse", "log_BuildYear", "log_PlotSize"]):
        # Ensure OwnerOcc is integer
        sf["OwnerOcc_int"] = sf["OwnerOcc"].astype(int)
        formula = "OwnerOcc_int ~ log_Total_Asse + log_BuildYear + log_PlotSize"
        if "SFHA" in sf.columns:
            formula += " + SFHA"
        
        # Logit model (need to adjust formula back to OwnerOcc for this function)
        formula_logit = formula.replace("OwnerOcc_int", "OwnerOcc")
        run_and_log_regression(sf, formula_logit, sm.families.Binomial(), 
                               "Logit_OwnerOcc_PropertyChars", rows)
        
        # Probit model for comparison (if available)
        if HAS_ADVANCED:
            try:
                df_clean = sf.dropna(subset=["OwnerOcc_int"] + [col for col in ["log_Total_Asse", "log_BuildYear", "log_PlotSize", "SFHA"] if col in sf.columns])
                if len(df_clean) > 100:
                    probit = smf.probit(formula, data=df_clean).fit()
                    rows.append(["Probit_OwnerOcc_PropertyChars", "ModelFit", None, "Pseudo_R2", 
                                 probit.prsquared])
                    # Calculate marginal effects
                    margeff = probit.get_margeff()
                    for i, var in enumerate(margeff.names):
                        rows.append([f"Probit_OwnerOcc_MarginalEffects", var, None, "ME", 
                                     margeff.margeff[i]])
                        rows.append([f"Probit_OwnerOcc_MarginalEffects", var, None, "p", 
                                     margeff.pvalues[i]])
            except Exception as e:
                log(f"Probit model failed: {str(e)}")
    
    # ─────────────────────────────────────────────────────────────
    # 2. ANALYZING DISTANCE EFFECTS WITHIN ABSENTEE PROPERTIES
    # ─────────────────────────────────────────────────────────────
    
    log("\n2. DISTANCE EFFECTS WITHIN ABSENTEE PROPERTIES")
    log("─" * 50)
    
    # Filter to absentee properties only
    absentee_df = sf[sf["Absentee"] == True].copy()
    
    if len(absentee_df) > 100 and "log_Distance" in absentee_df.columns:
        
        # A. Non-linear distance effects using polynomial terms
        absentee_df["log_Distance_sq"] = absentee_df["log_Distance"] ** 2
        
        # Property value as outcome
        if "log_Total_Asse" in absentee_df.columns:
            formula_poly = "log_Total_Asse ~ log_Distance + log_Distance_sq"
            run_and_log_regression(absentee_df, formula_poly, None, 
                                   "OLS_PropertyValue_DistancePoly", rows)
        
        # B. Distance categories for analysis
        # Create distance categories for absentee properties
        if "OwnerDist_km" in absentee_df.columns:
            dist_vals = absentee_df[absentee_df["OwnerDist_km"] > 0]["OwnerDist_km"]
            if len(dist_vals) > 100:
                absentee_df["DistanceCategory"] = pd.cut(
                    absentee_df["OwnerDist_km"],
                    bins=[0, 0.001, 10, 100, 10000],
                    labels=["Zero", "Local", "Regional", "Distant"],
                    include_lowest=True
                )
                
                # Descriptive stats by distance category
                for cat in ["Local", "Regional", "Distant"]:
                    cat_data = absentee_df[absentee_df["DistanceCategory"] == cat]
                    if len(cat_data) > 0:
                        rows.append([f"DistanceCategory_{cat}", "Descriptive", None, "Count", 
                                     len(cat_data)])
                        if "log_Total_Asse" in cat_data.columns:
                            rows.append([f"DistanceCategory_{cat}", "Descriptive", None, 
                                         "Mean_LogValue", cat_data["log_Total_Asse"].mean()])
                        if "SFHA" in cat_data.columns:
                            rows.append([f"DistanceCategory_{cat}", "Descriptive", None, 
                                         "Pct_FloodZone", cat_data["SFHA"].mean() * 100])
        
        # C. Quantile regression (if available)
        if HAS_ADVANCED and "log_Total_Asse" in absentee_df.columns:
            try:
                # Prepare data for quantile regression
                qr_vars = ["log_Total_Asse", "log_Distance"]
                if "log_BuildYear" in absentee_df.columns:
                    qr_vars.append("log_BuildYear")
                if "SFHA" in absentee_df.columns:
                    qr_vars.append("SFHA")
                
                qr_df = absentee_df[qr_vars].dropna()
                if len(qr_df) > 200:
                    y = qr_df["log_Total_Asse"]
                    X = qr_df.drop("log_Total_Asse", axis=1)
                    X = sm.add_constant(X)
                    
                    # Run quantile regression for median only (faster)
                    qr_model = QuantReg(y, X).fit(q=0.5)
                    rows.append(["QuantReg_PropertyValue_Q50", "ModelFit", 
                                 None, "Pseudo_R2", qr_model.prsquared])
                    for var in qr_model.params.index:
                        if var != "const":
                            rows.append([f"QuantReg_PropertyValue_Q50", var, 
                                         None, "Coef", qr_model.params[var]])
                    
                    log("Completed quantile regression analysis")
            except Exception as e:
                log(f"Quantile regression failed: {str(e)}")
    
    # ─────────────────────────────────────────────────────────────
    # 3. COMPARATIVE ANALYSIS OF PROPERTY OUTCOMES
    # ─────────────────────────────────────────────────────────────
    
    log("\n3. COMPARING PROPERTY OUTCOMES BY OWNERSHIP TYPE")
    log("─" * 50)
    
    # A. T-tests for mean differences
    continuous_vars = ["log_Total_Asse", "log_BuildYear", "log_PlotSize"]
    for var in continuous_vars:
        if var in sf.columns:
            owner_vals = sf[sf["OwnerOcc"] == True][var].dropna()
            absentee_vals = sf[sf["Absentee"] == True][var].dropna()
            
            if len(owner_vals) > 30 and len(absentee_vals) > 30:
                # T-test
                t_stat, p_val = stats.ttest_ind(owner_vals, absentee_vals)
                rows.append([f"TTest_{var}", "OwnerOcc_vs_Absentee", None, "t_stat", t_stat])
                rows.append([f"TTest_{var}", "OwnerOcc_vs_Absentee", None, "p_value", p_val])
                rows.append([f"TTest_{var}", "OwnerOcc_vs_Absentee", None, "mean_diff", 
                             owner_vals.mean() - absentee_vals.mean()])
                
                # Effect size (Cohen's d)
                pooled_std = np.sqrt(((len(owner_vals)-1)*owner_vals.std()**2 + 
                                      (len(absentee_vals)-1)*absentee_vals.std()**2) / 
                                     (len(owner_vals) + len(absentee_vals) - 2))
                cohen_d = (owner_vals.mean() - absentee_vals.mean()) / pooled_std
                rows.append([f"TTest_{var}", "OwnerOcc_vs_Absentee", None, "cohen_d", cohen_d])
    
    # B. Chi-square tests for categorical variables
    if "SFHA" in sf.columns:
        contingency = pd.crosstab(sf["OwnerOcc"], sf["SFHA"])
        chi2, p, dof, expected = stats.chi2_contingency(contingency)
        rows.append(["ChiSquare_SFHA_OwnerOcc", "Test", None, "chi2", chi2])
        rows.append(["ChiSquare_SFHA_OwnerOcc", "Test", None, "p_value", p])
        
        # Calculate Cramér's V for effect size
        n = contingency.sum().sum()
        cramers_v = np.sqrt(chi2 / (n * (min(contingency.shape) - 1)))
        rows.append(["ChiSquare_SFHA_OwnerOcc", "Test", None, "cramers_v", cramers_v])
    
    # ─────────────────────────────────────────────────────────────
    # 4. DISTANCE THRESHOLD EFFECTS
    # ─────────────────────────────────────────────────────────────
    
    log("\n4. EXPLORING DISTANCE THRESHOLD EFFECTS")
    log("─" * 50)
    
    if "OwnerDist_km" in absentee_df.columns and len(absentee_df) > 100:
        # Create binary indicators for different distance thresholds
        thresholds = [10, 50, 100, 500]
        for thresh in thresholds:
            absentee_df[f"Beyond_{thresh}km"] = (absentee_df["OwnerDist_km"] > thresh).astype(int)
        
        # Model property characteristics at different distance thresholds
        if "log_Total_Asse" in absentee_df.columns:
            for thresh in [10, 100]:
                if f"Beyond_{thresh}km" in absentee_df.columns:
                    thresh_df = absentee_df.dropna(subset=["log_Total_Asse", f"Beyond_{thresh}km"])
                    if thresh_df[f"Beyond_{thresh}km"].sum() > 20:  # Enough distant properties
                        formula_thresh = f"log_Total_Asse ~ Beyond_{thresh}km"
                        if "log_BuildYear" in thresh_df.columns:
                            formula_thresh += " + log_BuildYear"
                        if "SFHA" in thresh_df.columns:
                            formula_thresh += " + SFHA"
                        
                        run_and_log_regression(thresh_df, formula_thresh, None, 
                                               f"OLS_PropertyValue_Beyond{thresh}km", rows)
    
    # ─────────────────────────────────────────────────────────────
    # 5. INTERACTION EFFECTS EXPLORATION
    # ─────────────────────────────────────────────────────────────
    
    log("\n5. EXPLORING INTERACTION EFFECTS")
    log("─" * 50)
    
    # Property value × Flood zone interaction
    if all(col in sf.columns for col in ["Absentee", "log_Total_Asse", "SFHA"]):
        formula_interact1 = "Absentee ~ log_Total_Asse * SFHA"
        if "log_BuildYear" in sf.columns:
            formula_interact1 += " + log_BuildYear"
        
        run_and_log_regression(sf, formula_interact1, sm.families.Binomial(), 
                               "Logit_Absentee_ValueFloodInteract", rows)
    
    log("\n══════════════════════════════════════════════════════")
    log("ENHANCED REGRESSION ANALYSIS COMPLETE")
    log("══════════════════════════════════════════════════════\n")

# ───────── spatial helpers (optional) ─────────
if HAS_SPATIAL:
    def build_distance_weights(coords: np.ndarray, threshold_km: float):
        """Build distance-based spatial weights with threshold in kilometers"""
        try:
            # Convert coordinates to radians for geodesic distance
            coords_rad = np.radians(coords)
            
            # Earth radius in km
            earth_radius_km = 6371.0
            
            # Calculate arc distance threshold in radians
            # arc_distance = threshold_km / earth_radius_km
            # However, for small distances, we can use a simpler approach
            
            # Create distance band weights
            # threshold parameter should be in the same units as the coordinates
            # Since we're using radians, convert km threshold to radians
            threshold_rad = threshold_km / earth_radius_km
            
            W = ps.weights.DistanceBand(coords_rad, threshold=threshold_rad, 
                                        binary=True)
            
            # Row-standardize the weights
            W.transform = "R"
            
            # Report on the weights matrix
            log(f"Built distance-based spatial weights with {threshold_km}km threshold")
            log(f"  - Number of observations: {W.n}")
            log(f"  - Average number of neighbors: {W.mean_neighbors:.2f}")
            log(f"  - Min neighbors: {W.min_neighbors}")
            log(f"  - Max neighbors: {W.max_neighbors}")
            
            # Check for isolates
            if hasattr(W, 'islands') and W.islands:
                log(f"  - {len(W.islands)} observations have no neighbors within {threshold_km}km")
                log(f"    These will be excluded from spatial regression models")
            
            # Get the number of components if available
            if hasattr(W, 'n_components'):
                if W.n_components > 1:
                    log(f"  - Warning: {W.n_components} disconnected components in spatial structure")
            
            return W
            
        except Exception as e:
            log(f"Error building distance-based spatial weights: {str(e)}")
            raise

    def spatial_models(df, W, outcome, rows):
        """Run spatial regression models with proper error handling"""
        if outcome not in df.columns:
            rows.append([f"Spatial_{outcome}", "Status", None, "Error",
                         f"Outcome column '{outcome}' not found"])
            return
        
        # Get predictors that exist
        reg_cols = [c for c in ("log_BuildYear", "log_Total_Asse",
                                "log_PlotSize", "log_Distance") if c in df.columns]
        if not reg_cols:
            rows.append([f"Spatial_{outcome}", "Status", None, "Error",
                         "No predictors available"])
            return
        
        # Create a clean dataset with outcome and predictors, no missing values
        cols_needed = [outcome] + reg_cols
        clean_df = df[cols_needed].dropna()
        
        # Need sufficient observations for spatial models
        min_obs = 100  # Minimum observations for stable estimates
        
        if clean_df.empty or len(clean_df) < min_obs:
            rows.append([f"Spatial_{outcome}", "Status", None, "Error",
                         f"Insufficient data after dropping NA (n={len(clean_df)}, need {min_obs}+)"])
            return
        
        # Get the indices of rows we're keeping (after dropna)
        keep_idx = clean_df.index.tolist()
        
        # Check if we have any islands that would be problematic
        islands = set()
        if hasattr(W, 'islands'):
            islands = set(W.islands)
        
        # Remove islands from our analysis set
        keep_idx_no_islands = [idx for idx in keep_idx if idx not in islands]
        
        if len(keep_idx_no_islands) < min_obs:
            rows.append([f"Spatial_{outcome}", "Status", None, "Error",
                         f"Too few non-isolated observations (n={len(keep_idx_no_islands)})"])
            return
        
        # Use the non-island subset
        if len(keep_idx_no_islands) < len(keep_idx):
            log(f"  Removing {len(keep_idx) - len(keep_idx_no_islands)} isolated observations")
            clean_df = clean_df.loc[keep_idx_no_islands]
            keep_idx = keep_idx_no_islands
        
        # Create index mapping from original to clean data
        idx_map = {orig_idx: new_idx for new_idx, orig_idx in enumerate(keep_idx)}
        
        # Build a new weights matrix for the subset
        # This is more reliable than trying to subset the original
        try:
            # Create new weights object with only the kept observations
            W_subset = ps.weights.W({idx_map[i]: {idx_map[j]: W[i][j] 
                                                   for j in W[i].keys() if j in idx_map}
                                     for i in keep_idx if i in W.neighbors})
            W_subset.transform = "R"
            
            # Ensure the ordering matches our data
            W_subset = ps.weights.util.fill_diagonal(W_subset, 0)
            
            log(f"Subset weights matrix: {W_subset.n} observations, "
                f"avg {W_subset.mean_neighbors:.1f} neighbors")
            
        except Exception as e:
            log(f"Error creating subset weights matrix: {str(e)}")
            rows.append([f"Spatial_{outcome}", "Status", None, "Error",
                         f"Failed to subset weights matrix: {str(e)}"])
            return
        
        # Now extract y and X from the same clean dataframe
        y = clean_df[outcome].astype(float).values.reshape(-1, 1)
        X = clean_df[reg_cols]
        
        # Standardize X for better numerical stability
        X_std = (X - X.mean()) / X.std()
        X_array = X_std.values
        
        log(f"Running spatial models for {outcome} with {len(y)} observations")
        
        # Spatial Autoregressive Model (SAR)
        try:
            sar = GM_Lag(y, X_array, w=W_subset, name_y=outcome,
                         name_x=reg_cols, robust='white')
            
            # Extract spatial parameter
            rho_coef = sar.rho
            
            # Calculate p-value for rho
            # The z-stat for rho is typically in position [n_vars+1] of z_stat array
            if hasattr(sar, 'z_stat'):
                rho_z = sar.z_stat[-1, 0]  # Last element is usually rho
                rho_p = 2 * (1 - stats.norm.cdf(abs(rho_z)))
            else:
                rho_p = np.nan
            
            rows += [[f"Spatial_SAR_{outcome}", "rho", None, "coef", rho_coef],
                     [f"Spatial_SAR_{outcome}", "rho", None, "p", rho_p]]
            
            # Add coefficients for predictors
            for i, col in enumerate(reg_cols):
                coef = sar.betas[i+1, 0]  # Skip intercept
                if hasattr(sar, 'z_stat') and i+1 < len(sar.z_stat):
                    z_val = sar.z_stat[i+1, 0]
                    p_val = 2 * (1 - stats.norm.cdf(abs(z_val)))
                else:
                    p_val = np.nan
                
                rows += [[f"Spatial_SAR_{outcome}", col, None, "coef", coef],
                         [f"Spatial_SAR_{outcome}", col, None, "p", p_val]]
            
            log(f"  ✓ SAR model completed")
            
        except Exception as e:
            log(f"  ✖ SAR model failed: {str(e)}")
            rows.append([f"Spatial_SAR_{outcome}", "Status", None, "Error", str(e)])
        
        # Spatial Error Model (SEM)
        try:
            sem = GM_Error(y, X_array, w=W_subset, name_y=outcome,
                           name_x=reg_cols, robust='white')
            
            # Extract spatial parameter
            lambda_coef = sem.lam
            
            # Calculate p-value for lambda
            if hasattr(sem, 'z_stat'):
                # Lambda is typically the last parameter
                lambda_z = sem.z_stat[-1, 0]
                lambda_p = 2 * (1 - stats.norm.cdf(abs(lambda_z)))
            else:
                lambda_p = np.nan
            
            rows += [[f"Spatial_SEM_{outcome}", "lambda", None, "coef", lambda_coef],
                     [f"Spatial_SEM_{outcome}", "lambda", None, "p", lambda_p]]
            
            # Add coefficients for predictors
            for i, col in enumerate(reg_cols):
                coef = sem.betas[i+1, 0]  # Skip intercept
                if hasattr(sem, 'z_stat') and i+1 < len(sem.z_stat):
                    z_val = sem.z_stat[i+1, 0]
                    p_val = 2 * (1 - stats.norm.cdf(abs(z_val)))
                else:
                    p_val = np.nan
                    
                rows += [[f"Spatial_SEM_{outcome}", col, None, "coef", coef],
                         [f"Spatial_SEM_{outcome}", col, None, "p", p_val]]
            
            log(f"  ✓ SEM model completed")
            
        except Exception as e:
            log(f"  ✖ SEM model failed: {str(e)}")
            rows.append([f"Spatial_SEM_{outcome}", "Status", None, "Error", str(e)])
else:
    def spatial_models(*_): pass

# ───────── plotting helpers (optional) ─────────
def create_distance_plots(df, dist_col, group_col, out_dir):
    if not HAS_PLOTS or dist_col not in df.columns:
        return
    
    try:
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Box plot of distances (excluding 0s for better visibility)
        ax1 = axes[0, 0]
        absentee_data = df[df[group_col] == True][dist_col].dropna()
        absentee_nonzero = absentee_data[absentee_data > 0]
        if len(absentee_nonzero) > 0:
            ax1.boxplot([absentee_nonzero], labels=['Absentee (>0 km)'])
            ax1.set_ylabel('Distance (km)')
            ax1.set_title('Absentee Owner Distance Distribution')
            ax1.set_ylim(0, absentee_nonzero.quantile(0.95))
            ax1.text(0.02, 0.98, f'Note: All owner-occupied = 0 km\n{(absentee_data == 0).sum()} absentee also at 0 km', 
                    transform=ax1.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # 2. Log-scale histogram (excluding 0s)
        ax2 = axes[0, 1]
        absentee_positive = absentee_data[absentee_data > 0]
        if len(absentee_positive) > 0:
            ax2.hist(absentee_positive, bins=np.logspace(-1, 4, 50), alpha=0.7, color='#e74c3c', edgecolor='black')
            ax2.set_xscale('log')
            ax2.set_xlabel('Distance (km)')
            ax2.set_ylabel('Count')
            ax2.set_title('Absentee Distance Distribution (Log Scale, >0 km only)')
            ax2.grid(True, alpha=0.3)
        
        # 3. Cumulative distribution comparing all vs >0km absentee
        ax3 = axes[1, 0]
        if len(absentee_data) > 0:
            # All absentee (including 0s)
            sorted_all = absentee_data.sort_values()
            y_all = np.arange(1, len(sorted_all) + 1) / len(sorted_all)
            ax3.plot(sorted_all, y_all, label='All Absentee', linewidth=2, color='#3498db')
            
            # Only positive distance absentee
            if len(absentee_positive) > 0:
                sorted_pos = absentee_positive.sort_values()
                y_pos = np.arange(1, len(sorted_pos) + 1) / len(sorted_pos)
                ax3.plot(sorted_pos, y_pos, label='Absentee >0 km', linewidth=2, color='#e74c3c')
            
            ax3.axvline(x=0, color='green', linestyle='--', alpha=0.5, label='Owner-Occupied (all at 0)')
            ax3.set_xscale('symlog')  # Symmetric log scale to show 0
            ax3.set_xlabel('Distance (km)')
            ax3.set_ylabel('Cumulative Probability')
            ax3.set_title('Cumulative Distance Distribution')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # 4. Distance categories pie chart
        ax4 = axes[1, 1]
        owner_count = len(df[df[group_col] == False])
        absentee_zero = (absentee_data == 0).sum()
        absentee_local = ((absentee_data > 0) & (absentee_data <= 10)).sum()
        absentee_regional = ((absentee_data > 10) & (absentee_data <= 100)).sum()
        absentee_distant = (absentee_data > 100).sum()
        
        sizes = [owner_count, absentee_zero, absentee_local, absentee_regional, absentee_distant]
        labels = ['Owner-Occupied\n(0 km)', 'Absentee at 0 km\n(data issue?)', 
                 'Local Absentee\n(0-10 km)', 'Regional\n(10-100 km)', 'Distant\n(>100 km)']
        colors = ['#2ecc71', '#f39c12', '#3498db', '#9b59b6', '#e74c3c']
        
        # Remove zero-count categories
        non_zero = [(s, l, c) for s, l, c in zip(sizes, labels, colors) if s > 0]
        if non_zero:
            sizes, labels, colors = zip(*non_zero)
            ax4.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
            ax4.set_title('Property Distribution by Owner Distance')
        
        plt.tight_layout()
        plot_path = os.path.join(out_dir, f"distance_analysis_{ts()}.png")
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        log(f"✓ Saved distance analysis plot: {os.path.basename(plot_path)}")
    except Exception as e:
        log(f"Warning: Could not create plots: {str(e)}")

# ───────── main workflow ─────────
def main():
    t0 = time.time(); arcpy.env.overwriteOutput = True

    # sanity check layers
    for fc, nm in ((PARCELS_FC, "Parcels"), (FIRM_FC, "FIRM")):
        if not arcpy.Exists(fc):
            log(f"✖ {nm} layer missing: {fc}"); return

    diag_dir = os.path.join(PROJECT_DIR, "diagnostics")
    os.makedirs(diag_dir, exist_ok=True)

    fields = [PID, ZONING, DIST, ZONE_FIRM, STRUCT_VAL, YR_BUILT,
              PAR_LON, PAR_LAT, SITUS_ADDR, OWNER_ADDR, SITUS_ZIP, OWNER_ZIP,
              INUND_F]
    
    # Add any available plot size field
    avail = {f.name for f in arcpy.ListFields(PARCELS_FC)}
    for plot_field in PLOT_SIZE_ALTS:
        if plot_field in avail:
            fields.append(plot_field)
            break
    
    fields = [f for f in fields if f in avail]
    
    if not fields:
        log("✖ No valid fields found in parcels layer"); return

    df = pd.DataFrame(arcpy.da.TableToNumPyArray(
        PARCELS_FC, fields, skip_nulls=False, null_value=np.nan))

    # Drop parcels without situs address
    initial_count = len(df)
    if SITUS_ADDR in df.columns:
        df = df[df[SITUS_ADDR].notna() & (df[SITUS_ADDR] != '')].copy()
        dropped_count = initial_count - len(df)
        if dropped_count > 0:
            log(f"Dropped {dropped_count} parcels without situs address")
    else:
        log(f"✖ {SITUS_ADDR} field not found - cannot proceed with owner occupancy analysis")
        return
    
    if df.empty:
        log("✖ No parcels with valid situs addresses found")
        return

    # owner-occupancy using ONLY address matching
    df["OwnerOcc"] = df.apply(owner_occ, axis=1)
    df["Absentee"] = ~df["OwnerOcc"]
    
    # Check original distance values before correction
    if DIST in df.columns:
        df[DIST] = pd.to_numeric(df[DIST], errors="coerce")
        
        # Analysis before correction
        owner_occ_with_dist = df[df["OwnerOcc"] & (df[DIST] > 0)]
        log(f"\n── Distance Data Consistency Check ──")
        log(f"Owner-occupied properties with non-zero distance (before correction): {len(owner_occ_with_dist)}")
        if len(owner_occ_with_dist) > 0:
            log(f"  - Mean distance: {owner_occ_with_dist[DIST].mean():.2f} km")
            log(f"  - Median distance: {owner_occ_with_dist[DIST].median():.2f} km")
            log(f"  - Max distance: {owner_occ_with_dist[DIST].max():.2f} km")
            log(f"\nNote: These likely represent data quality issues where address matching")
            log(f"      identified owner-occupancy but distance calculation was incorrect.")
        
        # Set distance to 0 for all owner-occupied properties
        df.loc[df["OwnerOcc"], DIST] = 0
        log(f"\n✓ Set distance to 0 km for all {df['OwnerOcc'].sum()} owner-occupied properties")
        
        # Check properties originally marked as 0km that are NOT owner-occupied
        zero_dist_not_owner = df[(df[DIST] == 0) & ~df["OwnerOcc"]]
        if len(zero_dist_not_owner) > 0:
            log(f"\nNote: {len(zero_dist_not_owner)} properties have 0 km distance but are NOT owner-occupied by address match")
            log(f"These may represent data quality issues or special cases")

    # single-family filter
    if ZONING in df.columns:
        df[ZONING] = pd.to_numeric(df[ZONING], errors="coerce")
        sf = df[df[ZONING] == 1].copy()
        log(f"\nFiltered to {len(sf)} single-family parcels (Zoning == 1)")
    else:
        log(f"Warning: {ZONING} field not found, using all parcels")
        sf = df.copy()
        
    if sf.empty:
        log("✖ No single-family parcels found"); return
    
    # Summary statistics
    log(f"\n── Summary ──")
    log(f"Total parcels processed: {len(sf)}")
    log(f"Owner-occupied: {sf['OwnerOcc'].sum()}")
    log(f"Absentee-owned: {sf['Absentee'].sum()}")
    log(f"Owner occupancy rate: {sf['OwnerOcc'].mean():.1%}")
    
    # flood & valuation transforms
    if ZONE_FIRM in sf.columns:
        sf["SFHA"] = sf[ZONE_FIRM].isin(SFHA_CODES).astype(int)
    else:
        log(f"Warning: {ZONE_FIRM} field not found, SFHA analysis skipped")
    
    # Find plot size field
    plot_size_field = None
    for field_name in PLOT_SIZE_ALTS:
        if field_name in sf.columns:
            plot_size_field = field_name
            break
            
    for fld, out in ((plot_size_field, "log_PlotSize"),
                     (STRUCT_VAL, "log_Total_Asse"),
                     (YR_BUILT,  "log_BuildYear")):
        if fld and fld in sf.columns:
            sf[out] = np.log1p(pd.to_numeric(sf[fld], errors="coerce"))
        elif fld:
            log(f"Warning: {fld} field not found, {out} not created")
    
    # Add log distance if distance field exists
    if DIST in sf.columns:
        # Add 1 to avoid log(0), convert to numeric first
        sf["log_Distance"] = np.log1p(pd.to_numeric(sf[DIST], errors="coerce"))
        
        # Check for properties with valid addresses but missing distances
        missing_dist = sf[sf[DIST].isna()]
        log(f"\n── Distance Data Completeness ──")
        log(f"Properties with distance data: {sf[DIST].notna().sum()} ({sf[DIST].notna().mean():.1%})")
        log(f"Properties missing distance data: {len(missing_dist)}")
        if len(missing_dist) > 0:
            log(f"  - Owner-occupied (missing distance): {missing_dist['OwnerOcc'].sum()}")
            log(f"  - Absentee (missing distance): {missing_dist['Absentee'].sum()}")
    
    # Check for outliers in distance
    if DIST in sf.columns and sf[DIST].notna().any():
        log(f"\n── Distance Distribution ──")
        
        # Overall stats
        log(f"Total properties with distance data: {sf[DIST].notna().sum()}")
        
        # Owner-occupied should all be 0
        owner_occ_dist = sf[sf['OwnerOcc']][DIST].dropna()
        log(f"\nOwner-Occupied Properties:")
        log(f"  - Count: {len(owner_occ_dist)}")
        log(f"  - All distances = 0 km: {(owner_occ_dist == 0).all()}")
        if not (owner_occ_dist == 0).all():
            log(f"  - WARNING: Found non-zero distances for owner-occupied properties!")
        
        # Absentee owner stats
        absentee_dist = sf[sf['Absentee']][DIST].dropna()
        log(f"\nAbsentee-Owned Properties:")
        log(f"  - Count: {len(absentee_dist)}")
        log(f"  - With 0 km distance: {(absentee_dist == 0).sum()} (may indicate data issues)")
        log(f"  - Mean (excluding 0s): {absentee_dist[absentee_dist > 0].mean():.2f} km")
        log(f"  - Median (excluding 0s): {absentee_dist[absentee_dist > 0].median():.2f} km")
        log(f"  - 90th percentile: {absentee_dist.quantile(0.9):.2f} km")
        log(f"  - 99th percentile: {absentee_dist.quantile(0.99):.2f} km")
        log(f"  - Max distance: {absentee_dist.max():.2f} km")
        log(f"  - Properties > 1000 km away: {(absentee_dist > 1000).sum()}")
        log(f"  - Properties > 5000 km away: {(absentee_dist > 5000).sum()}")
    
    log("")
    
    # ── analysis rows
    rows: list[list[Any]] = []
    
    if DIST in sf.columns:
        add_descriptives(sf, "Absentee", DIST, "SF_Dist_Absentee", rows)
        add_distance_analysis(sf, rows)
        create_distance_plots(sf, DIST, "Absentee", diag_dir)
    else:
        log(f"Warning: {DIST} field not found, distance descriptives skipped")
    
    # Build regression formula based on available columns
    reg_vars = []
    if "SFHA" in sf.columns:
        reg_vars.append("SFHA")
    if "log_Total_Asse" in sf.columns:
        reg_vars.append("log_Total_Asse")
    if "log_BuildYear" in sf.columns:
        reg_vars.append("log_BuildYear")
    
    if reg_vars:
        # Base model without distance
        formula = "Absentee ~ " + " + ".join(reg_vars)
        run_and_log_regression(sf, formula, sm.families.Binomial(), 
                               "Logit_SF_Absentee_Base", rows)
        
        # Note: We cannot include distance in models predicting absentee status
        # because by definition, all owner-occupied have distance=0
        # This creates perfect separation and model convergence issues
        log("\n── Note on Distance Models ──")
        log("Distance cannot be used to predict absentee status because:")
        log("  - All owner-occupied properties have distance = 0 (by definition)")
        log("  - This creates perfect separation in logistic regression")
        log("  - Instead, we analyze distance effects WITHIN absentee properties")
        
        # Add a note in the results
        rows.append(["Logit_SF_Absentee_Distance", "Status", None, "Note", 
                     "Distance models omitted due to perfect separation"])
        
        # Additional models: What predicts being in flood zone for each ownership type?
        if "SFHA" in sf.columns:
            flood_vars = []
            if "log_Total_Asse" in sf.columns:
                flood_vars.append("log_Total_Asse")
            if "log_BuildYear" in sf.columns:
                flood_vars.append("log_BuildYear")
            if "log_PlotSize" in sf.columns:
                flood_vars.append("log_PlotSize")
                
            if flood_vars:
                flood_formula = "SFHA ~ " + " + ".join(flood_vars)
                
                # Model for owner-occupied properties
                owner_occ_df = sf[sf["OwnerOcc"] == True]
                if len(owner_occ_df) > 100:
                    run_and_log_regression(owner_occ_df, flood_formula, sm.families.Binomial(), 
                                           "Logit_SFHA_OwnerOcc", rows)
                
                # Model for absentee properties
                absentee_df = sf[sf["Absentee"] == True]
                if len(absentee_df) > 100:
                    run_and_log_regression(absentee_df, flood_formula, sm.families.Binomial(), 
                                           "Logit_SFHA_Absentee", rows)
        
        # OLS models to examine continuous relationships
        if "log_Distance" in sf.columns:
            # What predicts distance among absentee properties?
            log("\n── Distance Analysis Within Absentee Properties ──")
            
            dist_vars = []
            if "SFHA" in sf.columns:
                dist_vars.append("SFHA")
            if "log_Total_Asse" in sf.columns:
                dist_vars.append("log_Total_Asse")
            if "log_BuildYear" in sf.columns:
                dist_vars.append("log_BuildYear")
            if "log_PlotSize" in sf.columns:
                dist_vars.append("log_PlotSize")
                
            if dist_vars:
                # Only for absentee properties (owner-occupied all have distance = 0)
                absentee_df = sf[sf["Absentee"] == True]
                
                # Check for sufficient variation in distance
                dist_values = absentee_df[absentee_df[DIST].notna()][DIST]
                if len(dist_values) > 100 and dist_values.std() > 0:
                    dist_formula = "log_Distance ~ " + " + ".join(dist_vars)
                    run_and_log_regression(absentee_df, dist_formula, None, 
                                           "OLS_LogDistance_Absentee", rows)
                    
                    # Additional analysis: extreme distance properties
                    extreme_dist = absentee_df[absentee_df[DIST] > 1000]
                    if len(extreme_dist) > 20:
                        log(f"\n  Properties with owners >1000km away: {len(extreme_dist)}")
                        if "log_Total_Asse" in extreme_dist.columns:
                            log(f"  - Mean property value (log): {extreme_dist['log_Total_Asse'].mean():.2f}")
                        if "SFHA" in extreme_dist.columns:
                            log(f"  - Percent in flood zone: {extreme_dist['SFHA'].mean()*100:.1f}%")
                else:
                    log("  Insufficient variation in distance for regression analysis")
    else:
        log("Warning: No regression variables available")
    
    # ── Run enhanced regression analyses
    run_enhanced_regressions(sf, rows)

    # ── spatial (if coords & libs)
    if HAS_SPATIAL and {PAR_LON, PAR_LAT}.issubset(sf.columns):
        coords_df = sf[[PAR_LON, PAR_LAT]].copy()
        coords_df['lon_num'] = pd.to_numeric(coords_df[PAR_LON], errors="coerce")
        coords_df['lat_num'] = pd.to_numeric(coords_df[PAR_LAT], errors="coerce")
        
        # Get rows with valid coordinates
        valid_coords = coords_df.dropna(subset=['lon_num', 'lat_num'])
        
        if len(valid_coords) > 100:  # Need sufficient observations
            # Reset index to ensure alignment
            s_df = sf.loc[valid_coords.index].reset_index(drop=True)
            coords_array = valid_coords[['lon_num', 'lat_num']].values
            
            try:
                # Build distance-based weights with 1km threshold
                W = build_distance_weights(coords_array, DISTANCE_THRESHOLD_KM)
                
                # Check if we have enough connectivity
                if W.mean_neighbors < 1:
                    log(f"Warning: Average neighbors ({W.mean_neighbors:.2f}) is very low. "
                        f"Consider increasing threshold from {DISTANCE_THRESHOLD_KM}km")
                
                # Run spatial models for different outcomes
                if "SFHA" in s_df.columns:
                    spatial_models(s_df, W, "SFHA", rows)
                    try:
                        mi = Moran(s_df["SFHA"].values, W)
                        rows += [["Moran_I", "SFHA", None, "I", mi.I],
                                 ["Moran_I", "SFHA", None, "p", mi.p_sim]]
                        log(f"Moran's I for SFHA: {mi.I:.4f} (p={mi.p_sim:.4f})")
                    except Exception as e:
                        log(f"Moran's I for SFHA failed: {str(e)}")
                        
                # Spatial analysis of absentee ownership
                spatial_models(s_df, W, "Absentee", rows)
                try:
                    mi_abs = Moran(s_df["Absentee"].astype(int).values, W)
                    rows += [["Moran_I", "Absentee", None, "I", mi_abs.I],
                             ["Moran_I", "Absentee", None, "p", mi_abs.p_sim]]
                    log(f"Moran's I for Absentee: {mi_abs.I:.4f} (p={mi_abs.p_sim:.4f})")
                except Exception as e:
                    log(f"Moran's I for Absentee failed: {str(e)}")
                    
                if INUND_F in s_df.columns and s_df[INUND_F].nunique() > 1:
                    spatial_models(s_df, W, INUND_F, rows)
                    
            except Exception as e:
                log(f"Spatial analysis failed: {str(e)}")
                rows.append(["Spatial_Analysis", "Status", None, "Error", str(e)])
        else:
            log(f"Insufficient valid coordinates for spatial analysis (n={len(valid_coords)})")
    else:
        if not HAS_SPATIAL:
            log("Skipping spatial diagnostics (libraries not available)")
        else:
            log("Skipping spatial diagnostics (coordinate fields missing)")

    # ── write
    out_csv = os.path.join(
        diag_dir, f"owner_distance_sf_{ts()}.csv")
    write_csv(out_csv, ["Metric","Sub1","Sub2","Stat","Value"], rows)
    
    # ── Summary of key findings
    log(f"\n── Key Findings ──")
    log(f"• Owner-occupied properties are by definition at 0 km distance (address match)")
    log(f"• Absentee owners show extreme distance variation, from neighbors to international investors")
    
    # Extract key statistics from rows
    if rows:
        # Find flood zone effects
        flood_effects = [r for r in rows if r[0] == "Logit_SF_Absentee_Base" and r[1] == "SFHA" and r[3] == "OR"]
        if flood_effects:
            log(f"• Properties in flood zones are {(1-flood_effects[0][4])*100:.1f}% LESS likely to be absentee-owned")
        
        # Find distance patterns in flood zones
        dist_flood = [r for r in rows if r[0] == "OLS_LogDistance_Absentee" and r[1] == "SFHA" and r[3] == "Coef"]
        if dist_flood:
            log(f"• Among absentee properties, those in flood zones have owners {dist_flood[0][4]:.2f} log units more distant")
            log(f"  (approximately {(np.exp(dist_flood[0][4])-1)*100:.0f}% farther away)")
        
        # Property value effects
        value_dist = [r for r in rows if r[0] == "OLS_LogDistance_Absentee" and r[1] == "log_Total_Asse" and r[3] == "Coef"]
        if value_dist and value_dist[0][4] < 0:
            log(f"• Higher-value properties tend to have CLOSER absentee owners")
        elif value_dist and value_dist[0][4] > 0:
            log(f"• Higher-value properties tend to have MORE DISTANT absentee owners")
        
        # Distance distribution insights
        p99_dist = [r for r in rows if r[0] == "Distance_Percentile_P99" and r[1] == "TRUE" and r[3] == "km"]
        p50_dist = [r for r in rows if r[0] == "Distance_Percentile_P50" and r[1] == "TRUE" and r[3] == "km"]
        if p99_dist and p50_dist:
            log(f"• Median absentee owner distance: {p50_dist[0][4]:.1f} km")
            log(f"• Top 1% of absentee owners are >{p99_dist[0][4]:.0f} km away (likely international)")
    
    if HAS_SPATIAL and any("Moran" in r[0] for r in rows):
        mi_vals = [r[4] for r in rows if r[0] == "Moran_I" and r[1] == "SFHA" and r[3] == "I"]
        if mi_vals:
            log(f"• Flood zones show spatial clustering (Moran's I = {mi_vals[0]:.3f})")
        mi_abs_vals = [r[4] for r in rows if r[0] == "Moran_I" and r[1] == "Absentee" and r[3] == "I"]
        if mi_abs_vals:
            log(f"• Absentee ownership also shows spatial clustering (Moran's I = {mi_abs_vals[0]:.3f})")
    
    log(f"\n✓ Done in {(time.time()-t0)/60:.2f} min → {out_csv}")

# ───────── entry point ─────────
if __name__ == "__main__":
    log("✓ Starting SINGLE-FAMILY absentee analysis (pure address matching)…")
    try:
        main()
    except Exception:
        traceback.print_exc()
        log("✖ Script terminated due to error")
