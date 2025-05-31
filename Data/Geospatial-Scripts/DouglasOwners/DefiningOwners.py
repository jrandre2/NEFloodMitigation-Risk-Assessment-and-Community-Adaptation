#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Flood Risk Analysis with Distance Bands
================================================================================


Key Features:
• Correctly identifies owner-occupied properties through address matching
• Defines owner types based on distance bands and geographic adjacency
• Approximates adjacent counties using OWNER_CATEGORY field and distance thresholds
• Uses granular distance bands to analyze flood risk patterns
• Separates residential owner types from commercial properties
• Avoids investor/landlord terminology - uses neutral "owner" descriptions

OWNER TYPE CATEGORIES:
• Owner-Occupied: Owner address matches property address (residential only)
• Adjacent_County_Owner: Approximated using OWNER_CATEGORY and distance:
  - Same ZIP, different address, <5km away
  - Same county area (InCounty) <15km away
  - Out of county/state but <20km away (captures Pottawattamie, IA)
• Near_Owner_0_10km: Non-resident owners within 10km
• Near_Owner_10_25km: Non-resident owners 10-25km away
• Near_Owner_25_50km: Non-resident owners 25-50km away
• Regional_Owner_50_100km: Non-resident owners 50-100km away
• Regional_Owner_100_200km: Non-resident owners 100-200km away
• Distant_Owner_200_500km: Non-resident owners 200-500km away
• Distant_Owner_500km_plus: Non-resident owners beyond 500km
• Commercial: All commercial/industrial properties (separate category)

PRIMARY RESEARCH QUESTIONS:
1. Does flood risk vary by owner distance in granular bands?
2. Do owners in approximated adjacent counties show different risk patterns?
3. Is there a distance threshold where flood risk patterns change?
4. Are there spatial clusters of different ownership distance patterns?
"""

from __future__ import annotations
import os, csv, time, warnings, traceback, re
from datetime import datetime
from typing import List, Dict, Any, Tuple

# Set environment variable to reduce verbosity
os.environ['PYSAL_VERBOSE'] = 'False'
os.environ['LIBPYSAL_VERBOSE'] = 'False'

import arcpy
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.anova import anova_lm

# Import spatial libraries with suppressed warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import libpysal as ps
    from spreg import GM_Lag, GM_Error
    from esda.moran import Moran

# Try to set libpysal logger to reduce verbosity
try:
    import logging
    logging.getLogger('libpysal').setLevel(logging.ERROR)
    logging.getLogger('pysal').setLevel(logging.ERROR)
except:
    pass

# Suppress warnings
warnings.filterwarnings("ignore", message="overflow encountered")
warnings.filterwarnings("ignore", message="covariance")
warnings.filterwarnings("ignore", message="invalid value")
warnings.filterwarnings("ignore", message="Perfect separation")
warnings.filterwarnings("ignore", message="The weights matrix is not fully connected")
warnings.filterwarnings("ignore", message="is an island")

# Redirect stdout to suppress libpysal island warnings
import sys
from contextlib import contextmanager
import io

@contextmanager
def suppress_stdout():
    """Context manager to suppress stdout and stderr (for libpysal island warnings)."""
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = io.StringIO()
    sys.stderr = io.StringIO()
    try:
        yield
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr

# ── project paths & layers ─────────────────────────────────────────────
PROJECT_DIR = r"C:\Mac\Home\Documents\ArcGIS\Projects\OwnerDistanceProject"
GDB         = os.path.join(PROJECT_DIR, "OwnerDistanceProject.gdb")
PARCELS_FC  = os.path.join(GDB, "Parcels")
FIRM_FC     = os.path.join(GDB, "FIRM")
INUND_POLY  = os.path.join(GDB, "InundationPolygon")

# ── field names --------------------------------------------------------
PID, DIST   = "Parcel_ID", "OwnerDist_km"
ZONE_FIRM   = "FLD_ZONE"
LU_CODE     = "Property_P"
PLOT_SIZE   = "Shape_Area"
INUND_F     = "ActualInundated"
STRUCT_VAL  = "Total_Asse"
YR_BUILT    = "BuildingYe"
PAR_LON, PAR_LAT = "ParcelLon", "ParcelLat"

# Owner location fields
OWNER_INZIP = "Owner_InZIP"
OWNER_CATEGORY = "OwnerCategory"  # May not exist in all datasets
OWNER_COUNTY = "OwnerCounty"  # Field for owner county (not used in approximation)
OWNER_STATE = "OwnerState"    # Field for owner state (not used in approximation)
PARCEL_COUNTY = "ParcelCounty"  # Field for parcel county (not used in approximation)

# Address fields for matching
PARCEL_ADDR_OPTIONS = ["Ph_Full_Ad", "Situs_Addr", "Ph_Full_Add"]
OWNER_ADDR_OPTIONS = ["Street_Add", "OW1_Addres", "Current_Ow"]
PARCEL_ZIP = "ParcelZIP5"
OWNER_ZIP = "Zip"
OWNER_ZIP_ALT = "OwnerZIP5"

SFHA_CODES  = {"A","AE","AO","AH","AR","A99","V","VE"}
NEIGHBORS_K = 16

# No longer needed - we use OWNER_CATEGORY and distance to approximate adjacent owners

LU_MAP: Dict[int,str] = {
    0:"Other_LU",
    1:"Residential", 
    2:"Commercial",
    3:"Commercial",
    4:"Industrial/Mixed Use",
    6:"Industrial/Mixed Use",
    9:"Industrial/Mixed Use"
}

# ── helpers ------------------------------------------------------------
log = lambda m: (arcpy.AddMessage(m), print(m))
ts  = lambda: datetime.now().strftime("%Y%m%d_%H%M%S")

# Filtered logging to skip island warnings
def log_filtered(m: str) -> None:
    """Log message only if it's not an island warning."""
    if "is an island" not in str(m).lower():
        log(m)

def write_csv(path: str, header: List[str], rows: List[List[Any]]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows([header, *rows])

def safe_exp(value: float) -> float:
    """Safely exponentiate a value, capping at reasonable limits to avoid overflow."""
    if pd.isna(value):
        return np.nan
    capped_value = np.clip(value, -20, 20)
    return np.exp(capped_value)

def cliffs_d(x: np.ndarray, y: np.ndarray) -> float:
    """Cliff's δ effect-size (x − y)."""
    m, n = len(x), len(y)
    if m == 0 or n == 0: return np.nan
    gt = np.sum(x[:, None] > y)
    lt = np.sum(x[:, None] < y)
    return (gt - lt) / (m * n)

def standardize_address(addr: str) -> str:
    """Standardize address for comparison by removing common variations."""
    if pd.isna(addr) or addr == "" or addr is None:
        return ""
    
    # Convert to string and uppercase, strip whitespace
    addr = str(addr).upper().strip()
    
    # Remove PO BOX addresses as they won't match physical addresses
    if "PO BOX" in addr or "P.O. BOX" in addr:
        return ""
    
    # Remove punctuation
    addr = addr.replace(".", "").replace(",", "").replace("-", " ").replace("'", "")
    
    # Common replacements - comprehensive list
    replacements = {
        " STREET": " ST",
        " AVENUE": " AVE",
        " ROAD": " RD",
        " DRIVE": " DR",
        " LANE": " LN",
        " BOULEVARD": " BLVD",
        " PLACE": " PL",
        " COURT": " CT",
        " CIRCLE": " CIR",
        " PARKWAY": " PKWY",
        " HIGHWAY": " HWY",
        " TRAIL": " TRL",
        " TERRACE": " TER",
        " NORTH ": " N ",
        " SOUTH ": " S ",
        " EAST ": " E ",
        " WEST ": " W ",
        " NORTHEAST ": " NE ",
        " NORTHWEST ": " NW ",
        " SOUTHEAST ": " SE ",
        " SOUTHWEST ": " SW ",
        " FIRST ": " 1ST ",
        " SECOND ": " 2ND ",
        " THIRD ": " 3RD ",
        " FOURTH ": " 4TH ",
        " FIFTH ": " 5TH ",
        " SIXTH ": " 6TH ",
        " SEVENTH ": " 7TH ",
        " EIGHTH ": " 8TH ",
        " NINTH ": " 9TH ",
        " TENTH ": " 10TH ",
        "  ": " "  # Remove double spaces
    }
    
    # Apply replacements
    for old, new in replacements.items():
        addr = addr.replace(old, new)
    
    # Remove apartment/unit/suite numbers - more comprehensive
    addr = re.sub(r'\s+(APT|APARTMENT|UNIT|SUITE|STE|#|BLDG|BUILDING|FL|FLOOR|RM|ROOM)\s*[A-Z0-9\-]*', '', addr)
    
    # Remove trailing direction if it appears at the end (like "123 MAIN ST W")
    addr = re.sub(r'\s+(N|S|E|W|NE|NW|SE|SW)$', '', addr)
    
    # Final cleanup - remove any remaining extra spaces
    addr = " ".join(addr.split())
    
    return addr

def extract_street_number(addr: str) -> str:
    """Extract just the street number from an address."""
    if not addr:
        return ""
    # Match digits at the beginning of the address
    match = re.match(r'^(\d+)', addr)
    return match.group(1) if match else ""

def addresses_match(addr1: str, addr2: str, zip1: Any, zip2: Any) -> bool:
    """Check if two addresses match, with more flexible matching."""
    # First check if ZIPs match (if both are provided)
    if pd.notna(zip1) and pd.notna(zip2):
        try:
            # Convert to string and compare first 5 digits only
            zip1_str = str(int(float(str(zip1))))[:5] if pd.notna(zip1) else ""
            zip2_str = str(int(float(str(zip2))))[:5] if pd.notna(zip2) else ""
            if len(zip1_str) >= 5 and len(zip2_str) >= 5 and zip1_str != zip2_str:
                return False
        except (ValueError, TypeError):
            # If conversion fails, try string comparison
            zip1_str = str(zip1).strip()[:5]
            zip2_str = str(zip2).strip()[:5]
            if len(zip1_str) >= 5 and len(zip2_str) >= 5 and zip1_str != zip2_str:
                return False
    
    # Standardize both addresses
    std_addr1 = standardize_address(addr1)
    std_addr2 = standardize_address(addr2)
    
    # If either address is empty after standardization, consider it non-matching
    if not std_addr1 or not std_addr2:
        return False
    
    # Check exact match first
    if std_addr1 == std_addr2:
        return True
    
    # Try matching just the street numbers and first few words
    # This helps when one address is abbreviated
    num1 = extract_street_number(std_addr1)
    num2 = extract_street_number(std_addr2)
    
    if num1 and num2 and num1 == num2:
        # Get the street part after the number
        street1 = std_addr1[len(num1):].strip()
        street2 = std_addr2[len(num2):].strip()
        
        # Split into words
        words1 = street1.split()
        words2 = street2.split()
        
        # If both have at least one word and they match, consider it a match
        if words1 and words2:
            # Check if first word matches (street name)
            if words1[0] == words2[0]:
                # If we have street type, check if they match or are compatible
                if len(words1) > 1 and len(words2) > 1:
                    # Both have street type - they should match
                    return words1[1] == words2[1]
                else:
                    # One or both missing street type - consider it a match
                    return True
    
    return False

# ── data preparation utilities ─────────────────────────────────────────
def check_and_add_field(fc: str, field_name: str, field_type: str = "SHORT") -> None:
    """Checks if a field exists, adds it if not."""
    if field_name not in [f.name for f in arcpy.ListFields(fc)]:
        log(f"Adding field '{field_name}' to {os.path.basename(fc)}")
        arcpy.management.AddField(fc, field_name, field_type)
    else:
        log(f"Field '{field_name}' already exists in {os.path.basename(fc)}")

def flag_inundated_parcels(parcels_fc: str, inundation_poly_fc: str, flag_field: str, scratch_gdb: str) -> None:
    """Flags parcels that intersect with inundation polygons."""
    log(f"▸ Flagging inundated parcels using '{os.path.basename(inundation_poly_fc)}' into field '{flag_field}'...")
    check_and_add_field(parcels_fc, flag_field, "SHORT")

    log(f"  Initializing inundation flag to 0 for all parcels...")
    with arcpy.da.UpdateCursor(parcels_fc, [flag_field]) as cursor:
        for row in cursor:
            row[0] = 0
            cursor.updateRow(row)

    parcels_layer = "parcels_lyr_inund"
    if arcpy.Exists(parcels_layer): arcpy.management.Delete(parcels_layer)
    arcpy.management.MakeFeatureLayer(parcels_fc, parcels_layer)
    
    log(f"  Selecting parcels intersecting '{os.path.basename(inundation_poly_fc)}'...")
    arcpy.management.SelectLayerByLocation(parcels_layer, "INTERSECT", inundation_poly_fc, selection_type="NEW_SELECTION")
    
    count_selected = int(arcpy.management.GetCount(parcels_layer).getOutput(0))
    log(f"  Found {count_selected} parcels intersecting inundation polygons.")

    if count_selected > 0:
        log(f"  Updating '{flag_field}' to 1 for selected inundated parcels...")
        with arcpy.da.UpdateCursor(parcels_layer, [flag_field]) as cursor: 
            for row in cursor:
                row[0] = 1
                cursor.updateRow(row)

    arcpy.management.Delete(parcels_layer)
    log("✓ Inundation flagging complete.")

def firm_overlay_needed(fc: str) -> bool:
    """True if fc lacks ZONE_FIRM or still contains NULLs/empty strings in that field."""
    if ZONE_FIRM not in (f.name for f in arcpy.ListFields(fc)):
        return True
    lyr = "_nulls_firm_check"
    if arcpy.Exists(lyr): arcpy.management.Delete(lyr)
    sql = f"{arcpy.AddFieldDelimiters(fc, ZONE_FIRM)} IS NULL OR {arcpy.AddFieldDelimiters(fc, ZONE_FIRM)} = ''"
    arcpy.management.MakeFeatureLayer(fc, lyr, sql)
    nulls = int(arcpy.management.GetCount(lyr).getOutput(0))
    arcpy.management.Delete(lyr)
    return nulls > 0

def majority_tabulate_firm(parcels_fc: str, firm_fc: str, scratch_gdb: str) -> None:
    """Populate ZONE_FIRM on parcels with the area-majority FIRM zone."""
    log("▸ Assigning majority FIRM zone …")
    check_and_add_field(parcels_fc, ZONE_FIRM, "TEXT") 

    tbl = os.path.join(scratch_gdb, "TabulateIntersection_FIRM")
    if arcpy.Exists(tbl): arcpy.management.Delete(tbl)

    arcpy.analysis.TabulateIntersection(
        in_zone_features=parcels_fc,
        zone_fields=PID,
        in_class_features=firm_fc,
        out_table=tbl,
        class_fields=ZONE_FIRM
    )

    best: Dict[Any, tuple[str, float]] = {} 
    with arcpy.da.SearchCursor(tbl, [PID, ZONE_FIRM, "AREA"]) as cursor:
        for p_id_val, zone, area in cursor:
            if p_id_val not in best or (area is not None and (best[p_id_val][1] is None or area > best[p_id_val][1])):
                best[p_id_val] = (str(zone) if zone is not None else "", float(area) if area is not None else 0.0)
    
    updated_count = 0
    with arcpy.da.UpdateCursor(parcels_fc, [PID, ZONE_FIRM]) as urows:
        for row in urows:
            if row[0] in best:
                row[1] = best[row[0]][0]
                urows.updateRow(row)
                updated_count +=1
    log(f"  Updated FIRM zones for {updated_count} parcels.")
    if arcpy.Exists(tbl): arcpy.management.Delete(tbl)
    log("✓ FIRM overlay complete.")

# ── Improved spatial models focusing on flood risk ─────────────────────
def spatial_flood_models(df: pd.DataFrame, W, flood_var: str, owner_type_col: str, out: List[List[Any]]):
    """Fit spatial models predicting flood risk from owner types."""
    try:
        log(f"\n📊 Spatial models predicting {flood_var} from ownership patterns:")
        
        # Create dummy variables for owner types
        owner_types = [ot for ot in df[owner_type_col].unique() if pd.notna(ot)]
        if len(owner_types) < 2:
            log(f"⚠ Not enough owner types for spatial models")
            return
            
        # Use the most common type as reference
        owner_type_counts = df[owner_type_col].value_counts()
        reference_type = owner_type_counts.index[0]
        
        log(f"  Using '{reference_type}' as reference category")
        
        # Build design matrix with owner type dummies
        X_vars = []
        for otype in owner_types:
            if otype != reference_type and pd.notna(otype):
                dummy_col = f"owner_type_{otype}"
                df[dummy_col] = (df[owner_type_col] == otype).astype(int)
                X_vars.append(dummy_col)
        
        # Add control variables
        control_vars = []
        if "log_Total_Asse" in df.columns:
            control_vars.append("log_Total_Asse")
        if "log_BuildYear" in df.columns:
            control_vars.append("log_BuildYear")
        if "log_PlotSize" in df.columns:
            control_vars.append("log_PlotSize")
        
        all_vars = X_vars + control_vars
        if not all_vars:
            log(f"⚠ No valid predictors for spatial models")
            return
        
        # Prepare data
        valid_idx = df[all_vars + [flood_var]].dropna().index
        if len(valid_idx) < 50:
            log(f"⚠ Not enough valid observations ({len(valid_idx)})")
            return
        
        df_valid = df.loc[valid_idx]
        y = df_valid[flood_var].values.reshape(-1, 1)
        X = df_valid[all_vars].values
        
        # Standardize X
        X_mean = X.mean(axis=0)
        X_std = X.std(axis=0)
        X_std[X_std == 0] = 1
        X = (X - X_mean) / X_std
        
        # Subset weights matrix
        from libpysal.weights import w_subset
        with suppress_stdout():
            W_subset = w_subset(W, valid_idx.tolist())
            W_subset.transform = "R"
        
        log(f"  N = {len(valid_idx)}, predictors = {len(all_vars)}")
        
        # SAR model
        try:
            with suppress_stdout():
                sar = GM_Lag(y, X, w=W_subset, name_y=flood_var)
            
            # Handle rho value
            rho_val = float(sar.rho) if hasattr(sar, 'rho') else np.nan
            
            # Get p-value for rho
            if hasattr(sar, 'z_rho') and len(sar.z_rho) >= 2:
                rho_p = float(sar.z_rho[1])
            else:
                rho_p = np.nan
            
            out.extend([
                [f"Spatial_SAR_{flood_var}_OwnerType", "rho", None, "coef", rho_val],
                [f"Spatial_SAR_{flood_var}_OwnerType", "rho", None, "p", rho_p],
            ])
            
            # Report coefficients for owner types
            betas = sar.betas.flatten()
            std_errs = sar.std_err.flatten()
            for i, var in enumerate(X_vars):
                owner_type = var.replace("owner_type_", "")
                b_adj = betas[i+1] / X_std[i]  # +1 for intercept
                se_adj = std_errs[i+1] / X_std[i]
                z_stat = b_adj / se_adj if se_adj > 0 else 0
                p_val = 2 * (1 - stats.norm.cdf(abs(z_stat)))
                
                out.extend([
                    [f"Spatial_SAR_{flood_var}_OwnerType", owner_type, None, "beta", float(b_adj)],
                    [f"Spatial_SAR_{flood_var}_OwnerType", owner_type, None, "se", float(se_adj)],
                    [f"Spatial_SAR_{flood_var}_OwnerType", owner_type, None, "p", float(p_val)],
                ])
                
                log(f"  {owner_type}: β = {b_adj:.4f}, p = {p_val:.4f}")
            
            log(f"  ✓ SAR model completed (N={len(valid_idx)}, rho={rho_val:.3f})")
                
        except Exception as e:
            log(f"  ✖ SAR failed: {str(e)}")
            
    except Exception as e:
        log(f"⚠ Error in spatial flood models: {str(e)}")
        traceback.print_exc()

# ── Stratified spatial analysis ────────────────────────────────────────
def stratified_morans_i(df: pd.DataFrame, W, outcome_var: str, strata_col: str, out: List[List[Any]]):
    """Calculate Moran's I separately for each stratum."""
    log(f"\n📊 Stratified Moran's I for {outcome_var} by {strata_col}:")
    
    for stratum in df[strata_col].unique():
        if pd.isna(stratum):
            continue
            
        stratum_idx = df[df[strata_col] == stratum].index.tolist()
        if len(stratum_idx) < 30:  # Skip small strata
            continue
            
        try:
            from libpysal.weights import w_subset
            with suppress_stdout():
                W_stratum = w_subset(W, stratum_idx)
                W_stratum.transform = "R"
                
                y_stratum = df.loc[stratum_idx, outcome_var].values
                mi = Moran(y_stratum, W_stratum)
            
            out.extend([
                [f"Stratified_Moran_{outcome_var}", str(stratum), None, "I", float(mi.I)],
                [f"Stratified_Moran_{outcome_var}", str(stratum), None, "p_value", float(mi.p_sim)],
                [f"Stratified_Moran_{outcome_var}", str(stratum), None, "N", int(len(stratum_idx))],
            ])
            
            log(f"  {stratum}: I = {mi.I:.3f}, p = {mi.p_sim:.3f}, N = {len(stratum_idx)}")
            
        except Exception as e:
            log(f"  ✖ Failed for {stratum}: {str(e)}")

# ── Multinomial logistic regression ────────────────────────────────────
def multinomial_ownership_model(df: pd.DataFrame, out: List[List[Any]]):
    """Multinomial logistic regression predicting owner type from flood risk."""
    log("\n📊 Multinomial logistic regression: Owner Type ~ Flood Risk")
    
    try:
        # Filter to valid observations
        model_vars = ["OwnerType", "SFHA"]
        optional_vars = ["log_Total_Asse", "log_BuildYear"]
        for var in optional_vars:
            if var in df.columns:
                model_vars.append(var)
        
        df_model = df[model_vars].dropna()
        
        # Remove 'Unknown' and commercial categories, and small categories
        # Keep only categories with sufficient observations
        owner_type_counts = df_model["OwnerType"].value_counts()
        valid_types = owner_type_counts[owner_type_counts >= 100].index.tolist()
        
        df_model = df_model[df_model["OwnerType"].isin(valid_types)]
        df_model = df_model[~df_model["OwnerType"].isin(["Unknown", "Commercial"])]
        
        if len(df_model) < 100 or df_model["OwnerType"].nunique() < 2:
            log("⚠ Not enough data or categories for multinomial model")
            return
        
        # Create numeric encoding for owner types
        owner_types = df_model["OwnerType"].unique()
        owner_type_map = {ot: i for i, ot in enumerate(owner_types)}
        df_model["owner_type_num"] = df_model["OwnerType"].map(owner_type_map)
        
        # Try simpler model first - just SFHA
        try:
            from statsmodels.discrete.discrete_model import MNLogit
            
            X = sm.add_constant(df_model[["SFHA"]])
            y = df_model["owner_type_num"]
            
            # Fit with stricter convergence criteria
            model = MNLogit(y, X).fit(method='bfgs', maxiter=100, disp=False)
            
            # Extract results
            log(f"  Reference category: {owner_types[0]}")
            log(f"  Model converged: {model.mle_retvals['converged']}")
            
            # Only report if converged
            if model.mle_retvals['converged']:
                for i in range(1, len(owner_types)):
                    owner_type = owner_types[i]
                    
                    try:
                        # Get coefficients for this outcome - params is a 2D array
                        if hasattr(model.params, 'iloc'):
                            sfha_coef = model.params.iloc[1, i-1]  # Second row (SFHA), column for this outcome
                            sfha_p = model.pvalues.iloc[1, i-1]
                        else:
                            # Handle as numpy array
                            sfha_coef = model.params[1, i-1] if model.params.ndim > 1 else model.params[1]
                            sfha_p = model.pvalues[1, i-1] if model.pvalues.ndim > 1 else model.pvalues[1]
                        
                        sfha_rr = np.exp(sfha_coef) if not np.isnan(sfha_coef) else np.nan
                        
                        if not np.isnan(sfha_rr) and not np.isnan(sfha_p):
                            out.extend([
                                [f"Multinomial_{owner_type}_vs_{owner_types[0]}", "SFHA", None, "RRR", float(sfha_rr)],
                                [f"Multinomial_{owner_type}_vs_{owner_types[0]}", "SFHA", None, "coef", float(sfha_coef)],
                                [f"Multinomial_{owner_type}_vs_{owner_types[0]}", "SFHA", None, "p", float(sfha_p)],
                            ])
                            
                            log(f"  {owner_type} vs {owner_types[0]}: SFHA RRR = {sfha_rr:.3f}, p = {sfha_p:.3f}")
                    except Exception as e:
                        log(f"  ✖ Failed to extract coefficients for {owner_type}: {str(e)}")
                
                # Model fit statistics
                out.extend([
                    ["Multinomial_Model", "Fit", None, "LogLik", float(model.llf)],
                    ["Multinomial_Model", "Fit", None, "AIC", float(model.aic)],
                    ["Multinomial_Model", "Fit", None, "N", int(len(df_model))],
                ])
            else:
                log("  ✖ Model did not converge - results not reliable")
                
        except Exception as e:
            log(f"  ✖ Simplified model also failed: {str(e)}")
            
        log("✓ Multinomial model complete")
        
    except Exception as e:
        log(f"✖ Multinomial model failed: {str(e)}")
        traceback.print_exc()

# ── Improved descriptive statistics ────────────────────────────────────
def owner_type_flood_analysis(df: pd.DataFrame, out: List[List[Any]]):
    """Analyze flood risk by owner type with proper comparisons."""
    log("\n📊 Flood risk analysis by owner type:")
    
    # Overall statistics by owner type
    for owner_type in df["OwnerType"].unique():
        if pd.isna(owner_type):
            continue
            
        ot_df = df[df["OwnerType"] == owner_type]
        
        out.extend([
            [f"OwnerType_{owner_type}", "Summary", None, "N", int(len(ot_df))],
            [f"OwnerType_{owner_type}", "Summary", None, "SFHA_rate", float(ot_df["SFHA"].mean() if not ot_df["SFHA"].isna().all() else 0)],
            [f"OwnerType_{owner_type}", "Summary", None, "Inundated_rate", float(ot_df[INUND_F].mean() if not ot_df[INUND_F].isna().all() else 0)],
            [f"OwnerType_{owner_type}", "Summary", None, "Avg_distance_km", float(ot_df[DIST].mean() if not ot_df[DIST].isna().all() else 0)],
            [f"OwnerType_{owner_type}", "Summary", None, "Median_distance_km", float(ot_df[DIST].median() if not ot_df[DIST].isna().all() else 0)],
        ])
        
        # Shorten the display name for logging
        display_name = owner_type.replace("_Owner", "").replace("Owner_", "").replace("_", " ")
        median_dist = ot_df[DIST].median()
        if pd.isna(median_dist):
            dist_str = "nankm"
        else:
            dist_str = f"{median_dist:.1f}km"
        
        # Handle potentially missing values
        sfha_rate = ot_df["SFHA"].mean() if "SFHA" in ot_df.columns and not ot_df["SFHA"].isna().all() else 0
        inund_rate = ot_df[INUND_F].mean() if INUND_F in ot_df.columns and not ot_df[INUND_F].isna().all() else 0
        
        log(f"  {display_name}: N={len(ot_df)}, SFHA={sfha_rate:.3f}, "
            f"Inund={inund_rate:.3f}, Dist={dist_str}")
    
    # Chi-square tests for categorical relationships
    try:
        # Owner type vs SFHA
        ct_sfha = pd.crosstab(df["OwnerType"], df["SFHA_status"])
        chi2_sfha, p_sfha, dof_sfha, _ = stats.chi2_contingency(ct_sfha)
        
        # Cramér's V
        n = ct_sfha.sum().sum()
        min_dim = min(ct_sfha.shape[0] - 1, ct_sfha.shape[1] - 1)
        cramers_v_sfha = np.sqrt(chi2_sfha / (n * min_dim))
        
        out.extend([
            ["OwnerType_vs_SFHA", "ChiSquare", None, "statistic", float(chi2_sfha)],
            ["OwnerType_vs_SFHA", "ChiSquare", None, "p_value", float(p_sfha)],
            ["OwnerType_vs_SFHA", "ChiSquare", None, "dof", int(dof_sfha)],
            ["OwnerType_vs_SFHA", "ChiSquare", None, "CramersV", float(cramers_v_sfha)],
        ])
        
        log(f"\n  Owner Type vs SFHA: χ² = {chi2_sfha:.2f}, p = {p_sfha:.4f}, V = {cramers_v_sfha:.3f}")
        
        # Owner type vs Inundation
        if df[INUND_F].nunique() > 1:
            ct_inund = pd.crosstab(df["OwnerType"], df[INUND_F])
            chi2_inund, p_inund, dof_inund, _ = stats.chi2_contingency(ct_inund)
            cramers_v_inund = np.sqrt(chi2_inund / (n * min(ct_inund.shape[0] - 1, ct_inund.shape[1] - 1)))
            
            out.extend([
                ["OwnerType_vs_Inundation", "ChiSquare", None, "statistic", float(chi2_inund)],
                ["OwnerType_vs_Inundation", "ChiSquare", None, "p_value", float(p_inund)],
                ["OwnerType_vs_Inundation", "ChiSquare", None, "dof", int(dof_inund)],
                ["OwnerType_vs_Inundation", "ChiSquare", None, "CramersV", float(cramers_v_inund)],
            ])
            
            log(f"  Owner Type vs Inundation: χ² = {chi2_inund:.2f}, p = {p_inund:.4f}, V = {cramers_v_inund:.3f}")
            
    except Exception as e:
        log(f"✖ Chi-square tests failed: {str(e)}")

# ── Non-tautological regression analyses ───────────────────────────────
def flood_risk_regressions(df: pd.DataFrame, out: List[List[Any]]):
    """Run regressions predicting flood risk from owner characteristics."""
    log("\n📊 Logistic regressions predicting flood risk:")
    
    # Model 1: SFHA ~ Owner Type + controls
    if "OwnerType" in df.columns:
        # Find the most common owner type to use as reference
        owner_type_counts = df["OwnerType"].value_counts()
        reference_cat = owner_type_counts.index[0]  # Most common category
        
        formula_sfha = f"SFHA ~ C(OwnerType, Treatment(reference='{reference_cat}'))"
        
        # Add controls
        if "log_Total_Asse" in df.columns:
            formula_sfha += " + log_Total_Asse"
        if "log_BuildYear" in df.columns:
            formula_sfha += " + log_BuildYear"
        if "log_PlotSize" in df.columns:
            formula_sfha += " + log_PlotSize"
        
        run_and_log_regression(df, formula_sfha, sm.families.Binomial(), 
                              "Logit_SFHA_OwnerType", out)
    
    # Model 2: Among non-resident owners only, does distance matter?
    # Include only residential non-owner-occupied properties
    df_nonresident = df[~df["OwnerType"].isin(["Owner_Occupied", "Commercial", "Unknown"])].copy()
    if len(df_nonresident) > 50:
        log(f"\n  Among {len(df_nonresident)} non-resident owners:")
        
        # Continuous distance
        formula_dist = "SFHA ~ OwnerDist_km"
        if "log_Total_Asse" in df_nonresident.columns:
            formula_dist += " + log_Total_Asse"
        if "log_BuildYear" in df_nonresident.columns:
            formula_dist += " + log_BuildYear"
            
        run_and_log_regression(df_nonresident, formula_dist, sm.families.Binomial(),
                              "Logit_SFHA_Distance_NonResidentOnly", out)
        
        # Log distance
        formula_logdist = "SFHA ~ log_dist"
        if "log_Total_Asse" in df_nonresident.columns:
            formula_logdist += " + log_Total_Asse"
        if "log_BuildYear" in df_nonresident.columns:
            formula_logdist += " + log_BuildYear"
            
        run_and_log_regression(df_nonresident, formula_logdist, sm.families.Binomial(),
                              "Logit_SFHA_LogDistance_NonResidentOnly", out)
    
    # Model 3: Inundation models (if applicable)
    if INUND_F in df.columns and df[INUND_F].nunique() > 1:
        log("\n  Predicting actual inundation:")
        
        # Use the same reference category as above
        if "OwnerType" in df.columns:
            owner_type_counts = df["OwnerType"].value_counts()
            reference_cat = owner_type_counts.index[0]
            
            formula_inund = f"{INUND_F} ~ C(OwnerType, Treatment(reference='{reference_cat}'))"
            if "log_Total_Asse" in df.columns:
                formula_inund += " + log_Total_Asse"
            if "log_BuildYear" in df.columns:
                formula_inund += " + log_BuildYear"
                
            run_and_log_regression(df, formula_inund, sm.families.Binomial(),
                                  "Logit_Inundation_OwnerType", out)

def run_and_log_regression(df: pd.DataFrame, formula: str, family_obj: Any, 
                          model_type_label: str, rows: List[List[Any]]):
    """Runs a GLM or OLS regression, logs results to rows list."""
    try:
        # Extract variables from formula
        formula_parts = formula.split('~')
        dependent_var = formula_parts[0].strip()
        independent_vars = formula_parts[1].strip()
        
        # Parse out variable names more carefully
        import re
        all_vars = re.findall(r'[A-Za-z_]\w*', formula)
        all_vars = [v for v in all_vars if v not in ['C', 'Treatment', 'reference'] and v in df.columns]
        
        if not all_vars:
            log(f"✖ No valid variables found in formula for {model_type_label}")
            return
            
        df_model = df[all_vars].dropna()
        
        if len(df_model) < 50:
            log(f"✖ Not enough data for {model_type_label} (N={len(df_model)})")
            return

        if family_obj:  # GLM
            model = smf.glm(formula, data=df_model, family=family_obj).fit()
            pseudo_r2 = 1 - (model.llf / model.llnull) if model.llnull != 0 else np.nan
            rows.extend([
                [model_type_label, "ModelFit", None, "Pseudo_R2_McFadden", float(pseudo_r2)],
                [model_type_label, "ModelFit", None, "AIC", float(model.aic)],
                [model_type_label, "ModelFit", None, "BIC", float(model.bic)],
                [model_type_label, "ModelFit", None, "N", int(len(df_model))],
            ])
        else:  # OLS
            model = smf.ols(formula, data=df_model).fit()
            rows.extend([
                [model_type_label, "ModelFit", None, "R_Squared", float(model.rsquared)],
                [model_type_label, "ModelFit", None, "Adj_R_Squared", float(model.rsquared_adj)],
                [model_type_label, "ModelFit", None, "AIC", float(model.aic)],
                [model_type_label, "ModelFit", None, "BIC", float(model.bic)],
                [model_type_label, "ModelFit", None, "N", int(len(df_model))],
            ])

        # Log coefficients
        for term in model.params.index:
            coef = model.params[term]
            pval = model.pvalues[term]
            conf_int = model.conf_int()
            
            if family_obj and hasattr(family_obj, 'link') and isinstance(family_obj.link, sm.families.links.Logit):
                rows.extend([
                    [model_type_label, term, None, "OR", float(safe_exp(coef))],
                    [model_type_label, term, None, "coef", float(coef)],
                    [model_type_label, term, None, "p_value", float(pval)],
                ])
            else:
                rows.extend([
                    [model_type_label, term, None, "Coefficient", float(coef)],
                    [model_type_label, term, None, "p_value", float(pval)],
                ])
                
        log(f"✓ {model_type_label} complete (N={len(df_model)})")

    except Exception as e:
        log(f"✖ Error in {model_type_label}: {e}")
        traceback.print_exc()

# ── main analysis routine ──────────────────────────────────────────────
def run_analysis(df_base: pd.DataFrame, W) -> List[List[Any]]:
    """Run improved non-tautological analyses."""
    df = df_base.copy()
    out: List[List[Any]] = []
    
    # Data preparation
    df[DIST] = pd.to_numeric(df[DIST], errors="coerce")  # Ensure distance is numeric
    df["log_dist"] = pd.to_numeric(df[DIST], errors="coerce").pipe(np.log1p)
    df["SFHA"] = df[ZONE_FIRM].isin(SFHA_CODES).astype(int)
    df["SFHA_status"] = np.where(df["SFHA"] == 1, "InSFHA", "NotInSFHA")
    
    # Ensure land use mapping
    df[LU_CODE] = pd.to_numeric(df[LU_CODE], errors='coerce').fillna(0).astype(int)
    df["LandUse"] = df[LU_CODE].map(LU_MAP).fillna("Other_LU")
    
    # Create non-tautological owner types with proper geographic hierarchy and distance bands
    log("\n▸ Creating owner type categories with distance bands...")
    
    # Initialize
    df["OwnerType"] = "Unknown"
    
    # First separate commercial properties
    commercial_mask = df["LandUse"].isin(["Commercial", "Industrial/Mixed Use", "Other_LU"])
    df.loc[commercial_mask, "OwnerType"] = "Commercial"
    
    # For residential properties, determine owner type
    residential_mask = df["LandUse"] == "Residential"
    
    # Check address matching for owner-occupied
    parcel_addr_field = next((f for f in PARCEL_ADDR_OPTIONS if f in df.columns), None)
    owner_addr_field = next((f for f in OWNER_ADDR_OPTIONS if f in df.columns), None)
    parcel_zip_field = PARCEL_ZIP if PARCEL_ZIP in df.columns else None
    owner_zip_field = next((f for f in [OWNER_ZIP, "Zip", OWNER_ZIP_ALT] if f in df.columns), None)
    
    if parcel_addr_field and owner_addr_field:
        log(f"\n  Checking address matching for residential properties:")
        log(f"  Parcel address field: {parcel_addr_field}")
        log(f"  Owner address field: {owner_addr_field}")
        log(f"  Parcel ZIP field: {parcel_zip_field}")
        log(f"  Owner ZIP field: {owner_zip_field}")
        
        # Check address match - apply to ALL properties first
        df["AddressMatch"] = df.apply(
            lambda row: addresses_match(
                row.get(parcel_addr_field, ""), 
                row.get(owner_addr_field, ""),
                row.get(parcel_zip_field, "") if parcel_zip_field else None,
                row.get(owner_zip_field, "") if owner_zip_field else None
            ), axis=1
        )
        
        # Owner-occupied residential (address matches)
        df.loc[residential_mask & df["AddressMatch"], "OwnerType"] = "Owner_Occupied"
        
        # For non-owner-occupied residential, use distance bands and approximate county adjacency
        non_owner_res = residential_mask & ~df["AddressMatch"]
        
        # First check if we have OWNER_CATEGORY field to help identify adjacent owners
        if OWNER_CATEGORY in df.columns:
            log("\n  Using OWNER_CATEGORY and distance to approximate adjacent county owners...")
            
            # Adjacent county approximation:
            # - InZIP but different address = very local, could be adjacent
            # - InCounty = likely within same or adjacent county
            # - OutCounty/OutState but very close (< 20km) = likely adjacent county (e.g., Pottawattamie)
            
            # Same ZIP different address - these are very local
            same_zip_mask = (df[OWNER_CATEGORY] == "InZIP") & non_owner_res
            df.loc[same_zip_mask & (df[DIST] <= 5), "OwnerType"] = "Adjacent_County_Owner"
            
            # In County - likely same or adjacent county
            in_county_mask = (df[OWNER_CATEGORY] == "InCounty") & non_owner_res
            df.loc[in_county_mask & (df[DIST] <= 15), "OwnerType"] = "Adjacent_County_Owner"
            
            # Out of County/State but very close - captures Pottawattamie and other border areas
            out_area_mask = df[OWNER_CATEGORY].isin(["OutCounty", "OutState"]) & non_owner_res
            df.loc[out_area_mask & (df[DIST] <= 20), "OwnerType"] = "Adjacent_County_Owner"
            
            # For logging purposes, count different types
            adj_count = (df["OwnerType"] == "Adjacent_County_Owner").sum()
            if adj_count > 0:
                log(f"  Identified {adj_count:,} adjacent county owners using distance approximation")
        
        # Now assign distance-based categories for remaining non-owner-occupied
        remaining_mask = non_owner_res & (df["OwnerType"] == "Unknown")
        
        # Distance bands
        df.loc[remaining_mask & (df[DIST] <= 10), "OwnerType"] = "Near_Owner_0_10km"
        df.loc[remaining_mask & (df[DIST] > 10) & (df[DIST] <= 25), "OwnerType"] = "Near_Owner_10_25km"
        df.loc[remaining_mask & (df[DIST] > 25) & (df[DIST] <= 50), "OwnerType"] = "Near_Owner_25_50km"
        df.loc[remaining_mask & (df[DIST] > 50) & (df[DIST] <= 100), "OwnerType"] = "Regional_Owner_50_100km"
        df.loc[remaining_mask & (df[DIST] > 100) & (df[DIST] <= 200), "OwnerType"] = "Regional_Owner_100_200km"
        df.loc[remaining_mask & (df[DIST] > 200) & (df[DIST] <= 500), "OwnerType"] = "Distant_Owner_200_500km"
        df.loc[remaining_mask & (df[DIST] > 500), "OwnerType"] = "Distant_Owner_500km_plus"
        
        # Handle any remaining residential properties with missing distance
        still_unknown = residential_mask & (df["OwnerType"] == "Unknown")
        if still_unknown.any():
            df.loc[still_unknown, "OwnerType"] = "Near_Owner_0_10km"
            log(f"  Note: {still_unknown.sum()} residential properties had missing distance, assigned to Near_0_10km")
        
        # Handle any remaining residential properties with missing distance
        # Assign them to a nearby category based on available information
        still_unknown = non_owner_res & (df["OwnerType"] == "Unknown")
        if still_unknown.any():
            # If they have no distance, assign to near category
            df.loc[still_unknown, "OwnerType"] = "Near_Owner_0_10km"
            log(f"  Note: {still_unknown.sum()} non-owner residential properties had missing distance, assigned to Near_0_10km")
        
        # Log address matching statistics
        res_match_count = df[residential_mask]["AddressMatch"].sum()
        res_total = residential_mask.sum()
        log(f"\n  Residential address matching results:")
        log(f"  {res_match_count:,} owner-occupied (live at property)")
        log(f"  {res_total - res_match_count:,} non-owner-occupied")
        log(f"  Owner-occupancy rate: {res_match_count/res_total*100:.1f}%")
    else:
        # If address fields not available, use distance categories for all residential
        log("  ⚠ Address fields not available for matching, using distance categories only")
        
        # Check if we have OWNER_CATEGORY to approximate adjacent counties
        if OWNER_CATEGORY in df.columns:
            # Same logic as above but for all residential
            same_zip_mask = (df[OWNER_CATEGORY] == "InZIP") & residential_mask
            df.loc[same_zip_mask & (df[DIST] <= 5), "OwnerType"] = "Adjacent_County_Owner"
            
            in_county_mask = (df[OWNER_CATEGORY] == "InCounty") & residential_mask
            df.loc[in_county_mask & (df[DIST] <= 15), "OwnerType"] = "Adjacent_County_Owner"
            
            out_area_mask = df[OWNER_CATEGORY].isin(["OutCounty", "OutState"]) & residential_mask
            df.loc[out_area_mask & (df[DIST] <= 20), "OwnerType"] = "Adjacent_County_Owner"
        
        # Distance bands for remaining
        remaining_mask = residential_mask & (df["OwnerType"] == "Unknown")
        
        df.loc[remaining_mask & (df[DIST] <= 10), "OwnerType"] = "Near_Owner_0_10km"
        df.loc[remaining_mask & (df[DIST] > 10) & (df[DIST] <= 25), "OwnerType"] = "Near_Owner_10_25km"
        df.loc[remaining_mask & (df[DIST] > 25) & (df[DIST] <= 50), "OwnerType"] = "Near_Owner_25_50km"
        df.loc[remaining_mask & (df[DIST] > 50) & (df[DIST] <= 100), "OwnerType"] = "Regional_Owner_50_100km"
        df.loc[remaining_mask & (df[DIST] > 100) & (df[DIST] <= 200), "OwnerType"] = "Regional_Owner_100_200km"
        df.loc[remaining_mask & (df[DIST] > 200) & (df[DIST] <= 500), "OwnerType"] = "Distant_Owner_200_500km"
        df.loc[remaining_mask & (df[DIST] > 500), "OwnerType"] = "Distant_Owner_500km_plus"
    
    # Log owner type distribution
    log("\n▸ Owner type distribution:")
    owner_occupied_count = 0
    for ot, count in df["OwnerType"].value_counts().items():
        log(f"  {ot}: {count:,} ({count/len(df)*100:.1f}%)")
        out.append(["OwnerTypeCounts", ot, None, "count", int(count)])
        out.append(["OwnerTypeCounts", ot, None, "percent", float(count/len(df)*100)])
        if ot == "Owner_Occupied":
            owner_occupied_count = count
    
    # Log geographic breakdown for Adjacent County owners if available
    if "Adjacent_County_Owner" in df["OwnerType"].values and OWNER_CATEGORY in df.columns:
        adjacent_owners = df[df["OwnerType"] == "Adjacent_County_Owner"]
        
        log(f"\n  Adjacent County Owner breakdown by category:")
        
        # Break down by OWNER_CATEGORY
        in_zip = adjacent_owners[adjacent_owners[OWNER_CATEGORY] == "InZIP"]
        in_county = adjacent_owners[adjacent_owners[OWNER_CATEGORY] == "InCounty"]
        out_county = adjacent_owners[adjacent_owners[OWNER_CATEGORY] == "OutCounty"]
        out_state = adjacent_owners[adjacent_owners[OWNER_CATEGORY] == "OutState"]
        
        total_adj = len(adjacent_owners)
        if total_adj > 0:
            log(f"    Same ZIP (different address): {len(in_zip):,} ({len(in_zip)/total_adj*100:.1f}%)")
            log(f"    Same County area: {len(in_county):,} ({len(in_county)/total_adj*100:.1f}%)")
            log(f"    Different County (NE): {len(out_county):,} ({len(out_county)/total_adj*100:.1f}%)")
            log(f"    Out of State (likely IA): {len(out_state):,} ({len(out_state)/total_adj*100:.1f}%)")
            log(f"    Average distance: {adjacent_owners[DIST].mean():.1f}km")
    
    if owner_occupied_count == 0:
        log("\n  ⚠ No owner-occupied properties found. This could indicate:")
        log("    - Address fields don't match well between owner and property records")
        log("    - Different address formatting standards")
        log("    - Primarily non-resident owned area")
        out.append(["DataQuality", "AddressMatching", None, "owner_occupied_found", 0])
    
    # Process other variables
    df[STRUCT_VAL] = pd.to_numeric(df[STRUCT_VAL], errors="coerce")
    df["log_Total_Asse"] = np.log1p(df[STRUCT_VAL])
    
    df[YR_BUILT] = pd.to_numeric(df[YR_BUILT], errors="coerce")
    df["log_BuildYear"] = np.log1p(df[YR_BUILT])
    
    if PLOT_SIZE in df.columns:
        df["log_PlotSize"] = pd.to_numeric(df[PLOT_SIZE], errors="coerce").pipe(np.log1p)
    
    if INUND_F in df.columns:
        df[INUND_F] = pd.to_numeric(df[INUND_F], errors="coerce").fillna(0).astype(int)
    else:
        df[INUND_F] = 0  # Default to 0 if field doesn't exist
    
    # Summary statistics
    avg_dist = df[DIST].mean()
    med_dist = df[DIST].median()
    
    out.extend([
        ["Summary", "Sample_Size", None, "N", int(len(df))],
        ["Summary", "SFHA_Rate", None, "Percent", float(df["SFHA"].mean() * 100)],
        ["Summary", "Inundated_Rate", None, "Percent", float(df[INUND_F].mean() * 100 if INUND_F in df.columns else 0)],
        ["Summary", "Avg_Distance_km", None, "Mean", float(avg_dist if not pd.isna(avg_dist) else 0)],
        ["Summary", "Median_Distance_km", None, "Median", float(med_dist if not pd.isna(med_dist) else 0)],
        ["Methodology_Note", "Analysis_Type", None, "Text", "Non-tautological: Predicting flood risk from owner type"],
        ["Methodology_Note", "Reference_Category", None, "Text", f"Using '{df['OwnerType'].value_counts().index[0]}' as reference in models"],
        ["Methodology_Note", "Geographic_Categories", None, "Text", "Distance bands: Adjacent counties (approximated), 0-10km, 10-25km, 25-50km, 50-100km, 100-200km, 200-500km, 500km+"],
        ["Methodology_Note", "Adjacent_Counties", None, "Text", "Adjacent approximated by OWNER_CATEGORY + distance: InZIP<5km, InCounty<15km, Out<20km"],
    ])
    
    # Run improved analyses
    
    # 1. Flood risk by owner type
    owner_type_flood_analysis(df, out)
    
    # 2. Non-tautological regressions
    flood_risk_regressions(df, out)
    
    # 3. Multinomial model
    if "OwnerType" in df.columns:
        multinomial_ownership_model(df, out)
    
    # 4. Spatial analyses
    log("\n▸ Spatial analyses...")
    
    # Overall Moran's I
    try:
        with suppress_stdout():
            mi_sfha = Moran(df["SFHA"].values, W)
        out.extend([
            ["Moran_I", "SFHA", None, "I", float(mi_sfha.I)],
            ["Moran_I", "SFHA", None, "p_value", float(mi_sfha.p_sim)],
        ])
        log(f"  SFHA Moran's I = {mi_sfha.I:.3f}, p = {mi_sfha.p_sim:.3f}")
        
        if INUND_F in df.columns:
            with suppress_stdout():
                mi_inund = Moran(df[INUND_F].values, W)
            out.extend([
                ["Moran_I", INUND_F, None, "I", float(mi_inund.I)],
                ["Moran_I", INUND_F, None, "p_value", float(mi_inund.p_sim)],
            ])
            log(f"  Inundation Moran's I = {mi_inund.I:.3f}, p = {mi_inund.p_sim:.3f}")
    except Exception as e:
        log(f"⚠ Moran's I failed: {str(e)}")
    
    # Stratified Moran's I by owner type
    if "OwnerType" in df.columns:
        stratified_morans_i(df, W, "SFHA", "OwnerType", out)
        if INUND_F in df.columns:
            stratified_morans_i(df, W, INUND_F, "OwnerType", out)
    
    # Spatial regression models
    if "OwnerType" in df.columns:
        spatial_flood_models(df, W, "SFHA", "OwnerType", out)
        if INUND_F in df.columns and df[INUND_F].nunique() > 1:
            spatial_flood_models(df, W, INUND_F, "OwnerType", out)
    
    # Distance analysis among non-resident owners only
    # Exclude Commercial properties and owner-occupied
    df_nonres = df[~df["OwnerType"].isin(["Owner_Occupied", "Commercial", "Unknown"])]
    if len(df_nonres) > 100:
        log(f"\n▸ Distance analysis among {len(df_nonres)} non-resident owners:")
        
        # Correlation between distance and flood risk
        corr_dist_sfha = df_nonres[[DIST, "SFHA"]].corr().iloc[0, 1]
        corr_logdist_sfha = df_nonres[["log_dist", "SFHA"]].corr().iloc[0, 1]
        
        out.extend([
            ["NonResident_Distance_Analysis", "Correlation", None, "dist_SFHA", float(corr_dist_sfha)],
            ["NonResident_Distance_Analysis", "Correlation", None, "log_dist_SFHA", float(corr_logdist_sfha)],
            ["NonResident_Distance_Analysis", "Summary", None, "N", int(len(df_nonres))],
        ])
        
        log(f"  Correlation(distance, SFHA) = {corr_dist_sfha:.3f}")
        log(f"  Correlation(log_distance, SFHA) = {corr_logdist_sfha:.3f}")
        
        # Compare near vs distant owners
        # Near: Adjacent counties + 0-25km
        # Distant: 200km+
        near_types = ["Adjacent_County_Owner", "Near_Owner_0_10km", "Near_Owner_10_25km"]
        distant_types = ["Distant_Owner_200_500km", "Distant_Owner_500km_plus"]
        
        # Only include types that exist in the data
        near_types = [t for t in near_types if t in df_nonres["OwnerType"].values]
        distant_types = [t for t in distant_types if t in df_nonres["OwnerType"].values]
        
        if near_types and distant_types:
            near_owners = df_nonres[df_nonres["OwnerType"].isin(near_types)]
            distant_owners = df_nonres[df_nonres["OwnerType"].isin(distant_types)]
            
            if len(near_owners) > 0 and len(distant_owners) > 0:
                # Fisher's exact test for 2x2 table
                sfha_near = near_owners["SFHA"].sum()
                sfha_distant = distant_owners["SFHA"].sum()
                no_sfha_near = len(near_owners) - sfha_near
                no_sfha_distant = len(distant_owners) - sfha_distant
                
                if min(sfha_near, sfha_distant, no_sfha_near, no_sfha_distant) >= 0:
                    odds_ratio, p_fisher = stats.fisher_exact([[sfha_near, no_sfha_near], 
                                                               [sfha_distant, no_sfha_distant]])
                    
                    out.extend([
                        ["Near_vs_Distant_Owners", "SFHA", None, "odds_ratio", float(odds_ratio)],
                        ["Near_vs_Distant_Owners", "SFHA", None, "p_fisher", float(p_fisher)],
                        ["Near_vs_Distant_Owners", "Near", None, "SFHA_rate", float(near_owners["SFHA"].mean())],
                        ["Near_vs_Distant_Owners", "Distant", None, "SFHA_rate", float(distant_owners["SFHA"].mean())],
                        ["Near_vs_Distant_Owners", "Near", None, "N", int(len(near_owners))],
                        ["Near_vs_Distant_Owners", "Distant", None, "N", int(len(distant_owners))],
                    ])
                    
                    log(f"\n  Near vs Distant Owners:")
                    log(f"  Near SFHA rate (adjacent/0-25km): {near_owners['SFHA'].mean():.3f} (N={len(near_owners)})")
                    log(f"  Distant SFHA rate (200km+): {distant_owners['SFHA'].mean():.3f} (N={len(distant_owners)})")
                    log(f"  Odds ratio: {odds_ratio:.2f}, p = {p_fisher:.4f}")
        
        # Report SFHA rates by distance band
        log(f"\n  SFHA rates by distance band:")
        distance_categories = ["Adjacent_County_Owner", "Near_Owner_0_10km", "Near_Owner_10_25km", 
                              "Near_Owner_25_50km", "Regional_Owner_50_100km", "Regional_Owner_100_200km",
                              "Distant_Owner_200_500km", "Distant_Owner_500km_plus"]
        
        for cat in distance_categories:
            cat_data = df_nonres[df_nonres["OwnerType"] == cat]
            if len(cat_data) > 0:
                sfha_rate = cat_data["SFHA"].mean()
                avg_dist = cat_data[DIST].mean()
                out.extend([
                    [f"Distance_Band_{cat}", "SFHA_rate", None, "rate", sfha_rate],
                    [f"Distance_Band_{cat}", "Sample", None, "N", len(cat_data)],
                    [f"Distance_Band_{cat}", "Distance", None, "avg_km", avg_dist],
                ])
                
                # Shorten name for display
                display_name = cat.replace("_Owner", "").replace("Owner_", "").replace("_", " ")
                avg_dist_str = f"{avg_dist:.1f}" if not pd.isna(avg_dist) else "0.0"
                log(f"    {display_name}: SFHA={sfha_rate:.3f}, N={len(cat_data)}, avg_dist={avg_dist_str}km")
    
    return out

# ── 16 NN weights (row standardised) ----------------------------------
def build_weights(lonlat: np.ndarray) -> ps.weights.W:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="is an island")
        warnings.filterwarnings("ignore")
        with suppress_stdout():
            W = ps.weights.KNN(np.radians(lonlat), k=NEIGHBORS_K)
            W.transform = "R"
    
    if W.n_components > 1:
        log(f"⚠ Spatial weights has {W.n_components} disconnected components")
        log(f"  (Island warnings for individual parcels have been suppressed)")
    
    return W

# ── main driver ────────────────────────────────────────────────────────
def main():
    t0 = time.time()
    arcpy.env.overwriteOutput = True
    arcpy.env.outputCoordinateSystem = PARCELS_FC
    
    # Setup directories
    gdb_path = os.path.dirname(PARCELS_FC)
    proj_dir = (os.path.dirname(gdb_path) if gdb_path.lower().endswith(".gdb") 
                else os.path.dirname(PARCELS_FC))
    diag_dir = os.path.join(proj_dir, "diagnostics")
    os.makedirs(diag_dir, exist_ok=True)
    
    scratch_gdb = os.path.join(proj_dir, "scratch.gdb")
    if not arcpy.Exists(scratch_gdb):
        arcpy.management.CreateFileGDB(proj_dir, os.path.basename(scratch_gdb))
    
    try:
        # Check and process FIRM overlay if needed
        if firm_overlay_needed(PARCELS_FC):
            majority_tabulate_firm(PARCELS_FC, FIRM_FC, scratch_gdb)
        else:
            log("▸ FIRM overlay skipped – field and data seem present")
        
        # Process inundation flagging
        if arcpy.Exists(INUND_POLY):
            flag_inundated_parcels(PARCELS_FC, INUND_POLY, INUND_F, scratch_gdb)
        else:
            log(f"✖ Inundation polygon layer not found at: {INUND_POLY}")
            check_and_add_field(PARCELS_FC, INUND_F, "SHORT")
        
        # Load data
        log("\n▸ Loading data into DataFrame...")
        base_fields = [PID, DIST, ZONE_FIRM, LU_CODE, INUND_F, STRUCT_VAL,
                      YR_BUILT, PAR_LON, PAR_LAT, OWNER_INZIP, OWNER_CATEGORY]
        
        fields_to_try = base_fields + PARCEL_ADDR_OPTIONS + OWNER_ADDR_OPTIONS + \
                       [PARCEL_ZIP, OWNER_ZIP, OWNER_ZIP_ALT, "SHAPE_Area", "Shape_Area", "shape_area"]
        
        existing_fields = [f.name for f in arcpy.ListFields(PARCELS_FC)]
        final_fields = [f for f in fields_to_try if f in existing_fields]
        
        # Find shape area field
        shape_field_found = next((f for f in ["SHAPE_Area", "Shape_Area", "shape_area"] if f in existing_fields), None)
        if shape_field_found and PLOT_SIZE not in final_fields:
            final_fields.append(shape_field_found)
        
        rows = [dict(zip(final_fields, r)) for r in arcpy.da.SearchCursor(PARCELS_FC, final_fields)]
        df = pd.DataFrame(rows)
        
        if shape_field_found and shape_field_found != PLOT_SIZE:
            df[PLOT_SIZE] = df[shape_field_found]
        
        # Clean coordinates
        coord_cols = [PAR_LON, PAR_LAT]
        df[coord_cols] = df[coord_cols].apply(pd.to_numeric, errors='coerce')
        valid_coords = ~(df[coord_cols].isna().any(axis=1) | np.isinf(df[coord_cols].values).any(axis=1))
        
        if not valid_coords.all():
            log(f"⚠ Removing {(~valid_coords).sum()} parcels with invalid coordinates")
            df = df[valid_coords].reset_index(drop=True)
        
        # Build spatial weights
        log("\n▸ Building spatial weights matrix...")
        log("  (Note: Island warnings for disconnected parcels are suppressed)")
        W = build_weights(df[[PAR_LON, PAR_LAT]].values)
        
        # Run analysis
        rows_out = run_analysis(df, W)
        
        # Write results
        out_file = os.path.join(diag_dir, f"flood_risk_nontautological_{ts()}.csv")
        write_csv(out_file, ["Metric", "Group", "Sub", "Stat", "Value"], rows_out)
        
        log(f"\n✓ Results → {out_file}")
        log(f"✓ Finished in {(time.time() - t0) / 60:.1f} min")
        
    except arcpy.ExecuteError:
        log("✖ ArcGIS Geoprocessing Error:")
        log(arcpy.GetMessages(2))
        traceback.print_exc()
    except Exception:
        log("✖ Python Script Error:")
        traceback.print_exc()
    finally:
        arcpy.env.overwriteOutput = False

if __name__ == "__main__":
    try:
        assert arcpy.Exists(PARCELS_FC), f"Parcels layer not found at '{PARCELS_FC}'"
        assert arcpy.Exists(FIRM_FC), f"FIRM layer not found at '{FIRM_FC}'"
        assert arcpy.Exists(INUND_POLY), f"Inundation polygon not found at '{INUND_POLY}'"
        main()
    except AssertionError as e:
        log(f"✖ Input validation error: {e}")
    except Exception:
        log("✖ Script error:")
        traceback.print_exc()
