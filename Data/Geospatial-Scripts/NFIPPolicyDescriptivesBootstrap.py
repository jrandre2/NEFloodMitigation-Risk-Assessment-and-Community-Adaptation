#!/usr/bin/env python
"""
nfip_filter_diagnostics.py – Stand-Alone ArcPy Helper
=====================================================

Creates diagnostic tables that show:

* for **each NFIP policy**, how many candidate buildings remain after every
  filter in the baseline matching model; and
* how much variance those candidates exhibit when *N* bootstrap draws are made
  (default 1 000).

Outputs (CSV, timestamp-stamped in workspace parent folder)
-----------------------------------------------------------
1. NFIP_Filter_Diagnostics_<TS>.csv      – per-policy metrics
2. NFIP_Filter_Diagnostics_FZ_<TS>.csv   – flood-zone summary
3. NFIP_Filter_Diagnostics_ZIP_<TS>.csv  – ZIP-code summary
"""

from __future__ import annotations
import argparse, csv, datetime as _dt, json, os, sys, uuid
from pathlib import Path
from typing import Dict, List, Tuple, Sequence, Any

import numpy as np
import arcpy                                              # ArcPy must be available

# ────────── default project paths (identical to bootstrap) ───────────────────
WORKSPACE_DEFAULT = r"C:\Mac\Home\Documents\ArcGIS\Projects\NFIP Dodge County\NFIP Dodge County.gdb"
BUILDINGS_FC      = "Parcels_SpatialJoin"          # contains BldgID, BuildYear …
CLAIMS_TABLE      = "FEMA_Claims_Nebraska"
GT_TABLE          = "Inund_GT_OIDs"                # *not* needed here, kept for parity

# ────────── field names (identical to bootstrap) ─────────────────────────────
BLDG_ID   = "BldgID"
ZIP_FLD   = "ZIP"
FZ_FLD    = "FloodZone"
VAL_FLD   = "Total_Asse"
ELEV_FLD  = "ELEVATION"
PARCEL_FLD= "Parcel_ID"
BLDG_YEAR_TEMP = "BuildYear"

RESIDENTIAL_ZONE_FLD   = "Zone"
RESIDENTIAL_ZONE_VALUE = 1

POL_ID    = "OBJECTID"
POL_ZIP   = "reportedZipCode"
POL_FZ    = "floodZoneCurrent"
POL_BFE   = "baseFloodElevation"
POL_COST  = "buildingReplacementCost"
POL_ORIG_CONSTR_DATE = "originalConstructionDate"

# ────────── baseline filter parameters (match bootstrap “base case”) ─────────
USE_LARGEST_ON_PARCEL  = True
ELEV_TOLERANCE_FT      = 0.5            # ± 0.5 ft
VAL_TOLERANCE_PCT      = None           # value filter OFF in base case
USE_FLOOD_ZONE_MATCH   = True
YEAR_TOLERANCE_YRS     = None           # year filter OFF in base case

DEFAULT_N_ITERS        = 1000           # bootstrap draws per policy

# ────────── helpers identical to bootstrap (trimmed to essentials) ───────────
def ds(name: str, workspace: str) -> str:
    """Return full path to a dataset inside workspace, ensure it exists."""
    p = os.path.join(workspace, name)
    if not arcpy.Exists(p):  # type: ignore
        raise RuntimeError(f"Dataset '{name}' not found in {workspace}")
    return p

def parse_year_robustly(val: Any) -> int | None:
    """Attempt to coerce many year / date formats to an int."""
    if val in (None, "", " ", "<null>"): return None
    s = str(val).strip()
    # Try numeric directly
    try:
        f = float(s)
        if f.is_integer(): return int(f)
    except ValueError:
        pass
    # Try strptime formats
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y/%m/%d", "%Y-%m-%d",
                "%m/%d/%Y", "%d/%m/%Y", "%Y%m%d"):
        try:
            return _dt.datetime.strptime(s, fmt).year
        except ValueError:
            continue
    # Four-digit fallback
    if len(s) == 4 and s.isdigit():
        y = int(s)
        if 1700 <= y <= _dt.datetime.now().year + 1:
            return y
    return None

def stats(arr: Sequence[float]) -> Dict[str, float]:
    """Return mean, median, std, min, max, count; handles empty list."""
    if not arr:
        return dict(mean=np.nan, median=np.nan, std=np.nan,
                    min=np.nan, max=np.nan, count=0)
    a = np.asarray(arr, dtype=float)
    return dict(mean=a.mean(), median=np.median(a), std=a.std(ddof=0),
                min=a.min(), max=a.max(), count=a.size)

# ────────── data loaders (verbatim from bootstrap, pruned) ───────────────────
def load_buildings(workspace: str, largest_only: bool = True
                   ) -> Tuple[Dict[int, dict],
                              Dict[Tuple[str, str], List[int]],
                              Dict[str, List[int]]]:

    fields = [BLDG_ID, ZIP_FLD, FZ_FLD, VAL_FLD, ELEV_FLD,
              PARCEL_FLD, RESIDENTIAL_ZONE_FLD, BLDG_YEAR_TEMP]
    processed = []

    with arcpy.da.SearchCursor(ds(BUILDINGS_FC, workspace), fields) as cur:  # type: ignore
        for row in cur:
            rec = dict(zip(fields, row))
            if None in (rec[BLDG_ID], rec[ZIP_FLD], rec[FZ_FLD]):
                continue
            # residential filter
            try:
                if int(rec[RESIDENTIAL_ZONE_FLD]) != RESIDENTIAL_ZONE_VALUE:
                    continue
            except Exception:
                continue

            try:
                rec[BLDG_ID] = int(rec[BLDG_ID])
            except Exception:
                continue

            # force types / formatting
            rec[ZIP_FLD] = str(rec[ZIP_FLD]).zfill(5)
            rec[FZ_FLD]  = str(rec[FZ_FLD]).upper()
            rec[VAL_FLD] = float(rec[VAL_FLD]) if rec[VAL_FLD] not in (None, "", " ") else None
            rec[ELEV_FLD]= float(rec[ELEV_FLD]) if rec[ELEV_FLD] not in (None, "", " ") else None
            rec[BLDG_YEAR_TEMP] = parse_year_robustly(rec[BLDG_YEAR_TEMP])

            processed.append(rec)

    # largest-building-per-parcel
    if largest_only:
        best = {}
        for r in processed:
            pid = r[PARCEL_FLD]
            if pid is None:
                continue
            if pid not in best or (r[VAL_FLD] or 0) > (best[pid][VAL_FLD] or 0):
                best[pid] = r
        processed = list(best.values())

    # build indices
    b: Dict[int, dict] = {}
    grp_zipfz: Dict[Tuple[str, str], List[int]] = {}
    grp_zip: Dict[str, List[int]] = {}
    for r in processed:
        bid = r[BLDG_ID]
        b[bid] = r
        grp_zipfz.setdefault((r[ZIP_FLD], r[FZ_FLD]), []).append(bid)
        grp_zip.setdefault(r[ZIP_FLD], []).append(bid)
    return b, grp_zipfz, grp_zip

def load_claims(workspace: str) -> List[dict]:
    fields = [POL_ID, POL_ZIP, POL_FZ, POL_BFE, POL_COST, POL_ORIG_CONSTR_DATE]
    out = []
    with arcpy.da.SearchCursor(ds(CLAIMS_TABLE, workspace), fields) as cur:  # type: ignore
        for row in cur:
            cid, zip5, fz, bfe, cost, constr = row
            if None in (cid, zip5, fz):
                continue
            out.append(dict(
                ClaimID=int(cid),
                ZIP=str(zip5).zfill(5),
                FZ=str(fz).upper(),
                BFE=float(bfe) if bfe not in (None, "", " ") else None,
                COST=float(cost) if cost not in (None, "", " ") else None,
                ORIG_YEAR=parse_year_robustly(constr)
            ))
    return out

# ────────── core filtering helpers (baseline logic) ──────────────────────────
def filter_elev_val(bids: List[int], b: Dict[int, dict], claim: dict,
                    elev_tol: float | None, val_tol: float | None) -> List[int]:
    """Return bids that satisfy elevation and value tolerance."""
    out: List[int] = []
    for bid in bids:
        rec = b[bid]
        # elevation tolerance
        if elev_tol is not None and claim["BFE"] is not None and rec[ELEV_FLD] is not None:
            if abs(rec[ELEV_FLD] - claim["BFE"]) > elev_tol:
                continue
        # value tolerance (percentage)
        if val_tol is not None and claim["COST"] is not None and rec[VAL_FLD] is not None and rec[VAL_FLD] > 0:
            tol = val_tol / 100.0
            lo, hi = rec[VAL_FLD] * (1 - tol), rec[VAL_FLD] * (1 + tol)
            if not (lo <= claim["COST"] <= hi):
                continue
        elif val_tol is not None and claim["COST"] is not None and (rec[VAL_FLD] is None or rec[VAL_FLD] <= 0):
            continue
        out.append(bid)
    return out

def apply_year_filter(bids: List[int], b: Dict[int, dict], claim_year: int | None,
                      year_tol: int | None) -> List[int]:
    """Filter by absolute or ±5-year tolerance; return bids that pass."""
    if year_tol is None or claim_year is None:
        return bids
    out = []
    for bid in bids:
        yr = b[bid][BLDG_YEAR_TEMP]
        if yr is None:
            continue
        diff = abs(yr - claim_year)
        if (year_tol == 0 and diff == 0) or (year_tol == 5 and diff <= 5):
            out.append(bid)
    return out

# ────────── main diagnostics routine ─────────────────────────────────────────
def run_diagnostics(workspace: str, n_iters: int) -> None:
    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(workspace).parent

    print(f"Loading buildings/claims from: {workspace}")
    b, grp_zipfz, grp_zip = load_buildings(workspace, USE_LARGEST_ON_PARCEL)
    claims = load_claims(workspace)
    print(f"  Buildings in model      : {len(b):,}")
    print(f"  NFIP claims (policies)  : {len(claims):,}")

    policy_rows = []

    rng = np.random.default_rng(seed=42)   # deterministic

    for c in claims:
        # stage 1 – ZIP (+ FZ)
        initial = grp_zipfz[(c["ZIP"], c["FZ"])] if USE_FLOOD_ZONE_MATCH \
                  else grp_zip[c["ZIP"]]
        n_initial = len(initial)

        # stage 2 – elev / value tolerance
        after_ev = filter_elev_val(initial, b, c, ELEV_TOLERANCE_FT, VAL_TOLERANCE_PCT)
        n_after_ev = len(after_ev)

        # stage 3 – year tolerance
        after_year = apply_year_filter(after_ev, b, c["ORIG_YEAR"], YEAR_TOLERANCE_YRS)
        n_final = len(after_year)

        # bootstrap
        n_unique_sel = var_sel = top_share = 0
        if n_iters and n_final:
            draws = rng.choice(after_year, n_iters, replace=True)
            _, counts = np.unique(draws, return_counts=True)
            n_unique_sel = counts.size
            var_sel = counts.var(ddof=0)
            top_share = counts.max() / n_iters

        policy_rows.append(dict(
            ClaimID=c["ClaimID"],
            ZIP=c["ZIP"],
            FZ=c["FZ"],
            n_initial=n_initial,
            n_after_elev_val=n_after_ev,
            n_final=n_final,
            n_unique_sel=n_unique_sel,
            var_sel=round(var_sel, 3),
            top_share=round(top_share, 3)
        ))

    # ---------- write policy-level CSV ---------------------------------------
    policy_csv = out_dir / f"NFIP_Filter_Diagnostics_{ts}.csv"
    with policy_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=policy_rows[0].keys())
        writer.writeheader()
        writer.writerows(policy_rows)
    print("Wrote policy diagnostics →", policy_csv)

    # ---------- summaries -----------------------------------------------------
    def summarise(group_key: str, summary_path: Path):
        groups: Dict[str, List[float]] = {}
        for row in policy_rows:
            key = row[group_key]
            groups.setdefault(key, []).append(row["var_sel"])
        with summary_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([group_key, "mean", "median", "std", "min", "max", "count"])
            for k, arr in sorted(groups.items()):
                s = stats(arr)
                writer.writerow([k, *(round(s[m], 3) for m in ("mean", "median", "std", "min", "max")), int(s["count"])])

    fz_csv  = out_dir / f"NFIP_Filter_Diagnostics_FZ_{ts}.csv"
    zip_csv = out_dir / f"NFIP_Filter_Diagnostics_ZIP_{ts}.csv"
    summarise("FZ",  fz_csv)
    summarise("ZIP", zip_csv)
    print("Wrote flood-zone summary  →", fz_csv)
    print("Wrote ZIP-code summary    →", zip_csv)

# ────────── CLI / entry point ────────────────────────────────────────────────
def cli() -> None:
    parser = argparse.ArgumentParser(description="NFIP filter diagnostics")
    parser.add_argument("--workspace", type=str, default=None,
                        help="Path to an ArcGIS .gdb workspace "
                             "(defaults to NFIP_WORKSPACE env or preset)")
    parser.add_argument("--n_iters", type=int, default=DEFAULT_N_ITERS,
                        help="Bootstrap draws per policy (0 = skip bootstrapping)")
    args = parser.parse_args()

    ws = args.workspace or os.environ.get("NFIP_WORKSPACE") or WORKSPACE_DEFAULT
    if not arcpy.Exists(ws):  # type: ignore
        sys.exit(f"Workspace not found: {ws}")

    # set ArcPy env in case script is run outside Pro
    arcpy.env.workspace = ws  # type: ignore

    run_diagnostics(ws, max(0, int(args.n_iters)))

if __name__ == "__main__":
    cli()
