#!/usr/bin/env python3
"""
Spatial model (Conley spatial-HAC Poisson) applied to pre/post March 2019 windows.

This mirrors the spatial model used in the Named Entity (owner-type) analysis:
  - Outcome: `in_sfha` (binary); family: Poisson (log link)
  - Covariates: owner dummies + log_acres + log_total_value [+ log_owner_distance if present]
  - Spatial covariance: Conley (triangular kernel) at user-specified km bandwidths

Optional: Neighborhood FE variant (heavy); disabled by default to keep runtime reasonable.

Inputs
- results/integration_run/sfr_regression_data.csv (prepared by prepare_regression_data.py)
- Optional FE metadata from results/integration_run/parcels_with_classification.csv

Outputs
- projects/flood_2019_sales_douglas/outputs/spatial_pre_post_conley.csv
- projects/flood_2019_sales_douglas/outputs/spatial_pre_post_conley.md
"""
from __future__ import annotations
import os, sys, argparse
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.genmod.families import Poisson
from scipy.spatial import cKDTree
from scipy.stats import norm


DEF_DATA = Path("results/integration_run/sfr_regression_data.csv")
CLASS_CSV = Path("results/integration_run/parcels_with_classification.csv")
OUT_DIR = Path("projects/flood_2019_sales_douglas/outputs")

OWNER_DUMMIES = ['owner_Corporation','owner_LLC','owner_Other','owner_Trust']
CORE_COVARS = ['log_acres','log_total_value']
DIST_VAR = 'log_owner_distance'
TARGET = 'in_sfha'


def ensure_env():
    if not os.getenv('VIRTUAL_ENV') or not os.getenv('VIRTUAL_ENV').endswith('/.venv'):
        print('ERROR: Please activate project .venv (source .venv/bin/activate) before running.', file=sys.stderr)
        sys.exit(1)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Conley spatial HAC Poisson on pre/post March 2019 windows")
    p.add_argument("--data", type=Path, default=DEF_DATA, help="Path to sfr_regression_data.csv")
    p.add_argument("--event-date", type=str, default="2019-03-14", help="Flood event date (YYYY-MM-DD)")
    p.add_argument("--window-days", type=int, default=180, help="Half-window days for pre/post windows")
    p.add_argument("--bandwidths-km", type=str, default="0.5,1.0,2.0", help="Comma-separated Conley bandwidths in km")
    p.add_argument("--sample-n", type=int, default=40000, help="Optional row cap per window (0 for full)")
    p.add_argument("--with-fe", action="store_true", help="Include Neighborhood FE dummies (heavy)")
    return p.parse_args()


def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"Missing data at {path}. Generate integration outputs first.")
    df = pd.read_csv(path)
    # Coerce date and ensure numeric for required columns
    df['sale_date'] = pd.to_datetime(df['sale_date'], errors='coerce')
    for c in OWNER_DUMMIES + CORE_COVARS + [TARGET, DIST_VAR, 'x_coord','y_coord']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    return df


def subset_pre_post(df: pd.DataFrame, event_date: pd.Timestamp, window_days: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    start = event_date - pd.Timedelta(days=window_days)
    end = event_date + pd.Timedelta(days=window_days)
    dwin = df[(df['sale_date'] >= start) & (df['sale_date'] <= end) & (df['has_valid_sale_date'] == 1)].copy()
    pre = dwin[dwin['sale_date'] < event_date].copy()
    post = dwin[dwin['sale_date'] >= event_date].copy()
    return pre, post


def estimate_unit_scale_km_per_unit(xy: np.ndarray) -> float:
    max_abs = np.max(np.abs(xy)) if xy.size else 0.0
    # Heuristic: State Plane feet typically ~ millions; else treat as meters
    return 0.0003048 if max_abs > 1e6 else 0.001


def triangular_kernel(d: np.ndarray, c: float) -> np.ndarray:
    w = 1.0 - (d / c)
    w[(d > c) | (w < 0)] = 0.0
    return w


def conley_cov(res, X: np.ndarray, y: np.ndarray, mu: np.ndarray, xy: np.ndarray, radius: float) -> np.ndarray:
    n, p = X.shape
    # Bread: (X' W X)^{-1} with W = diag(mu)
    XTWX = X.T @ (X * mu[:, None])
    bread = np.linalg.inv(XTWX)
    # Scores
    u = (y - mu)
    S = X * u[:, None]
    # Neighbors within radius
    tree = cKDTree(xy)
    neigh = tree.query_ball_tree(tree, r=radius)
    M = np.zeros((p, p), dtype=float)
    for i, js in enumerate(neigh):
        if not js:
            continue
        d = np.linalg.norm(xy[js] - xy[i], axis=1)
        w = triangular_kernel(d, radius)
        if np.all(w == 0):
            continue
        Sj = S[js]  # (k,p)
        wsj = (w[:, None] * Sj).sum(axis=0)  # (p,)
        M += np.outer(S[i], wsj)
    return bread @ M @ bread


def build_design_core(df: pd.DataFrame, with_distance: bool) -> tuple[np.ndarray, list[str]]:
    cols = OWNER_DUMMIES + CORE_COVARS
    if with_distance and DIST_VAR in df.columns and df[DIST_VAR].notna().any():
        cols = cols + [DIST_VAR]
    X = df[cols].astype(float).values
    X = sm.add_constant(X, has_constant='add')
    names = ['const'] + cols
    return X, names


def attach_neighborhood_fe(df: pd.DataFrame) -> pd.DataFrame:
    if not CLASS_CSV.exists():
        raise FileNotFoundError(CLASS_CSV)
    # Ensure a string Parcel_ID key exists in df
    if 'Parcel_ID' in df.columns:
        df = df.copy()
        df['Parcel_ID'] = df['Parcel_ID'].astype(str)
    elif 'parcel_id' in df.columns:
        df = df.copy()
        df['Parcel_ID'] = df['parcel_id'].astype(str)
    else:
        raise RuntimeError("Neither 'Parcel_ID' nor 'parcel_id' present in regression data")

    # Read minimal columns, robust to _x/_y suffixes
    head = pd.read_csv(CLASS_CSV, nrows=0).columns.tolist()
    use = [c for c in ['Parcel_ID','Neighborho_x','Neighborho_y'] if c in head]
    meta = pd.read_csv(CLASS_CSV, usecols=use)
    meta['Parcel_ID'] = meta['Parcel_ID'].astype(str)
    fe = meta['Neighborho_x'] if 'Neighborho_x' in meta.columns else None
    if fe is None or fe.isna().all():
        fe = meta['Neighborho_y'] if 'Neighborho_y' in meta.columns else None
    meta['Neighborho'] = fe
    out = df.merge(meta[['Parcel_ID','Neighborho']], on='Parcel_ID', how='left')
    # Drop singleton FE groups
    vc = out['Neighborho'].value_counts()
    keep = vc[vc >= 2].index
    return out[out['Neighborho'].isin(keep)].copy()


def build_design_with_fe(df: pd.DataFrame, with_distance: bool) -> tuple[np.ndarray, list[str]]:
    cols = OWNER_DUMMIES + CORE_COVARS
    if with_distance and DIST_VAR in df.columns and df[DIST_VAR].notna().any():
        cols = cols + [DIST_VAR]
    Xblk = df[cols].astype('float32')
    dummies = pd.get_dummies(df['Neighborho'].astype('category'), prefix='Neighborho', drop_first=True).astype('float32')
    X = pd.concat([Xblk, dummies], axis=1)
    X = sm.add_constant(X, has_constant='add')
    names = ['const'] + cols + list(dummies.columns)
    return X.values.astype('float64', copy=False), names


def fit_poisson(y: np.ndarray, X: np.ndarray):
    return sm.GLM(y, X, family=Poisson()).fit()


def summarize(res, cov: np.ndarray, names: list[str], window_label: str, model_label: str, bw_km: float) -> pd.DataFrame:
    b = res.params
    se = np.sqrt(np.diag(cov))
    rr = np.exp(b)
    lo = np.exp(b - 1.96*se)
    hi = np.exp(b + 1.96*se)
    rows = []
    for term in OWNER_DUMMIES:
        if term in names:
            idx = names.index(term)
            rows.append({
                'window': window_label,
                'model': model_label,
                'bandwidth_km': bw_km,
                'term': term.replace('owner_',''),
                'beta': float(b[idx]),
                'rr_conley': float(rr[idx]),
                'ci_low_rr_conley': float(lo[idx]),
                'ci_high_rr_conley': float(hi[idx])
            })
    return pd.DataFrame(rows)


def run_for_subset(label: str, df: pd.DataFrame, bws_km: list[float], with_fe: bool, sample_n: int) -> pd.DataFrame:
    # Clean and sample
    need = OWNER_DUMMIES + CORE_COVARS + [TARGET, 'x_coord','y_coord']
    df = df.dropna(subset=need).copy()
    if sample_n and len(df) > sample_n:
        df = df.sample(sample_n, random_state=42).copy()
    # Coordinates
    xy = df[['x_coord','y_coord']].astype(float).values
    km_per_unit = estimate_unit_scale_km_per_unit(xy)
    y = df[TARGET].astype(float).values

    # CORE MODELS (M1_core + M3_core_distance)
    out_rows = []
    for with_distance in (False, True):
        X, names = build_design_core(df, with_distance=with_distance)
        model_name = 'M1_core' if not with_distance else 'M3_core_distance'
        res = fit_poisson(y, X)
        mu = res.mu
        for bw in bws_km:
            cov = conley_cov(res, X, y, mu, xy, radius=bw / km_per_unit)
            out_rows.append(summarize(res, cov, names, label, model_name, bw))

    # FE MODELS (optional; heavy)
    if with_fe:
        df_fe = attach_neighborhood_fe(df)
        if not df_fe.empty:
            xyf = df_fe[['x_coord','y_coord']].astype(float).values
            kmu_f = estimate_unit_scale_km_per_unit(xyf)
            y_f = df_fe[TARGET].astype(float).values
            for with_distance in (False, True):
                Xf, namesf = build_design_with_fe(df_fe, with_distance=with_distance)
                model_name_f = 'M2_core_FE' if not with_distance else 'M4_core_distance_FE'
                resf = fit_poisson(y_f, Xf)
                muf = resf.mu
                for bw in bws_km:
                    covf = conley_cov(resf, Xf, y_f, muf, xyf, radius=bw / kmu_f)
                    out_rows.append(summarize(resf, covf, namesf, label, model_name_f, bw))

    return pd.concat(out_rows, ignore_index=True) if out_rows else pd.DataFrame()


def main():
    ensure_env()
    args = parse_args()
    event_date = pd.Timestamp(args.event_date)
    bws_km = [float(x.strip()) for x in args.bandwidths_km.split(',') if x.strip()]

    df = load_data(args.data)
    pre, post = subset_pre_post(df, event_date, args.window_days)

    if pre.empty or post.empty:
        print(f"WARNING: One of the windows is empty. pre={len(pre)}, post={len(post)}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    res_pre = run_for_subset('pre', pre, bws_km, args.with_fe, args.sample_n)
    res_post = run_for_subset('post', post, bws_km, args.with_fe, args.sample_n)

    all_res = pd.concat([r for r in [res_pre, res_post] if r is not None and not r.empty], ignore_index=True)
    out_csv = OUT_DIR / 'spatial_pre_post_conley.csv'
    all_res.to_csv(out_csv, index=False)

    # Build markdown summary
    lines = []
    lines.append('# Spatial (Conley) Pre/Post Flood Analysis')
    lines.append('Spec: Poisson GLM on in_sfha with owner-type dummies + core covariates; Conley spatial HAC SEs.')
    lines.append(f"Event date: {event_date.date()} | Window ±{args.window_days} days | Bandwidths (km): {', '.join(str(b) for b in bws_km)}")
    lines.append('')
    def make_rr_ci_table(sub: pd.DataFrame) -> pd.DataFrame:
        # Expect columns: ['term','bandwidth_km','window','rr_conley','ci_low_rr_conley','ci_high_rr_conley']
        vals = ['rr_conley','ci_low_rr_conley','ci_high_rr_conley']
        piv = sub.pivot_table(index=['term','bandwidth_km'], columns='window', values=vals, aggfunc='first')
        # Flatten columns to e.g., pre_rr, pre_ci_lo, pre_ci_hi, post_rr, post_ci_lo, post_ci_hi
        flat_cols = []
        for win in ['pre','post']:
            if (('rr_conley', win) in piv.columns):
                flat_cols.append(('rr_conley', win))
                flat_cols.append(('ci_low_rr_conley', win))
                flat_cols.append(('ci_high_rr_conley', win))
        piv = piv.reindex(columns=flat_cols)
        piv.columns = [f"{w}_rr" if v=='rr_conley' else (f"{w}_ci_lo" if v=='ci_low_rr_conley' else f"{w}_ci_hi") for v,w in piv.columns]
        # Order rows by term then bandwidth
        piv = piv.sort_index(level=[0,1])
        return piv

    if all_res.empty:
        lines.append('No results (empty windows or insufficient data).')
    else:
        # Pivot RRs for quick comparison (core models only for brevity)
        core = all_res[all_res['model'].isin(['M1_core','M3_core_distance'])].copy()
        if not core.empty:
            for m in ['M1_core','M3_core_distance']:
                sub = core[core['model'] == m]
                lines.append(f'## {m} risk ratios (Conley RR with 95% CI)')
                piv = make_rr_ci_table(sub)
                try:
                    lines.append(piv.to_markdown(floatfmt='.3f'))
                except Exception:
                    lines.append(str(piv))
                lines.append('')
        # FE panels
        fe = all_res[all_res['model'].isin(['M2_core_FE','M4_core_distance_FE'])].copy()
        if not fe.empty:
            for m in ['M2_core_FE','M4_core_distance_FE']:
                sub = fe[fe['model'] == m]
                lines.append(f'## {m} risk ratios (Conley RR with 95% CI, Neighborhood FE)')
                piv = make_rr_ci_table(sub)
                try:
                    lines.append(piv.to_markdown(floatfmt='.3f'))
                except Exception:
                    lines.append(str(piv))
                lines.append('')
    (OUT_DIR / 'spatial_pre_post_conley.md').write_text("\n".join(lines))

    print('Wrote', out_csv, 'and', OUT_DIR / 'spatial_pre_post_conley.md')


if __name__ == "__main__":
    main()
