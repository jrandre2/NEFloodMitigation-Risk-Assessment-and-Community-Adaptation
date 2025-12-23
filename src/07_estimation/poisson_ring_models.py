#!/usr/bin/env python3
"""
Module: poisson_ring_models.py
Purpose: Estimate near-but-dry ring models using Poisson and Negative Binomial regression.

This module tests for demand substitution by estimating sales count models for
distance rings around SFHA and inundation boundaries. Positive ring×post coefficients
indicate increased sales activity outside the boundary after the flood.

Input Files
-----------
- data_work/panel_parcel_month.parquet

Output Files
------------
- data_work/tab_poisson_ring_models.csv
- data_work/tab_nb_ring_models.csv

Functions
---------
ensure_env : Verify virtual environment is activated
months_since_event : Convert year-month to event time
build_counts : Aggregate sales counts by ring and month
fit_poisson : Fit Poisson GLM with ring interactions
fit_negative_binomial : Fit Negative Binomial model with ring interactions
main : Execute ring model estimation
"""
from __future__ import annotations
import os, sys
from pathlib import Path
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import NegativeBinomial as NBDiscrete

PANEL = Path('data_work/panel_parcel_month.parquet')
OUT_DIR = Path('data_work')

def ensure_env():
    if not os.getenv('VIRTUAL_ENV') or not os.getenv('VIRTUAL_ENV').endswith('/.venv'):
        print('ERROR: Please activate project .venv (source .venv/bin/activate) before running.', file=sys.stderr)
        sys.exit(1)

def months_since_event(ym: str) -> int:
    y, m = ym.split('-')
    y = int(y); m = int(m)
    return (y - 2019) * 12 + (m - 3)

def build_counts(panel: pd.DataFrame, boundary: str) -> pd.DataFrame:
    # boundary in {'inund','sfha'}; build ring label including inside
    dist_col = f'signed_dist_{boundary}_m'
    inside_col = f'inside_{boundary}'
    d = panel.copy()
    d = d[np.isfinite(d[dist_col])]
    d = d[d[dist_col].abs() <= 300].copy()
    # ring label
    def label_row(row):
        if row[inside_col] == 1:
            return 'inside'
        x = row[dist_col]
        if x <= 250: return '0_250m'
        else: return '250_300m'
    d['ring'] = d.apply(label_row, axis=1)
    d['event_m'] = d['ym'].apply(months_since_event)
    d['post'] = (d['event_m'] >= 0).astype(int)
    # counts by ring×month
    counts = d.groupby(['ring','ym']).agg(count=('sold_this_month','sum'), n_parcels=('sold_this_month','size')).reset_index()
    counts['post'] = counts['ym'].apply(lambda s: 1 if months_since_event(s) >= 0 else 0)
    return counts

def fit_poisson(counts: pd.DataFrame, baseline: str='inside') -> pd.DataFrame:
    # Design: count ~ post + ring + ring×post + month FE, offset=log(n_parcels)
    df = counts.copy()
    # Month FE
    month_dummies = pd.get_dummies(df['ym'].astype('category'), prefix='m', drop_first=True)
    ring_dummies = pd.get_dummies(df['ring'].astype('category'), prefix='ring')
    # Set baseline by dropping its column if present
    base_col = f'ring_{baseline}'
    if base_col in ring_dummies.columns:
        ring_dummies = ring_dummies.drop(columns=[base_col])
    # Interactions ring×post
    X = pd.concat([ring_dummies, month_dummies], axis=1)
    X['post'] = df['post'].astype(float)
    for col in ring_dummies.columns:
        X[f'{col}:post'] = X[col] * X['post']
    X = sm.add_constant(X, has_constant='add')
    # Ensure numeric types
    X = X.fillna(0).astype(float)
    y = df['count'].fillna(0).astype(float)
    offset = np.log(df['n_parcels'].clip(lower=1)).astype(float)
    model = sm.GLM(y, X, family=sm.families.Poisson(), offset=offset)
    res = model.fit(cov_type='HC3')
    rows = []
    for col in X.columns:
        if col.endswith(':post') or col in ['post']:
            b = res.params.get(col, np.nan)
            se = res.bse.get(col, np.nan)
            lo, hi = (b - 1.96*se, b + 1.96*se) if np.isfinite(se) else (np.nan, np.nan)
            rr = float(np.exp(b)) if np.isfinite(b) else np.nan
            rr_lo = float(np.exp(lo)) if np.isfinite(lo) else np.nan
            rr_hi = float(np.exp(hi)) if np.isfinite(hi) else np.nan
            rows.append({'term': col, 'beta': b, 'se': se, 'rr': rr, 'rr_ci_lo': rr_lo, 'rr_ci_hi': rr_hi})
    return pd.DataFrame(rows)

def fit_negative_binomial(counts: pd.DataFrame, baseline: str='inside') -> pd.DataFrame:
    df = counts.copy()
    month_dummies = pd.get_dummies(df['ym'].astype('category'), prefix='m', drop_first=True)
    ring_dummies = pd.get_dummies(df['ring'].astype('category'), prefix='ring')
    base_col = f'ring_{baseline}'
    if base_col in ring_dummies.columns:
        ring_dummies = ring_dummies.drop(columns=[base_col])
    X = pd.concat([ring_dummies, month_dummies], axis=1)
    X['post'] = df['post'].astype(float)
    for col in ring_dummies.columns:
        X[f'{col}:post'] = X[col] * X['post']
    X = sm.add_constant(X, has_constant='add')
    X = X.fillna(0).astype(float)
    y = df['count'].fillna(0).astype(float)
    # Discrete NB with exposure
    exposure = df['n_parcels'].clip(lower=1).astype(float)
    try:
        nb = NBDiscrete(y, X, exposure=np.log(exposure))
        res = nb.fit(disp=False, method='newton', maxiter=100)
    except Exception as e:
        # Fallback: return empty
        return pd.DataFrame()
    rows = []
    for col in X.columns:
        if col.endswith(':post') or col in ['post']:
            b = res.params.get(col, np.nan)
            se = res.bse.get(col, np.nan)
            lo, hi = (b - 1.96*se, b + 1.96*se) if np.isfinite(se) else (np.nan, np.nan)
            rr = float(np.exp(b)) if np.isfinite(b) else np.nan
            rr_lo = float(np.exp(lo)) if np.isfinite(lo) else np.nan
            rr_hi = float(np.exp(hi)) if np.isfinite(hi) else np.nan
            rows.append({'term': col, 'beta': b, 'se': se, 'rr': rr, 'rr_ci_lo': rr_lo, 'rr_ci_hi': rr_hi})
    return pd.DataFrame(rows)

def main():
    """
    Execute ring model estimation.

    Fits Poisson and Negative Binomial models for both inundation and SFHA
    boundaries, estimating ring×post interaction effects. Writes results
    with rate ratios and confidence intervals.

    Raises
    ------
    SystemExit
        If panel file does not exist.
    """
    ensure_env()
    if not PANEL.exists():
        raise SystemExit(f'Missing {PANEL}')
    panel = pd.read_parquet(PANEL)
    outs_p, outs_nb = [], []
    for boundary in ['inund','sfha']:
        counts = build_counts(panel, boundary)
        resp = fit_poisson(counts, baseline='inside')
        if not resp.empty:
            resp.insert(0, 'boundary', boundary)
            outs_p.append(resp)
        resnb = fit_negative_binomial(counts, baseline='inside')
        if not resnb.empty:
            resnb.insert(0, 'boundary', boundary)
            outs_nb.append(resnb)
    if outs_p:
        outp = pd.concat(outs_p, ignore_index=True)
        pathp = OUT_DIR / 'tab_poisson_ring_models.csv'
        outp.to_csv(pathp, index=False)
        print('Wrote', pathp)
    if outs_nb:
        outnb = pd.concat(outs_nb, ignore_index=True)
        pathnb = OUT_DIR / 'tab_nb_ring_models.csv'
        outnb.to_csv(pathnb, index=False)
        print('Wrote', pathnb)

if __name__ == '__main__':
    main()
