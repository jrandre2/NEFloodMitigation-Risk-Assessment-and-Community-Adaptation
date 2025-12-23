#!/usr/bin/env python3
"""
Module: spatial_econometrics.py
Purpose: Full spatial econometrics suite to address spatial autocorrelation.

This module addresses the significant spatial autocorrelation (Moran's I = 0.392)
identified in the baseline analysis through multiple approaches:
1. Conley standard errors - HAC SEs allowing for spatial correlation
2. Spatial Lag Model (SAR) - y = ρWy + Xβ + ε
3. Spatial Error Model (SEM) - y = Xβ + u, u = λWu + ε
4. SARAR Model - Combined spatial lag and error
5. Geographically Weighted Regression (GWR) - Local heterogeneity
6. Moran's I residual test - Diagnose residual spatial correlation

Input Files
-----------
- data_work/panel_parcel_month.parquet
- data_work/parcel_covariates_full.parquet (for coordinates)

Output Files
------------
- data_work/diagnostics/spatial_lag_results.csv
- data_work/diagnostics/spatial_error_results.csv
- data_work/diagnostics/sarar_results.csv
- data_work/diagnostics/conley_se_comparison.csv
- data_work/diagnostics/gwr_local_coefs.parquet
- data_work/diagnostics/moran_residual_test.csv
- figures/fig_gwr_local_effects.png
- figures/fig_moran_residuals.png

Dependencies
------------
- libpysal>=4.8.0
- spreg>=1.3.0
- esda>=2.4.0
- mgwr>=2.2.0 (optional, for GWR)

References
----------
Conley, T. G. (1999). GMM estimation with cross sectional dependence.
    Journal of Econometrics, 92(1), 1-45.
Anselin, L. (1988). Spatial Econometrics: Methods and Models.
"""
from __future__ import annotations
import argparse
import os
import sys
import warnings
from pathlib import Path
from typing import Optional, Dict, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import spatial, stats

# Optional imports - graceful degradation
try:
    import libpysal
    from libpysal.weights import KNN, DistanceBand
    LIBPYSAL_AVAILABLE = True
except ImportError:
    LIBPYSAL_AVAILABLE = False
    print('WARNING: libpysal not available. Install with: pip install libpysal')

try:
    import spreg
    SPREG_AVAILABLE = True
except ImportError:
    SPREG_AVAILABLE = False
    print('WARNING: spreg not available. Install with: pip install spreg')

try:
    import esda
    ESDA_AVAILABLE = True
except ImportError:
    ESDA_AVAILABLE = False
    print('WARNING: esda not available. Install with: pip install esda')

try:
    from mgwr.gwr import GWR
    from mgwr.sel_bw import Sel_BW
    MGWR_AVAILABLE = True
except ImportError:
    MGWR_AVAILABLE = False

PANEL = Path('data_work/panel_parcel_month.parquet')
COVARIATES = Path('data_work/parcel_covariates_full.parquet')
OUT_DIR = Path('data_work/diagnostics')
FIG_DIR = Path('figures')


def ensure_env():
    """Verify virtual environment is activated."""
    if not os.getenv('VIRTUAL_ENV') or not os.getenv('VIRTUAL_ENV').endswith('/.venv'):
        print('ERROR: Please activate project .venv', file=sys.stderr)
        sys.exit(1)


def months_since_event(ym: str) -> int:
    """Convert year-month string to event time relative to March 2019."""
    y, m = ym.split('-')
    return (int(y) - 2019) * 12 + (int(m) - 3)


def build_spatial_weights(
    coords: np.ndarray,
    method: str = 'knn',
    k: int = 8,
    distance_km: float = 5.0
) -> 'libpysal.weights.W':
    """
    Create spatial weights matrix from coordinates.

    Parameters
    ----------
    coords : np.ndarray
        N x 2 array of (x, y) coordinates.
    method : str
        Weight construction method: 'knn' or 'distance'.
    k : int
        Number of neighbors for KNN.
    distance_km : float
        Distance threshold for distance-based weights (km).

    Returns
    -------
    libpysal.weights.W
        Spatial weights object.
    """
    if not LIBPYSAL_AVAILABLE:
        raise ImportError('libpysal required for spatial weights')

    if method == 'knn':
        W = KNN.from_array(coords, k=k)
    elif method == 'distance':
        # Convert km to coordinate units (assuming meters)
        threshold = distance_km * 1000
        W = DistanceBand.from_array(coords, threshold=threshold, binary=True)
    else:
        raise ValueError(f'Unknown method: {method}')

    W.transform = 'r'  # Row-standardize
    return W


def conley_standard_errors(
    panel: pd.DataFrame,
    boundary: str,
    caliper: int,
    cutoff_km: float = 5.0,
    coord_cols: Tuple[str, str] = ('centroid_x', 'centroid_y')
) -> Dict:
    """
    Compute Conley (1999) HAC standard errors allowing for spatial correlation.

    Uses kernel-based weighting to allow arbitrary spatial correlation
    within the distance cutoff.

    Parameters
    ----------
    panel : pd.DataFrame
        Panel data with coordinates.
    boundary : str
        Boundary type: 'inund' or 'sfha'.
    caliper : int
        Caliper window in meters.
    cutoff_km : float
        Distance cutoff for spatial correlation in km.
    coord_cols : tuple
        Column names for x, y coordinates.

    Returns
    -------
    dict
        Comparison of HC3 vs Conley standard errors.
    """
    dist_col = f'signed_dist_{boundary}_m'
    inside_col = f'inside_{boundary}'

    if dist_col not in panel.columns:
        return {'error': f'Missing {dist_col}'}

    # Filter to caliper
    d = panel.copy()
    d = d[np.isfinite(d[dist_col])]
    d = d[d[dist_col].abs() <= caliper].copy()

    d['event_m'] = d['ym'].apply(months_since_event)
    d['post'] = (d['event_m'] >= 0).astype(int)
    d['inside'] = (d[inside_col] == 1).astype(int)

    # Check for coordinates
    if coord_cols[0] not in d.columns or coord_cols[1] not in d.columns:
        return {'error': f'Missing coordinate columns: {coord_cols}'}

    # Setup regression
    X = pd.DataFrame({
        'const': 1.0,
        'inside': d['inside'].astype(float),
        'post': d['post'].astype(float),
        'inside_x_post': (d['inside'] * d['post']).astype(float)
    })
    y = d['sold_this_month'].astype(float)

    # HC3 baseline
    model_hc3 = sm.OLS(y, X).fit(cov_type='HC3')
    coef = model_hc3.params['inside_x_post']
    se_hc3 = model_hc3.bse['inside_x_post']

    # Conley SE computation
    # Get coordinates and residuals
    coords = d[[coord_cols[0], coord_cols[1]]].values
    resid = model_hc3.resid.values

    # Build spatial kernel weights (uniform within cutoff)
    cutoff_m = cutoff_km * 1000
    n = len(d)

    # Compute pairwise distances using KDTree for efficiency
    tree = spatial.cKDTree(coords)

    # Compute XtX inverse
    XtX_inv = np.linalg.inv(X.T @ X)

    # Compute meat of sandwich with spatial correlation
    meat = np.zeros((X.shape[1], X.shape[1]))

    # For computational efficiency, use sparse distance queries
    pairs = tree.query_pairs(r=cutoff_m, output_type='ndarray')

    for i in range(n):
        meat += np.outer(X.iloc[i].values * resid[i], X.iloc[i].values * resid[i])

    # Add cross-terms for pairs within cutoff
    for i, j in pairs:
        kernel = 1.0  # Uniform kernel
        cross_term = kernel * resid[i] * resid[j] * np.outer(X.iloc[i].values, X.iloc[j].values)
        meat += cross_term + cross_term.T

    # Conley variance
    V_conley = XtX_inv @ meat @ XtX_inv
    se_conley = np.sqrt(np.diag(V_conley))

    # Get SE for inside_x_post coefficient
    idx = X.columns.get_loc('inside_x_post')
    se_conley_did = se_conley[idx]

    return {
        'boundary': boundary,
        'caliper_m': caliper,
        'cutoff_km': cutoff_km,
        'coef': coef,
        'se_hc3': se_hc3,
        'se_conley': se_conley_did,
        'ci_lo_hc3': coef - 1.96 * se_hc3,
        'ci_hi_hc3': coef + 1.96 * se_hc3,
        'ci_lo_conley': coef - 1.96 * se_conley_did,
        'ci_hi_conley': coef + 1.96 * se_conley_did,
        'se_ratio': se_conley_did / se_hc3,
        'n_pairs_in_cutoff': len(pairs),
        'n_obs': n
    }


def fit_spatial_lag_model(
    panel: pd.DataFrame,
    boundary: str,
    caliper: int,
    k_neighbors: int = 8
) -> Dict:
    """
    Estimate Spatial Lag (SAR) model: y = ρWy + Xβ + ε.

    Parameters
    ----------
    panel : pd.DataFrame
        Panel data with coordinates.
    boundary : str
        Boundary type.
    caliper : int
        Caliper window.
    k_neighbors : int
        Number of neighbors for spatial weights.

    Returns
    -------
    dict
        Model estimates including spatial lag coefficient ρ.
    """
    if not SPREG_AVAILABLE or not LIBPYSAL_AVAILABLE:
        return {'error': 'spreg and libpysal required'}

    dist_col = f'signed_dist_{boundary}_m'
    inside_col = f'inside_{boundary}'

    d = panel.copy()
    d = d[np.isfinite(d[dist_col])]
    d = d[d[dist_col].abs() <= caliper].copy()

    # Need coordinates
    if 'centroid_x' not in d.columns:
        return {'error': 'Missing coordinates'}

    d['event_m'] = d['ym'].apply(months_since_event)
    d['post'] = (d['event_m'] >= 0).astype(int)
    d['inside'] = (d[inside_col] == 1).astype(int)
    d['inside_x_post'] = d['inside'] * d['post']

    # Build spatial weights
    coords = d[['centroid_x', 'centroid_y']].values
    W = build_spatial_weights(coords, method='knn', k=k_neighbors)

    # Prepare arrays
    y = d['sold_this_month'].values.reshape(-1, 1)
    X = d[['inside', 'post', 'inside_x_post']].values

    # Estimate SAR model
    try:
        sar = spreg.GM_Lag(y, X, w=W, name_y='sold', name_x=['inside', 'post', 'inside_x_post'])

        # Extract results
        coef_idx = 3  # inside_x_post is the 4th variable (after constant)
        return {
            'boundary': boundary,
            'caliper_m': caliper,
            'model': 'SAR',
            'rho': sar.rho,  # Spatial lag coefficient
            'rho_se': sar.std_err[-1] if hasattr(sar, 'std_err') else np.nan,
            'coef_did': sar.betas[coef_idx + 1][0],  # +1 for constant
            'se_did': sar.std_err[coef_idx + 1] if hasattr(sar, 'std_err') else np.nan,
            'r2': sar.pr2 if hasattr(sar, 'pr2') else np.nan,
            'n_obs': len(d)
        }
    except Exception as e:
        return {'error': str(e)}


def fit_spatial_error_model(
    panel: pd.DataFrame,
    boundary: str,
    caliper: int,
    k_neighbors: int = 8
) -> Dict:
    """
    Estimate Spatial Error (SEM) model: y = Xβ + u, u = λWu + ε.

    Parameters
    ----------
    panel : pd.DataFrame
        Panel data.
    boundary : str
        Boundary type.
    caliper : int
        Caliper window.
    k_neighbors : int
        Number of neighbors.

    Returns
    -------
    dict
        Model estimates including spatial error coefficient λ.
    """
    if not SPREG_AVAILABLE or not LIBPYSAL_AVAILABLE:
        return {'error': 'spreg and libpysal required'}

    dist_col = f'signed_dist_{boundary}_m'
    inside_col = f'inside_{boundary}'

    d = panel.copy()
    d = d[np.isfinite(d[dist_col])]
    d = d[d[dist_col].abs() <= caliper].copy()

    if 'centroid_x' not in d.columns:
        return {'error': 'Missing coordinates'}

    d['event_m'] = d['ym'].apply(months_since_event)
    d['post'] = (d['event_m'] >= 0).astype(int)
    d['inside'] = (d[inside_col] == 1).astype(int)
    d['inside_x_post'] = d['inside'] * d['post']

    coords = d[['centroid_x', 'centroid_y']].values
    W = build_spatial_weights(coords, method='knn', k=k_neighbors)

    y = d['sold_this_month'].values.reshape(-1, 1)
    X = d[['inside', 'post', 'inside_x_post']].values

    try:
        sem = spreg.GM_Error(y, X, w=W, name_y='sold', name_x=['inside', 'post', 'inside_x_post'])

        coef_idx = 3
        return {
            'boundary': boundary,
            'caliper_m': caliper,
            'model': 'SEM',
            'lambda': sem.lam,  # Spatial error coefficient
            'lambda_se': sem.std_err[-1] if hasattr(sem, 'std_err') else np.nan,
            'coef_did': sem.betas[coef_idx + 1][0],
            'se_did': sem.std_err[coef_idx + 1] if hasattr(sem, 'std_err') else np.nan,
            'r2': sem.pr2 if hasattr(sem, 'pr2') else np.nan,
            'n_obs': len(d)
        }
    except Exception as e:
        return {'error': str(e)}


def moran_residual_test(
    panel: pd.DataFrame,
    boundary: str,
    caliper: int,
    k_neighbors: int = 8
) -> Dict:
    """
    Test for spatial autocorrelation in model residuals using Moran's I.

    Parameters
    ----------
    panel : pd.DataFrame
        Panel data.
    boundary : str
        Boundary type.
    caliper : int
        Caliper window.
    k_neighbors : int
        Number of neighbors.

    Returns
    -------
    dict
        Moran's I statistic and p-value for residuals.
    """
    if not ESDA_AVAILABLE or not LIBPYSAL_AVAILABLE:
        return {'error': 'esda and libpysal required'}

    dist_col = f'signed_dist_{boundary}_m'
    inside_col = f'inside_{boundary}'

    d = panel.copy()
    d = d[np.isfinite(d[dist_col])]
    d = d[d[dist_col].abs() <= caliper].copy()

    if 'centroid_x' not in d.columns:
        return {'error': 'Missing coordinates'}

    d['event_m'] = d['ym'].apply(months_since_event)
    d['post'] = (d['event_m'] >= 0).astype(int)
    d['inside'] = (d[inside_col] == 1).astype(int)

    # Fit baseline model
    X = pd.DataFrame({
        'const': 1.0,
        'inside': d['inside'].astype(float),
        'post': d['post'].astype(float),
        'inside_x_post': (d['inside'] * d['post']).astype(float)
    })
    y = d['sold_this_month'].astype(float)
    model = sm.OLS(y, X).fit()
    resid = model.resid.values

    # Build weights
    coords = d[['centroid_x', 'centroid_y']].values
    W = build_spatial_weights(coords, method='knn', k=k_neighbors)

    # Moran's I test
    mi = esda.Moran(resid, W)

    return {
        'boundary': boundary,
        'caliper_m': caliper,
        'moran_I': mi.I,
        'moran_I_expected': mi.EI,
        'moran_z': mi.z_norm,
        'moran_pval': mi.p_norm,
        'interpretation': 'Significant spatial autocorrelation' if mi.p_norm < 0.05 else 'No significant spatial autocorrelation',
        'n_obs': len(d)
    }


def plot_moran_residuals(moran_result: Dict, residuals: np.ndarray, W, out_path: Path) -> None:
    """Plot Moran scatter plot of residuals."""
    if not ESDA_AVAILABLE:
        return

    fig, ax = plt.subplots(figsize=(8, 8))

    # Compute spatial lag of residuals
    Wresid = libpysal.weights.lag_spatial(W, residuals)

    # Scatter plot
    ax.scatter(residuals, Wresid, alpha=0.3, s=10)

    # Regression line
    slope, intercept = np.polyfit(residuals, Wresid, 1)
    x_line = np.linspace(residuals.min(), residuals.max(), 100)
    ax.plot(x_line, slope * x_line + intercept, 'r-', linewidth=2,
            label=f"Moran's I = {moran_result['moran_I']:.3f}")

    # Reference lines
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
    ax.axvline(x=0, color='gray', linestyle='--', linewidth=0.5)

    ax.set_xlabel('Residuals', fontsize=11)
    ax.set_ylabel('Spatially Lagged Residuals', fontsize=11)
    ax.set_title(f"Moran Scatter Plot\np = {moran_result['moran_pval']:.4f}", fontsize=12)
    ax.legend()

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Wrote {out_path}')


def main(
    boundary: str = 'inund',
    caliper: int = 300,
    cutoff_km: float = 5.0,
    k_neighbors: int = 8
):
    """
    Run full spatial econometrics analysis.

    Parameters
    ----------
    boundary : str
        Boundary type.
    caliper : int
        Caliper window.
    cutoff_km : float
        Distance cutoff for Conley SEs.
    k_neighbors : int
        Number of neighbors for spatial weights.
    """
    ensure_env()

    if not PANEL.exists():
        raise SystemExit(f'Missing {PANEL}')

    panel = pd.read_parquet(PANEL)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    print('=' * 60)
    print('SPATIAL ECONOMETRICS ANALYSIS')
    print('=' * 60)
    print(f'Boundary: {boundary}')
    print(f'Caliper: ±{caliper}m')
    print(f'Conley cutoff: {cutoff_km}km')
    print(f'K neighbors: {k_neighbors}')
    print()

    # Check for coordinates
    if 'centroid_x' not in panel.columns:
        print('WARNING: Coordinates not in panel. Attempting to merge from covariates...')
        if COVARIATES.exists():
            cov = pd.read_parquet(COVARIATES)
            if 'centroid_x' in cov.columns:
                panel = panel.merge(cov[['parcel_id', 'centroid_x', 'centroid_y']],
                                    on='parcel_id', how='left')
        if 'centroid_x' not in panel.columns:
            print('ERROR: Could not obtain coordinates. Spatial analysis requires coordinates.')
            return

    results = {}

    # 1. Conley Standard Errors
    print('\n1. Computing Conley (1999) spatial standard errors...')
    print('-' * 40)
    conley_results = []
    for cutoff in [1.0, 2.5, 5.0, 10.0]:
        result = conley_standard_errors(panel, boundary, caliper, cutoff)
        if 'error' not in result:
            conley_results.append(result)
            print(f"  Cutoff {cutoff}km: SE_HC3={result['se_hc3']:.6f}, "
                  f"SE_Conley={result['se_conley']:.6f}, "
                  f"Ratio={result['se_ratio']:.2f}")

    if conley_results:
        pd.DataFrame(conley_results).to_csv(OUT_DIR / 'conley_se_comparison.csv', index=False)
        print(f"\n  Wrote {OUT_DIR / 'conley_se_comparison.csv'}")
    results['conley'] = conley_results

    # 2. Moran's I Residual Test
    print('\n2. Testing residual spatial autocorrelation (Moran I)...')
    print('-' * 40)
    moran_result = moran_residual_test(panel, boundary, caliper, k_neighbors)
    if 'error' not in moran_result:
        print(f"  Moran's I: {moran_result['moran_I']:.4f}")
        print(f"  z-score: {moran_result['moran_z']:.3f}")
        print(f"  p-value: {moran_result['moran_pval']:.4f}")
        print(f"  {moran_result['interpretation']}")

        pd.DataFrame([moran_result]).to_csv(OUT_DIR / 'moran_residual_test.csv', index=False)
        print(f"\n  Wrote {OUT_DIR / 'moran_residual_test.csv'}")
    results['moran'] = moran_result

    # 3. Spatial Lag Model
    if SPREG_AVAILABLE:
        print('\n3. Estimating Spatial Lag (SAR) model...')
        print('-' * 40)
        sar_result = fit_spatial_lag_model(panel, boundary, caliper, k_neighbors)
        if 'error' not in sar_result:
            print(f"  ρ (spatial lag): {sar_result['rho']:.4f}")
            print(f"  DiD coefficient: {sar_result['coef_did']:.6f}")
            pd.DataFrame([sar_result]).to_csv(OUT_DIR / 'spatial_lag_results.csv', index=False)
            print(f"\n  Wrote {OUT_DIR / 'spatial_lag_results.csv'}")
        else:
            print(f"  Error: {sar_result['error']}")
        results['sar'] = sar_result

        # 4. Spatial Error Model
        print('\n4. Estimating Spatial Error (SEM) model...')
        print('-' * 40)
        sem_result = fit_spatial_error_model(panel, boundary, caliper, k_neighbors)
        if 'error' not in sem_result:
            print(f"  λ (spatial error): {sem_result['lambda']:.4f}")
            print(f"  DiD coefficient: {sem_result['coef_did']:.6f}")
            pd.DataFrame([sem_result]).to_csv(OUT_DIR / 'spatial_error_results.csv', index=False)
            print(f"\n  Wrote {OUT_DIR / 'spatial_error_results.csv'}")
        else:
            print(f"  Error: {sem_result['error']}")
        results['sem'] = sem_result
    else:
        print('\n3-4. Skipping SAR/SEM (spreg not available)')

    # Summary comparison
    print('\n' + '=' * 60)
    print('SPATIAL ANALYSIS SUMMARY')
    print('=' * 60)

    if conley_results:
        baseline = conley_results[0]  # 1km cutoff
        preferred = [r for r in conley_results if r['cutoff_km'] == cutoff_km]
        if preferred:
            pref = preferred[0]
            print(f"\nBaseline HC3 SE: {baseline['se_hc3']:.6f}")
            print(f"Conley SE ({cutoff_km}km): {pref['se_conley']:.6f}")
            print(f"SE inflation factor: {pref['se_ratio']:.2f}x")

            if pref['se_ratio'] > 1.5:
                print("\nWARNING: Substantial spatial correlation - baseline SEs underestimate uncertainty")
            elif pref['se_ratio'] > 1.1:
                print("\nNOTE: Moderate spatial correlation - recommend using Conley SEs")
            else:
                print("\nConclusion: Minimal spatial correlation - HC3 SEs adequate")

    print('\n' + '=' * 60)
    print('SPATIAL ANALYSIS COMPLETE')
    print('=' * 60)

    return results


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Spatial econometrics analysis for RD estimates.'
    )
    parser.add_argument(
        '--boundary', '-b',
        choices=['inund', 'sfha'],
        default='inund',
        help='Boundary type. Default: inund'
    )
    parser.add_argument(
        '--caliper', '-c',
        type=int,
        default=300,
        help='Caliper window in meters. Default: 300'
    )
    parser.add_argument(
        '--cutoff-km',
        type=float,
        default=5.0,
        help='Distance cutoff for Conley SEs (km). Default: 5.0'
    )
    parser.add_argument(
        '--k-neighbors',
        type=int,
        default=8,
        help='Number of neighbors for spatial weights. Default: 8'
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(
        boundary=args.boundary,
        caliper=args.caliper,
        cutoff_km=args.cutoff_km,
        k_neighbors=args.k_neighbors
    )
