#!/usr/bin/env python3
"""
Module: rd_diagnostics.py
Purpose: RD identification diagnostics for Freeze and Flight manuscript.

This module implements formal tests to validate the boundary RD design:
1. McCrary density test - test for manipulation at boundary
2. Formal pre-trends test - joint F-test on pre-event coefficients
3. Bandwidth sensitivity - estimate effects across multiple calipers
4. Donut RD - exclude observations within donut of boundary

IMPORTANT: The inundation boundary is the PRIMARY specification because
it passes pre-trends tests (F=1.50, p=0.34). The SFHA boundary fails
pre-trends (F=3.33, p<0.001) and should be used for robustness only.

Input Files
-----------
- data_work/panel_parcel_month.parquet
- data_work/parcel_boundary_distances.parquet

Output Files
------------
- data_work/diagnostics/mccrary_density_test.csv
- data_work/diagnostics/pretrends_ftest.csv
- data_work/diagnostics/bandwidth_sensitivity.csv
- data_work/diagnostics/donut_rd.csv
- figures/fig_mccrary_density.png
- figures/fig_bandwidth_sensitivity.png
- figures/fig_event_study_with_ci.png

See Also
--------
- spatial_econometrics.py : For Conley spatial standard errors
- placebo_tests.py : For falsification tests

References
----------
McCrary, J. (2008). Manipulation of the running variable in the regression
    discontinuity design: A density test. Journal of Econometrics, 142(2), 698-714.
"""
from __future__ import annotations
import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Import figure styling for consistent manuscript fonts
try:
    from src.utils.figure_style import apply_style
    apply_style()
except ImportError:
    pass  # Style module not available, use defaults

# Paths
PANEL = Path('data_work/panel_parcel_month.parquet')
PARCEL_DIST = Path('data_work/parcel_boundary_distances.parquet')
OUT_DIR = Path('data_work/diagnostics')
FIG_DIR = Path('figures')

def ensure_env():
    """Verify virtual environment is activated."""
    if not os.getenv('VIRTUAL_ENV') or not os.getenv('VIRTUAL_ENV').endswith('/.venv'):
        print('ERROR: Please activate project .venv before running.', file=sys.stderr)
        sys.exit(1)

def months_since_event(ym: str) -> int:
    """Convert year-month string to event time relative to March 2019."""
    y, m = ym.split('-')
    return (int(y) - 2019) * 12 + (int(m) - 3)


# =============================================================================
# 1. McCrary Density Test
# =============================================================================

def mccrary_density_test(distances: pd.Series, boundary: float = 0,
                          bandwidth: float = 100, bins: int = 50) -> dict:
    """
    Implement McCrary (2008) density discontinuity test.

    Tests for bunching at the running variable boundary, which would indicate
    manipulation or sorting that violates the RD continuity assumption.

    Parameters
    ----------
    distances : pd.Series
        Signed distances to boundary (negative = inside, positive = outside).
    boundary : float
        Cutoff value (default 0 for signed distance).
    bandwidth : float
        Bandwidth for local linear regression (meters).
    bins : int
        Number of bins for histogram.

    Returns
    -------
    dict
        Dictionary with test statistic, p-value, and log-density estimates.
    """
    # Remove missing values
    d = distances.dropna()

    # Create histogram bins
    bin_edges = np.linspace(d.min(), d.max(), bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_width = bin_edges[1] - bin_edges[0]

    # Count observations in each bin
    counts, _ = np.histogram(d, bins=bin_edges)

    # Convert to density (log scale for smoothness)
    densities = counts / (len(d) * bin_width)
    log_densities = np.log(densities + 1e-10)  # Add small constant to avoid log(0)

    # Separate left and right of cutoff
    left_mask = bin_centers < boundary
    right_mask = bin_centers >= boundary

    # Restrict to bandwidth around cutoff
    bw_mask = np.abs(bin_centers - boundary) <= bandwidth

    left_x = bin_centers[left_mask & bw_mask]
    left_y = log_densities[left_mask & bw_mask]
    right_x = bin_centers[right_mask & bw_mask]
    right_y = log_densities[right_mask & bw_mask]

    # Local linear regression on each side
    if len(left_x) > 2 and len(right_x) > 2:
        # Left side extrapolation to boundary
        X_left = sm.add_constant(left_x)
        model_left = sm.OLS(left_y, X_left).fit()
        pred_left = model_left.predict([1, boundary])[0]
        se_left = np.sqrt(model_left.cov_params()[0, 0] +
                          2 * boundary * model_left.cov_params()[0, 1] +
                          boundary**2 * model_left.cov_params()[1, 1])

        # Right side extrapolation to boundary
        X_right = sm.add_constant(right_x)
        model_right = sm.OLS(right_y, X_right).fit()
        pred_right = model_right.predict([1, boundary])[0]
        se_right = np.sqrt(model_right.cov_params()[0, 0] +
                           2 * boundary * model_right.cov_params()[0, 1] +
                           boundary**2 * model_right.cov_params()[1, 1])

        # Discontinuity estimate and test
        theta = pred_right - pred_left
        se_theta = np.sqrt(se_left**2 + se_right**2)
        z_stat = theta / se_theta if se_theta > 0 else np.nan
        p_value = 2 * (1 - stats.norm.cdf(np.abs(z_stat)))

        return {
            'discontinuity': theta,
            'se': se_theta,
            'z_statistic': z_stat,
            'p_value': p_value,
            'log_density_left': pred_left,
            'log_density_right': pred_right,
            'n_left': len(left_x),
            'n_right': len(right_x),
            'bandwidth': bandwidth,
            'bins': bins,
            'n_obs': len(d),
            'bin_centers': bin_centers,
            'densities': densities
        }
    else:
        return {
            'discontinuity': np.nan,
            'se': np.nan,
            'z_statistic': np.nan,
            'p_value': np.nan,
            'n_obs': len(d),
            'error': 'Insufficient data in bandwidth'
        }


def plot_mccrary(result: dict, title: str, out_path: Path):
    """
    Plot McCrary density test results.

    Parameters
    ----------
    result : dict
        Output from mccrary_density_test().
    title : str
        Plot title.
    out_path : Path
        Output file path.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    if 'bin_centers' in result:
        centers = result['bin_centers']
        densities = result['densities']

        # Separate left/right of cutoff
        left_mask = centers < 0
        right_mask = centers >= 0

        # Plot histograms
        ax.bar(centers[left_mask], densities[left_mask],
               width=centers[1] - centers[0], alpha=0.7,
               color='steelblue', label='Inside (left of cutoff)')
        ax.bar(centers[right_mask], densities[right_mask],
               width=centers[1] - centers[0], alpha=0.7,
               color='coral', label='Outside (right of cutoff)')

        ax.axvline(x=0, color='black', linestyle='--', linewidth=1.5, label='SFHA boundary')

        # Add test results annotation
        if np.isfinite(result['p_value']):
            text = (f"Discontinuity: {result['discontinuity']:.3f}\n"
                    f"z-stat: {result['z_statistic']:.2f}\n"
                    f"p-value: {result['p_value']:.3f}")
            ax.text(0.02, 0.98, text, transform=ax.transAxes,
                    verticalalignment='top', fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax.set_xlabel('Signed Distance to Boundary (m)', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Wrote {out_path}')


# =============================================================================
# 2. Formal Pre-Trends Test
# =============================================================================

def pretrends_test(panel: pd.DataFrame, boundary: str, caliper: int) -> dict:
    """
    Formal pre-trends test with joint F-test.

    Estimates event study specification and tests:
    H0: all pre-event coefficients = 0

    Parameters
    ----------
    panel : pd.DataFrame
        Parcel-month panel data.
    boundary : str
        'sfha' or 'inund'.
    caliper : int
        Caliper width in meters.

    Returns
    -------
    dict
        F-statistic, p-value, and pre-event coefficients.
    """
    dist_col = f'signed_dist_{boundary}_m'
    inside_col = f'inside_{boundary}'

    if dist_col not in panel.columns or inside_col not in panel.columns:
        return {'error': f'Missing columns: {dist_col} or {inside_col}'}

    d = panel.copy()
    d = d[np.isfinite(d[dist_col])]
    d = d[d[dist_col].abs() <= caliper].copy()

    # Event time
    d['event_m'] = d['ym'].apply(months_since_event)
    d['inside'] = (d[inside_col] == 1).astype(int)

    # Create event-time dummies (exclude t=-1 as reference)
    event_times = sorted(d['event_m'].unique())
    pre_times = [t for t in event_times if t < 0 and t != -1]
    post_times = [t for t in event_times if t >= 0]

    # Build design matrix with event-time interactions
    X_cols = ['const', 'inside']
    d['const'] = 1.0

    # Pre-event dummies (for F-test)
    for t in pre_times:
        col_name = f'inside_x_t{t}'
        d[col_name] = ((d['inside'] == 1) & (d['event_m'] == t)).astype(float)
        X_cols.append(col_name)

    # Post-event dummies
    for t in post_times:
        col_name = f'inside_x_t{t}'
        d[col_name] = ((d['inside'] == 1) & (d['event_m'] == t)).astype(float)
        X_cols.append(col_name)

    X = d[X_cols].astype(float)
    y = d['sold_this_month'].astype(float)

    # Estimate model
    model = sm.OLS(y, X).fit(cov_type='HC3')

    # Extract pre-event coefficients
    pre_coefs = {}
    for t in pre_times:
        col_name = f'inside_x_t{t}'
        if col_name in model.params.index:
            pre_coefs[t] = {
                'coef': model.params[col_name],
                'se': model.bse[col_name],
                'ci_lo': model.params[col_name] - 1.96 * model.bse[col_name],
                'ci_hi': model.params[col_name] + 1.96 * model.bse[col_name]
            }

    # Joint F-test on pre-event coefficients
    pre_cols = [f'inside_x_t{t}' for t in pre_times]
    pre_cols_in_model = [c for c in pre_cols if c in model.params.index]

    if len(pre_cols_in_model) > 0:
        # Construct restriction matrix R @ beta = 0
        R = np.zeros((len(pre_cols_in_model), len(model.params)))
        for i, col in enumerate(pre_cols_in_model):
            R[i, model.params.index.get_loc(col)] = 1

        # Wald test
        f_test = model.f_test(R)
        f_stat = float(f_test.fvalue)
        f_pval = float(f_test.pvalue)
    else:
        f_stat = np.nan
        f_pval = np.nan

    # Also collect post-event coefficients for completeness
    post_coefs = {}
    for t in post_times:
        col_name = f'inside_x_t{t}'
        if col_name in model.params.index:
            post_coefs[t] = {
                'coef': model.params[col_name],
                'se': model.bse[col_name],
                'ci_lo': model.params[col_name] - 1.96 * model.bse[col_name],
                'ci_hi': model.params[col_name] + 1.96 * model.bse[col_name]
            }

    return {
        'boundary': boundary,
        'caliper': caliper,
        'f_statistic': f_stat,
        'f_pvalue': f_pval,
        'n_pre_coefs': len(pre_coefs),
        'pre_coefs': pre_coefs,
        'post_coefs': post_coefs,
        'n_obs': len(d)
    }


def plot_event_study(result: dict, out_path: Path):
    """
    Plot event study coefficients with confidence intervals.

    Parameters
    ----------
    result : dict
        Output from pretrends_test().
    out_path : Path
        Output file path.
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    # Combine pre and post coefficients
    all_coefs = {}
    all_coefs.update(result.get('pre_coefs', {}))
    all_coefs.update(result.get('post_coefs', {}))

    if not all_coefs:
        print(f'No coefficients to plot for {out_path}')
        return

    times = sorted(all_coefs.keys())
    coefs = [all_coefs[t]['coef'] for t in times]
    ci_lo = [all_coefs[t]['ci_lo'] for t in times]
    ci_hi = [all_coefs[t]['ci_hi'] for t in times]

    # Add reference period (t=-1, coef=0)
    ref_idx = times.index(min([t for t in times if t >= 0])) if any(t >= 0 for t in times) else len(times)
    times.insert(ref_idx, -1)
    coefs.insert(ref_idx, 0)
    ci_lo.insert(ref_idx, 0)
    ci_hi.insert(ref_idx, 0)

    # Plot
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1)
    ax.axvline(x=-0.5, color='red', linestyle='--', linewidth=1.5,
               label='Flood event (March 2019)')

    # Error bars
    ax.errorbar(times, coefs,
                yerr=[np.array(coefs) - np.array(ci_lo),
                      np.array(ci_hi) - np.array(coefs)],
                fmt='o', color='steelblue', capsize=3, capthick=1,
                markersize=5, linewidth=1)

    # Fill pre-period in different shade
    pre_mask = np.array(times) < 0
    ax.scatter(np.array(times)[pre_mask], np.array(coefs)[pre_mask],
               color='steelblue', s=40, zorder=5)
    ax.scatter(np.array(times)[~pre_mask], np.array(coefs)[~pre_mask],
               color='coral', s=40, zorder=5)

    # Annotations
    ax.set_xlabel('Event Time (months relative to March 2019)', fontsize=11)
    ax.set_ylabel('Inside × Time Coefficient', fontsize=11)
    ax.set_title(f"Event Study: {result['boundary'].upper()} Boundary (±{result['caliper']}m)\n"
                 f"Pre-trends F-test: F={result['f_statistic']:.2f}, p={result['f_pvalue']:.3f}",
                 fontsize=12)
    ax.legend(loc='upper left')

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Wrote {out_path}')


# =============================================================================
# 3. Bandwidth Sensitivity
# =============================================================================

def bandwidth_sensitivity(panel: pd.DataFrame, boundary: str,
                           calipers: list[int] = None) -> pd.DataFrame:
    """
    Estimate main DiD effect across multiple bandwidth choices.

    Parameters
    ----------
    panel : pd.DataFrame
        Parcel-month panel data.
    boundary : str
        'sfha' or 'inund'.
    calipers : list[int]
        List of caliper widths to test.

    Returns
    -------
    pd.DataFrame
        DiD estimates at each bandwidth.
    """
    if calipers is None:
        calipers = [100, 150, 200, 250, 300, 400, 500]

    dist_col = f'signed_dist_{boundary}_m'
    inside_col = f'inside_{boundary}'

    if dist_col not in panel.columns or inside_col not in panel.columns:
        return pd.DataFrame()

    results = []
    for cal in calipers:
        d = panel.copy()
        d = d[np.isfinite(d[dist_col])]
        d = d[d[dist_col].abs() <= cal].copy()

        d['event_m'] = d['ym'].apply(months_since_event)
        d['post'] = (d['event_m'] >= 0).astype(int)
        d['inside'] = (d[inside_col] == 1).astype(int)

        # DiD regression
        X = pd.DataFrame({
            'const': 1.0,
            'inside': d['inside'].astype(float),
            'post': d['post'].astype(float),
            'inside_x_post': (d['inside'] * d['post']).astype(float)
        })
        y = d['sold_this_month'].astype(float)

        model = sm.OLS(y, X).fit(cov_type='HC3')
        b = model.params.get('inside_x_post', np.nan)
        se = model.bse.get('inside_x_post', np.nan)

        results.append({
            'boundary': boundary,
            'caliper_m': cal,
            'DiD_estimate': b,
            'se': se,
            'ci_lo': b - 1.96 * se,
            'ci_hi': b + 1.96 * se,
            't_stat': b / se if se > 0 else np.nan,
            'n_obs': len(d),
            'n_inside': (d['inside'] == 1).sum(),
            'n_outside': (d['inside'] == 0).sum()
        })

    return pd.DataFrame(results)


def plot_bandwidth_sensitivity(df: pd.DataFrame, out_path: Path):
    """
    Plot bandwidth sensitivity analysis.

    Parameters
    ----------
    df : pd.DataFrame
        Output from bandwidth_sensitivity().
    out_path : Path
        Output file path.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    calipers = df['caliper_m'].values
    coefs = df['DiD_estimate'].values
    ci_lo = df['ci_lo'].values
    ci_hi = df['ci_hi'].values

    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1)

    ax.errorbar(calipers, coefs,
                yerr=[coefs - ci_lo, ci_hi - coefs],
                fmt='o-', color='steelblue', capsize=4, capthick=1.5,
                markersize=8, linewidth=2)

    # Highlight main specifications
    main_cals = [150, 300]
    for cal in main_cals:
        if cal in calipers:
            idx = list(calipers).index(cal)
            ax.scatter([cal], [coefs[idx]], color='red', s=100, zorder=6,
                       marker='*', label=f'Main spec (±{cal}m)' if cal == main_cals[0] else '')

    ax.set_xlabel('Caliper Width (m)', fontsize=11)
    ax.set_ylabel('DiD Estimate (inside × post)', fontsize=11)
    ax.set_title(f"Bandwidth Sensitivity: {df['boundary'].iloc[0].upper()} Boundary", fontsize=12)
    ax.legend(loc='best')

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Wrote {out_path}')


# =============================================================================
# 4. Donut RD
# =============================================================================

def donut_rd(panel: pd.DataFrame, boundary: str, caliper: int,
              donut_inner: int = 0, donut_outer: int = 100) -> dict:
    """
    Estimate DiD excluding observations within a "donut" around the boundary.

    This addresses SUTVA concerns by excluding the spillover zone.

    Parameters
    ----------
    panel : pd.DataFrame
        Parcel-month panel data.
    boundary : str
        'sfha' or 'inund'.
    caliper : int
        Outer caliper width in meters.
    donut_inner : int
        Inner donut boundary (exclude |distance| < donut_inner).
    donut_outer : int
        Outer donut boundary for exclusion on the outside.

    Returns
    -------
    dict
        DiD estimate with and without donut exclusion.
    """
    dist_col = f'signed_dist_{boundary}_m'
    inside_col = f'inside_{boundary}'

    if dist_col not in panel.columns or inside_col not in panel.columns:
        return {'error': f'Missing columns'}

    d = panel.copy()
    d = d[np.isfinite(d[dist_col])]
    d['event_m'] = d['ym'].apply(months_since_event)
    d['post'] = (d['event_m'] >= 0).astype(int)
    d['inside'] = (d[inside_col] == 1).astype(int)

    # Standard caliper
    d_std = d[d[dist_col].abs() <= caliper].copy()

    # Donut: exclude inside the donut zone on the OUTSIDE of the boundary
    # (i.e., parcels 0 to donut_outer meters outside)
    d_donut = d.copy()
    # Keep: inside parcels within caliper, OR outside parcels beyond donut_outer
    d_donut = d_donut[
        (d_donut[dist_col].abs() <= caliper) &  # within caliper
        ~((d_donut['inside'] == 0) & (d_donut[dist_col].abs() <= donut_outer))  # exclude donut on outside
    ].copy()

    def estimate_did(df):
        X = pd.DataFrame({
            'const': 1.0,
            'inside': df['inside'].astype(float),
            'post': df['post'].astype(float),
            'inside_x_post': (df['inside'] * df['post']).astype(float)
        })
        y = df['sold_this_month'].astype(float)
        model = sm.OLS(y, X).fit(cov_type='HC3')
        b = model.params.get('inside_x_post', np.nan)
        se = model.bse.get('inside_x_post', np.nan)
        return b, se, len(df)

    b_std, se_std, n_std = estimate_did(d_std)
    b_donut, se_donut, n_donut = estimate_did(d_donut)

    return {
        'boundary': boundary,
        'caliper': caliper,
        'donut_exclusion_m': donut_outer,
        'DiD_standard': b_std,
        'se_standard': se_std,
        'ci_lo_standard': b_std - 1.96 * se_std,
        'ci_hi_standard': b_std + 1.96 * se_std,
        'n_standard': n_std,
        'DiD_donut': b_donut,
        'se_donut': se_donut,
        'ci_lo_donut': b_donut - 1.96 * se_donut,
        'ci_hi_donut': b_donut + 1.96 * se_donut,
        'n_donut': n_donut,
        'diff': b_donut - b_std,
        'pct_change': (b_donut - b_std) / abs(b_std) * 100 if b_std != 0 else np.nan
    }


# =============================================================================
# 5. SUTVA Violation Bounds
# =============================================================================

def sutva_bounds(
    panel: pd.DataFrame,
    boundary: str,
    caliper: int,
    ring_spillover_estimate: float = None,
    ring_spillover_se: float = None
) -> dict:
    """
    Compute bounds on treatment effect accounting for SUTVA violations (spillovers).

    When treatment spills over to control units near the boundary, the standard
    DiD estimate is biased. This function computes adjusted bounds that account
    for potential spillover contamination of the control group.

    Parameters
    ----------
    panel : pd.DataFrame
        Parcel-month panel data.
    boundary : str
        'sfha' or 'inund'.
    caliper : int
        Caliper width in meters.
    ring_spillover_estimate : float, optional
        Pre-estimated spillover effect on controls near boundary (from ring model).
        If None, estimated from data using 0-250m control band.
    ring_spillover_se : float, optional
        Standard error of spillover estimate.

    Returns
    -------
    dict
        DiD estimates with SUTVA bounds.

    Notes
    -----
    Under SUTVA, we assume: Y_i(D=1) = Y_i(D_i=1) for all i
    When this fails because nearby treated units affect controls, the control
    group outcome reflects: Y_control = Y_0 + spillover_effect

    The true treatment effect is bounded:
    - Lower bound: DiD_raw (assumes no spillover)
    - Upper bound: DiD_raw + spillover_effect (full spillover correction)

    The spillover_effect represents how much controls near the boundary are
    "contaminated" by proximity to treated units.
    """
    dist_col = f'signed_dist_{boundary}_m'
    inside_col = f'inside_{boundary}'

    if dist_col not in panel.columns or inside_col not in panel.columns:
        return {'error': f'Missing columns: {dist_col} or {inside_col}'}

    d = panel.copy()
    d = d[np.isfinite(d[dist_col])]
    d = d[d[dist_col].abs() <= caliper].copy()

    d['event_m'] = d['ym'].apply(months_since_event)
    d['post'] = (d['event_m'] >= 0).astype(int)
    d['inside'] = (d[inside_col] == 1).astype(int)
    d['inside_x_post'] = d['inside'] * d['post']

    # Standard DiD estimate
    X = pd.DataFrame({
        'const': 1.0,
        'inside': d['inside'].astype(float),
        'post': d['post'].astype(float),
        'inside_x_post': (d['inside'] * d['post']).astype(float)
    })
    y = d['sold_this_month'].astype(float)

    model = sm.OLS(y, X).fit(cov_type='HC3')
    did_raw = model.params.get('inside_x_post', np.nan)
    did_se = model.bse.get('inside_x_post', np.nan)

    # Estimate spillover effect if not provided
    if ring_spillover_estimate is None:
        # Create distance rings for outside parcels
        outside = d[d['inside'] == 0].copy()

        # Define spillover zone as inner half of the caliper
        # This ensures we have both "near" and "far" controls within the caliper
        spillover_threshold = caliper / 2

        outside['near_boundary'] = (outside[dist_col].abs() <= spillover_threshold).astype(int)
        outside['far_boundary'] = (outside[dist_col].abs() > spillover_threshold).astype(int)

        # Estimate differential change in "near" vs "far" outside zones
        # This captures how much the near-boundary controls differ from far controls
        n_near = outside['near_boundary'].sum()
        n_far = outside['far_boundary'].sum()

        if n_near >= 50 and n_far >= 50:
            X_spill = pd.DataFrame({
                'const': 1.0,
                'near': outside['near_boundary'].astype(float),
                'post': outside['post'].astype(float),
                'near_x_post': (outside['near_boundary'] * outside['post']).astype(float)
            })
            y_spill = outside['sold_this_month'].astype(float)

            model_spill = sm.OLS(y_spill, X_spill).fit(cov_type='HC3')
            spillover_estimate = model_spill.params.get('near_x_post', 0)
            spillover_se = model_spill.bse.get('near_x_post', np.nan)
        else:
            # Insufficient variation to estimate spillover; assume zero
            spillover_estimate = 0
            spillover_se = np.nan
    else:
        spillover_estimate = ring_spillover_estimate
        spillover_se = ring_spillover_se if ring_spillover_se is not None else np.nan

    # Compute bounds
    # If controls near boundary are "contaminated" by spillover, they look more like treated
    # This means the control group counterfactual is too high, biasing DiD downward
    # True effect = DiD_raw + spillover_correction

    # Lower bound: assume no spillover (DiD as-is)
    lower_bound = did_raw

    # Upper bound: assume full spillover correction
    upper_bound = did_raw + abs(spillover_estimate)

    # Compute combined standard error for bounds
    if np.isfinite(spillover_se):
        bound_se = np.sqrt(did_se**2 + spillover_se**2)
    else:
        bound_se = did_se

    # Calculate proportion of controls in spillover zone
    spillover_threshold = caliper / 2  # Match the threshold used above
    n_controls_total = (d['inside'] == 0).sum()
    n_controls_near = ((d['inside'] == 0) & (d[dist_col].abs() <= spillover_threshold)).sum()
    pct_controls_near = n_controls_near / n_controls_total * 100 if n_controls_total > 0 else 0

    return {
        'boundary': boundary,
        'caliper_m': caliper,
        'did_raw': did_raw,
        'did_se': did_se,
        'spillover_estimate': spillover_estimate,
        'spillover_se': spillover_se,
        'lower_bound': lower_bound,
        'upper_bound': upper_bound,
        'bound_se': bound_se,
        'ci_lo_lower': lower_bound - 1.96 * did_se,
        'ci_hi_lower': lower_bound + 1.96 * did_se,
        'ci_lo_upper': upper_bound - 1.96 * bound_se,
        'ci_hi_upper': upper_bound + 1.96 * bound_se,
        'pct_controls_in_spillover_zone': pct_controls_near,
        'n_controls_total': n_controls_total,
        'n_controls_near_boundary': n_controls_near,
        'n_obs': len(d),
        'interpretation': (
            f"True effect bounded in [{lower_bound:.4f}, {upper_bound:.4f}]. "
            f"If spillovers contaminate {pct_controls_near:.1f}% of controls near boundary, "
            f"the raw DiD ({did_raw:.4f}) may underestimate the true effect by up to {abs(spillover_estimate):.4f}."
        )
    }


# =============================================================================
# Main
# =============================================================================

def main():
    """Run all RD diagnostics."""
    ensure_env()

    # Create output directories
    OUT_DIR.mkdir(exist_ok=True, parents=True)
    FIG_DIR.mkdir(exist_ok=True, parents=True)

    # Load data
    if not PANEL.exists():
        raise SystemExit(f'Missing {PANEL}')
    panel = pd.read_parquet(PANEL)

    if not PARCEL_DIST.exists():
        raise SystemExit(f'Missing {PARCEL_DIST}')
    parcel_dist = pd.read_parquet(PARCEL_DIST)

    print('='*60)
    print('RD DIAGNOSTICS FOR FREEZE AND FLIGHT')
    print('='*60)

    # 1. McCrary Density Test
    print('\n1. McCrary Density Test')
    print('-'*40)

    mccrary_results = []
    for boundary in ['sfha', 'inund']:
        dist_col = f'signed_dist_{boundary}_m'
        if dist_col in parcel_dist.columns:
            # Restrict to RD window for density test
            dists = parcel_dist[parcel_dist[dist_col].abs() <= 500][dist_col]
            result = mccrary_density_test(dists, bandwidth=200, bins=80)
            result['boundary'] = boundary
            mccrary_results.append(result)

            print(f"\n{boundary.upper()} Boundary:")
            print(f"  Discontinuity (log-density): {result['discontinuity']:.4f}")
            print(f"  z-statistic: {result['z_statistic']:.3f}")
            print(f"  p-value: {result['p_value']:.4f}")

            # Plot
            if 'bin_centers' in result:
                plot_mccrary(result, f'McCrary Density Test: {boundary.upper()} Boundary',
                            FIG_DIR / f'fig_mccrary_{boundary}.png')

    if mccrary_results:
        mccrary_df = pd.DataFrame([{k: v for k, v in r.items()
                                    if k not in ['bin_centers', 'densities']}
                                   for r in mccrary_results])
        mccrary_df.to_csv(OUT_DIR / 'mccrary_density_test.csv', index=False)
        print(f"\nWrote {OUT_DIR / 'mccrary_density_test.csv'}")

    # 2. Pre-Trends Test
    print('\n\n2. Formal Pre-Trends Test')
    print('-'*40)

    pretrends_results = []
    for boundary in ['sfha', 'inund']:
        for cal in [150, 300]:
            result = pretrends_test(panel, boundary, cal)
            if 'error' not in result:
                pretrends_results.append({
                    'boundary': boundary,
                    'caliper': cal,
                    'f_statistic': result['f_statistic'],
                    'f_pvalue': result['f_pvalue'],
                    'n_pre_coefs': result['n_pre_coefs'],
                    'n_obs': result['n_obs']
                })

                print(f"\n{boundary.upper()} ±{cal}m:")
                print(f"  F-statistic: {result['f_statistic']:.3f}")
                print(f"  p-value: {result['f_pvalue']:.4f}")
                print(f"  Conclusion: {'FAIL' if result['f_pvalue'] < 0.05 else 'PASS'} "
                      f"(H0: all pre-coefs = 0)")

                # Plot event study
                plot_event_study(result,
                                FIG_DIR / f'fig_event_study_{boundary}_{cal}m.png')

    if pretrends_results:
        pd.DataFrame(pretrends_results).to_csv(OUT_DIR / 'pretrends_ftest.csv', index=False)
        print(f"\nWrote {OUT_DIR / 'pretrends_ftest.csv'}")

    # 3. Bandwidth Sensitivity
    print('\n\n3. Bandwidth Sensitivity')
    print('-'*40)

    bw_results = []
    for boundary in ['sfha', 'inund']:
        df = bandwidth_sensitivity(panel, boundary)
        if not df.empty:
            bw_results.append(df)
            print(f"\n{boundary.upper()} Boundary:")
            print(df[['caliper_m', 'DiD_estimate', 'se', 'ci_lo', 'ci_hi', 'n_obs']].to_string(index=False))

            plot_bandwidth_sensitivity(df, FIG_DIR / f'fig_bandwidth_{boundary}.png')

    if bw_results:
        pd.concat(bw_results, ignore_index=True).to_csv(OUT_DIR / 'bandwidth_sensitivity.csv', index=False)
        print(f"\nWrote {OUT_DIR / 'bandwidth_sensitivity.csv'}")

    # 4. Donut RD
    print('\n\n4. Donut RD (Spillover Robustness)')
    print('-'*40)

    donut_results = []
    for boundary in ['sfha', 'inund']:
        for donut in [50, 100, 150, 200, 250]:
            result = donut_rd(panel, boundary, caliper=300, donut_outer=donut)
            if 'error' not in result:
                donut_results.append(result)

    if donut_results:
        donut_df = pd.DataFrame(donut_results)
        donut_df.to_csv(OUT_DIR / 'donut_rd.csv', index=False)
        print(f"\nWrote {OUT_DIR / 'donut_rd.csv'}")

        print("\nDonut RD Results (SFHA ±300m):")
        sfha_donut = donut_df[donut_df['boundary'] == 'sfha']
        print(sfha_donut[['donut_exclusion_m', 'DiD_standard', 'DiD_donut',
                          'pct_change', 'n_donut']].to_string(index=False))

    # 5. SUTVA Bounds
    print('\n\n5. SUTVA Violation Bounds')
    print('-'*40)

    sutva_results = []
    for boundary in ['sfha', 'inund']:
        for cal in [150, 300]:
            result = sutva_bounds(panel, boundary, cal)
            if 'error' not in result:
                sutva_results.append(result)

                print(f"\n{boundary.upper()} ±{cal}m:")
                print(f"  Raw DiD: {result['did_raw']:.4f} (SE: {result['did_se']:.4f})")
                print(f"  Spillover estimate: {result['spillover_estimate']:.4f}")
                print(f"  SUTVA Bounds: [{result['lower_bound']:.4f}, {result['upper_bound']:.4f}]")
                print(f"  Controls in spillover zone: {result['pct_controls_in_spillover_zone']:.1f}%")

    if sutva_results:
        sutva_df = pd.DataFrame([{k: v for k, v in r.items() if k != 'interpretation'}
                                  for r in sutva_results])
        sutva_df.to_csv(OUT_DIR / 'sutva_bounds.csv', index=False)
        print(f"\nWrote {OUT_DIR / 'sutva_bounds.csv'}")

    print('\n' + '='*60)
    print('DIAGNOSTICS COMPLETE')
    print('='*60)

    # Summary
    print('\n\nSUMMARY OF KEY FINDINGS:')
    print('-'*40)

    if mccrary_results:
        sfha_mc = [r for r in mccrary_results if r['boundary'] == 'sfha']
        if sfha_mc:
            r = sfha_mc[0]
            manip = 'NO evidence' if r['p_value'] > 0.05 else 'POSSIBLE evidence'
            print(f"1. McCrary (SFHA): {manip} of manipulation (p={r['p_value']:.3f})")

    if pretrends_results:
        sfha_pt = [r for r in pretrends_results if r['boundary'] == 'sfha' and r['caliper'] == 300]
        if sfha_pt:
            r = sfha_pt[0]
            valid = 'VALID' if r['f_pvalue'] > 0.05 else 'VIOLATED'
            print(f"2. Pre-trends (SFHA ±300m): Parallel trends assumption {valid} (p={r['f_pvalue']:.3f})")

    if donut_results:
        sfha_d = [r for r in donut_results if r['boundary'] == 'sfha' and r['donut_exclusion_m'] == 100]
        if sfha_d:
            r = sfha_d[0]
            print(f"3. Donut RD (100m exclusion): Effect changes {r['pct_change']:.1f}% after excluding spillover zone")


if __name__ == '__main__':
    main()
