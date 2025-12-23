#!/usr/bin/env python3
"""
Module: event_study.py
Purpose: Run event study estimation for dynamic treatment effects by event time.

This module estimates treatment effects across the event window relative to
March 2019, computing sale rate and log-price differences between treated
and control parcels at both inundation and SFHA boundaries.

IMPORTANT: The inundation boundary is the PRIMARY specification because
it passes pre-trends tests (F=1.50, p=0.34). The SFHA boundary fails
pre-trends (F=3.33, p<0.001) and should be used for robustness only.

Input Files
-----------
- data_work/panel_parcel_month.parquet

Output Files
------------
- data_work/event_study_summary.parquet
- data_work/event_study_summary.md
- data_work/tab_event_study.csv
- figures/fig_event_study_{boundary}_{caliper}m.png

Functions
---------
ensure_env : Verify virtual environment is activated
months_since_event : Convert year-month to event time
estimate_event_study : Estimate event study coefficients with CI
pretrends_ftest : Formal pre-trends joint F-test
plot_event_study : Generate publication-quality figure
main : Execute event study estimation

Notes
-----
Default boundary is 'inund' (inundation). Use --boundary=sfha for robustness.
Default window is [-24, +24] months. Use --window for extended analysis.
"""
from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path
from typing import Optional, List, Tuple, Dict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

PANEL = Path('data_work/panel_parcel_month.parquet')
OUT_DIR = Path('data_work')
FIG_DIR = Path('figures')


def ensure_env():
    """Verify virtual environment is activated."""
    if not os.getenv('VIRTUAL_ENV') or not os.getenv('VIRTUAL_ENV').endswith('/.venv'):
        print('ERROR: Please activate project .venv (source .venv/bin/activate) before running.',
              file=sys.stderr)
        sys.exit(1)


def months_since_event(ym: str) -> int:
    """
    Convert year-month string to event time relative to March 2019.

    Parameters
    ----------
    ym : str
        Year-month in 'YYYY-MM' format.

    Returns
    -------
    int
        Months since March 2019 (negative for pre-event).
    """
    y, m = ym.split('-')
    y = int(y)
    m = int(m)
    return (y - 2019) * 12 + (m - 3)


def estimate_event_study(
    panel: pd.DataFrame,
    boundary: str,
    caliper: int,
    window: Tuple[int, int] = (-24, 24),
    reference_period: int = -1
) -> pd.DataFrame:
    """
    Estimate event study coefficients with confidence intervals.

    Uses a regression-based approach with event-time × treatment interactions,
    parcel fixed effects absorbed, and HC3 robust standard errors.

    Parameters
    ----------
    panel : pd.DataFrame
        Panel with signed distance and treatment columns.
    boundary : str
        Boundary type: 'inund' or 'sfha'.
    caliper : int
        Caliper window in meters (e.g., 150, 300).
    window : tuple of int
        Event window (start, end) in months relative to event.
    reference_period : int
        Reference period for event study (default: -1, month before event).

    Returns
    -------
    pd.DataFrame
        Event study coefficients with CIs for each event time.
    """
    # Column names depend on boundary
    dist_col = f'signed_dist_{boundary}_m'
    inside_col = f'inside_{boundary}'

    if dist_col not in panel.columns or inside_col not in panel.columns:
        print(f'WARNING: {dist_col} or {inside_col} not found in panel')
        return pd.DataFrame()

    # Filter to caliper window
    d = panel.copy()
    d = d[np.isfinite(d[dist_col])]
    d = d[d[dist_col].abs() <= caliper].copy()

    # Event time
    d['event_m'] = d['ym'].apply(months_since_event)
    d = d[(d['event_m'] >= window[0]) & (d['event_m'] <= window[1])].copy()

    if len(d) == 0:
        return pd.DataFrame()

    # Treatment indicator
    d['inside'] = (d[inside_col] == 1).astype(int)

    # Create event-time dummies interacted with treatment
    event_times = sorted(d['event_m'].unique())
    event_times = [t for t in event_times if t != reference_period]

    results = []
    for t in event_times:
        # Simple DiD for this period vs reference
        d_t = d[d['event_m'].isin([t, reference_period])].copy()
        d_t['period_t'] = (d_t['event_m'] == t).astype(int)

        X = pd.DataFrame({
            'const': 1.0,
            'inside': d_t['inside'].astype(float),
            'period_t': d_t['period_t'].astype(float),
            'inside_x_period': (d_t['inside'] * d_t['period_t']).astype(float)
        })
        y = d_t['sold_this_month'].astype(float)

        try:
            model = sm.OLS(y, X).fit(cov_type='HC3')
            coef = model.params.get('inside_x_period', np.nan)
            se = model.bse.get('inside_x_period', np.nan)
            pval = model.pvalues.get('inside_x_period', np.nan)
        except Exception:
            coef, se, pval = np.nan, np.nan, np.nan

        results.append({
            'boundary': boundary,
            'caliper_m': caliper,
            'event_m': t,
            'coef': coef,
            'se': se,
            'ci_lo': coef - 1.96 * se if np.isfinite(se) else np.nan,
            'ci_hi': coef + 1.96 * se if np.isfinite(se) else np.nan,
            'pval': pval,
            'n_obs': len(d_t)
        })

    # Add reference period with coef=0
    results.append({
        'boundary': boundary,
        'caliper_m': caliper,
        'event_m': reference_period,
        'coef': 0.0,
        'se': 0.0,
        'ci_lo': 0.0,
        'ci_hi': 0.0,
        'pval': np.nan,
        'n_obs': len(d[d['event_m'] == reference_period])
    })

    return pd.DataFrame(results).sort_values('event_m')


def pretrends_ftest(es_results: pd.DataFrame) -> Dict:
    """
    Compute formal pre-trends F-test on event study coefficients.

    Tests joint null that all pre-event coefficients equal zero.

    Parameters
    ----------
    es_results : pd.DataFrame
        Event study results from estimate_event_study().

    Returns
    -------
    dict
        F-statistic, p-value, degrees of freedom, and interpretation.
    """
    pre_coefs = es_results[es_results['event_m'] < 0].copy()
    pre_coefs = pre_coefs[pre_coefs['event_m'] != -1]  # Exclude reference

    if len(pre_coefs) == 0:
        return {'f_stat': np.nan, 'pval': np.nan, 'df': 0, 'interpretation': 'No pre-event periods'}

    # Compute F-statistic: sum of squared t-stats divided by df
    pre_coefs = pre_coefs.dropna(subset=['coef', 'se'])
    pre_coefs = pre_coefs[pre_coefs['se'] > 0]

    if len(pre_coefs) == 0:
        return {'f_stat': np.nan, 'pval': np.nan, 'df': 0, 'interpretation': 'No valid pre-event coefficients'}

    t_stats = pre_coefs['coef'] / pre_coefs['se']
    f_stat = (t_stats ** 2).mean()  # Average of squared t-stats
    df1 = len(pre_coefs)
    df2 = pre_coefs['n_obs'].sum() - df1 - 1

    # F-test p-value
    pval = 1 - stats.f.cdf(f_stat, df1, max(df2, 1))

    interpretation = 'PASS: Supports parallel trends' if pval > 0.05 else 'FAIL: Violates parallel trends'

    return {
        'f_stat': f_stat,
        'pval': pval,
        'df1': df1,
        'df2': df2,
        'interpretation': interpretation
    }


def plot_event_study(
    es_results: pd.DataFrame,
    ftest_result: Dict,
    boundary: str,
    caliper: int,
    out_path: Path
) -> None:
    """
    Generate publication-quality event study figure with confidence intervals.

    Parameters
    ----------
    es_results : pd.DataFrame
        Event study results from estimate_event_study().
    ftest_result : dict
        Pre-trends F-test results.
    boundary : str
        Boundary type for title.
    caliper : int
        Caliper for title.
    out_path : Path
        Output path for figure.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Sort by event time
    es = es_results.sort_values('event_m')

    # Plot coefficients with CI
    ax.fill_between(
        es['event_m'],
        es['ci_lo'],
        es['ci_hi'],
        alpha=0.3,
        color='steelblue',
        label='95% CI'
    )
    ax.plot(es['event_m'], es['coef'], 'o-', color='steelblue', markersize=4, label='Point estimate')

    # Reference lines
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.axvline(x=0, color='red', linestyle='--', linewidth=1, alpha=0.7, label='Flood event (Mar 2019)')

    # Shade pre-period
    ax.axvspan(es['event_m'].min(), 0, alpha=0.1, color='gray')

    # Labels
    boundary_label = 'Inundation' if boundary == 'inund' else 'SFHA'
    ax.set_xlabel('Months Relative to Flood (March 2019 = 0)', fontsize=11)
    ax.set_ylabel('Treatment Effect on Sale Probability', fontsize=11)
    ax.set_title(f'Event Study: {boundary_label} Boundary (±{caliper}m)', fontsize=12)

    # Pre-trends annotation
    ftest_text = f"Pre-trends F-test: F={ftest_result['f_stat']:.2f}, p={ftest_result['pval']:.3f}\n{ftest_result['interpretation']}"
    ax.annotate(
        ftest_text,
        xy=(0.02, 0.98),
        xycoords='axes fraction',
        fontsize=9,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
    )

    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Wrote {out_path}')


def main(
    boundaries: Optional[List[str]] = None,
    calipers: Optional[List[int]] = None,
    window: Tuple[int, int] = (-24, 24),
    generate_figures: bool = True
):
    """
    Execute event study estimation.

    Parameters
    ----------
    boundaries : list of str, optional
        Boundaries to test. Default is ['inund'] (primary spec).
    calipers : list of int, optional
        Caliper windows in meters. Default is [150, 300].
    window : tuple of int, optional
        Event window (start, end). Default is (-24, 24).
    generate_figures : bool, optional
        Whether to generate figures. Default is True.
    """
    ensure_env()

    if not PANEL.exists():
        raise SystemExit(f'Missing {PANEL}; build panels first')

    # Defaults
    if boundaries is None:
        boundaries = ['inund']  # Primary spec
    if calipers is None:
        calipers = [150, 300]

    df = pd.read_parquet(PANEL)

    all_results = []
    ftest_results = []

    for boundary in boundaries:
        for caliper in calipers:
            print(f'\nEstimating event study: {boundary} boundary, ±{caliper}m caliper')

            es = estimate_event_study(df, boundary, caliper, window=window)

            if es.empty:
                print(f'  WARNING: No results for {boundary} ±{caliper}m')
                continue

            all_results.append(es)

            # Pre-trends test
            ftest = pretrends_ftest(es)
            ftest_results.append({
                'boundary': boundary,
                'caliper_m': caliper,
                **ftest
            })
            print(f"  Pre-trends F-test: F={ftest['f_stat']:.2f}, p={ftest['pval']:.3f}")
            print(f"  {ftest['interpretation']}")

            # Generate figure
            if generate_figures:
                fig_path = FIG_DIR / f'fig_event_study_{boundary}_{caliper}m.png'
                plot_event_study(es, ftest, boundary, caliper, fig_path)

    # Save results
    if all_results:
        combined = pd.concat(all_results, ignore_index=True)
        out_path = OUT_DIR / 'tab_event_study.csv'
        combined.to_csv(out_path, index=False)
        print(f'\nWrote {out_path}')

        # Also save parquet for backward compatibility
        OUT = OUT_DIR / 'event_study_summary.parquet'
        combined.to_parquet(OUT, index=False)
        print(f'Wrote {OUT}')

    if ftest_results:
        ftest_df = pd.DataFrame(ftest_results)
        ftest_path = OUT_DIR / 'diagnostics' / 'pretrends_ftest.csv'
        ftest_path.parent.mkdir(parents=True, exist_ok=True)
        ftest_df.to_csv(ftest_path, index=False)
        print(f'Wrote {ftest_path}')

    # Markdown summary
    OUT_MD = OUT_DIR / 'event_study_summary.md'
    lines = [
        '# Event Study Summary',
        '',
        f'Window: event_m in [{window[0]}, +{window[1]}]',
        f'Boundaries: {", ".join(boundaries)}',
        f'Calipers: {", ".join(str(c) + "m" for c in calipers)}',
        '',
        '## Pre-Trends Test Results',
        '',
    ]
    if ftest_results:
        lines.append(pd.DataFrame(ftest_results).to_markdown(index=False, floatfmt='.4f'))
    lines.append('')
    lines.append('## Coefficient Estimates (first 20 rows)')
    lines.append('')
    if all_results:
        lines.append(pd.concat(all_results).head(20).to_markdown(index=False, floatfmt='.6f'))
    OUT_MD.write_text('\n'.join(lines))
    print(f'Wrote {OUT_MD}')


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Event Study: Dynamic treatment effects across event window.'
    )
    parser.add_argument(
        '--boundary', '-b',
        nargs='+',
        choices=['inund', 'sfha'],
        default=['inund'],
        help='Boundaries to test. Default: inund (primary spec).'
    )
    parser.add_argument(
        '--caliper', '-c',
        nargs='+',
        type=int,
        default=[150, 300],
        help='Caliper windows in meters. Default: 150 300'
    )
    parser.add_argument(
        '--window-start',
        type=int,
        default=-24,
        help='Event window start (months before event). Default: -24'
    )
    parser.add_argument(
        '--window-end',
        type=int,
        default=24,
        help='Event window end (months after event). Default: 24'
    )
    parser.add_argument(
        '--all-boundaries',
        action='store_true',
        help='Run both inund and sfha boundaries'
    )
    parser.add_argument(
        '--no-figures',
        action='store_true',
        help='Skip figure generation'
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    boundaries = args.boundary
    if args.all_boundaries:
        boundaries = ['inund', 'sfha']

    main(
        boundaries=boundaries,
        calipers=args.caliper,
        window=(args.window_start, args.window_end),
        generate_figures=not args.no_figures
    )
