#!/usr/bin/env python3
"""
Quantify sales differences between multi-holding Individuals and LLCs around the March 2019 flood.

Models:
- Value model (OLS, HC3): log_sales_value ~ LLC_multi + post + LLC_multi×post + controls + Neighborhood FE
- Timing model (Logit, HC3): post ~ LLC_multi + controls + Neighborhood FE

Inputs
- results/integration_run/sfr_regression_data.csv
- results/integration_run/parcels_with_classification.csv (for owner_key and Neighborho FE)

Outputs
- projects/flood_2019_sales_douglas/outputs/sales_single_multi_diffs.md
- projects/flood_2019_sales_douglas/outputs/sales_single_multi_diffs.csv
"""
from __future__ import annotations
import os, sys, argparse
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

DEF_DATA = Path("results/integration_run/sfr_regression_data.csv")
CLASS_CSV = Path("results/integration_run/parcels_with_classification.csv")
OUT_DIR = Path("projects/flood_2019_sales_douglas/outputs")


def ensure_env():
    if not os.getenv('VIRTUAL_ENV') or not os.getenv('VIRTUAL_ENV').endswith('/.venv'):
        print('ERROR: Please activate project .venv (source .venv/bin/activate) before running.', file=sys.stderr)
        sys.exit(1)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sales comparison: multi Individuals vs multi LLCs (± window)")
    p.add_argument("--data", type=Path, default=DEF_DATA)
    p.add_argument("--class-csv", type=Path, default=CLASS_CSV)
    p.add_argument("--event-date", type=str, default="2019-03-14")
    p.add_argument("--window-days", type=int, default=730)
    return p.parse_args()


def load_and_build(args: argparse.Namespace) -> pd.DataFrame:
    df = pd.read_csv(args.data)
    df['sale_date'] = pd.to_datetime(df['sale_date'], errors='coerce')
    # owner_type from dummies in regression data
    def owner_type(row):
        if row.get('owner_LLC', 0) == 1: return 'LLC'
        if row.get('owner_Corporation', 0) == 1: return 'Corporation'
        if row.get('owner_Trust', 0) == 1: return 'Trust'
        if row.get('owner_Other', 0) == 1: return 'Other'
        return 'Individual'
    df['owner_type'] = df.apply(owner_type, axis=1)
    # Attach owner_key and Neighborho FE from classification CSV (fast path)
    head = pd.read_csv(args.class_csv, nrows=0).columns.tolist()
    use = [c for c in ['Parcel_ID','Current_Ow','owner_name','Neighborho_x','Neighborho_y'] if c in head]
    meta = pd.read_csv(args.class_csv, usecols=use)
    meta['Parcel_ID'] = meta['Parcel_ID'].astype(str)
    if 'Current_Ow' in meta.columns:
        meta['owner_key'] = meta['Current_Ow'].astype(str).str.strip()
    elif 'owner_name' in meta.columns:
        meta['owner_key'] = meta['owner_name'].astype(str).str.strip()
    else:
        raise RuntimeError('No owner identifier in classification CSV')
    # Prefer Neighborho_x then Neighborho_y
    fe = meta['Neighborho_x'] if 'Neighborho_x' in meta.columns else None
    if fe is None or fe.isna().all():
        fe = meta['Neighborho_y'] if 'Neighborho_y' in meta.columns else None
    meta['Neighborho'] = fe
    df['Parcel_ID'] = df['Parcel_ID'].astype(str)
    m = df.merge(meta[['Parcel_ID','owner_key','Neighborho']], on='Parcel_ID', how='left')
    m = m.dropna(subset=['owner_key']).copy()
    # Owner portfolio size across entire regression dataset
    counts = m.groupby('owner_key').size().rename('owner_parcel_count')
    m = m.merge(counts, on='owner_key', how='left')
    m['parcel_type'] = np.where(m['owner_parcel_count'] > 1, 'multi', 'single')
    # Sold rows in window
    event = pd.Timestamp(args.event_date)
    start = event - pd.Timedelta(days=args.window_days)
    end = event + pd.Timedelta(days=args.window_days)
    sold = m[(m.get('has_valid_sale_date', 0) == 1) & (m['sale_date'].between(start, end))].copy()
    sold['post'] = (sold['sale_date'] >= event).astype(int)
    # Focus: multi-holding Only (Individuals vs LLCs)
    sold = sold[sold['parcel_type'] == 'multi'].copy()
    sold['LLC_multi'] = (sold['owner_type'] == 'LLC').astype(int)
    # Controls and FE
    # Clean numeric
    for c in ['log_sales_value','log_total_value','log_acres','in_sfha']:
        if c in sold.columns:
            sold[c] = pd.to_numeric(sold[c], errors='coerce')
    # Drop missing essentials
    sold = sold.dropna(subset=['log_sales_value','log_total_value','log_acres']).copy()
    # Drop singleton FE groups
    if 'Neighborho' in sold.columns and sold['Neighborho'].notna().any():
        vc = sold['Neighborho'].value_counts()
        keep = vc[vc >= 2].index
        sold = sold[sold['Neighborho'].isin(keep)].copy()
    return sold


def fit_value_model(df: pd.DataFrame) -> dict:
    # OLS with FE via dummies
    # Baseline: Individual_multi (LLC_multi=0)
    # Interaction: LLC_multi:post
    dummies = pd.get_dummies(df['Neighborho'].astype('category'), prefix='Neighborho', drop_first=True) if 'Neighborho' in df.columns else None
    X = pd.DataFrame({
        'LLC_multi': df['LLC_multi'].astype(float),
        'post': df['post'].astype(float),
        'LLC_multi_post': (df['LLC_multi'] * df['post']).astype(float),
        'log_total_value': df['log_total_value'].astype(float),
        'log_acres': df['log_acres'].astype(float),
        'in_sfha': pd.to_numeric(df.get('in_sfha', 0), errors='coerce').fillna(0).astype(float),
    })
    if dummies is not None:
        X = pd.concat([X, dummies.astype('float32')], axis=1)
    y = df['log_sales_value'].astype(float)
    Xc = sm.add_constant(X, has_constant='add')
    res = sm.OLS(y, Xc).fit(cov_type='HC3')
    # Extract effects and convert to % where applicable
    def pct_effect(b):
        return (np.exp(b) - 1) * 100
    coefs = res.params
    conf = res.conf_int()
    out = []
    for term in ['LLC_multi','post','LLC_multi_post']:
        b = coefs.get(term, np.nan)
        lo, hi = conf.loc[term] if term in conf.index else (np.nan, np.nan)
        out.append({
            'model': 'value_ols',
            'term': term,
            'beta': float(b),
            'beta_ci_lo': float(lo),
            'beta_ci_hi': float(hi),
            'pct_effect': float(pct_effect(b)) if np.isfinite(b) else np.nan,
            'pct_ci_lo': float(pct_effect(lo)) if np.isfinite(lo) else np.nan,
            'pct_ci_hi': float(pct_effect(hi)) if np.isfinite(hi) else np.nan,
            'p_value': float(res.pvalues.get(term, np.nan)),
            'n_obs': int(len(df))
        })
    return {'results': out, 'summary': res.summary().as_text()}


def fit_timing_model(df: pd.DataFrame) -> dict:
    # Logit with FE via dummies; outcome = post
    dummies = pd.get_dummies(df['Neighborho'].astype('category'), prefix='Neighborho', drop_first=True) if 'Neighborho' in df.columns else None
    X = pd.DataFrame({
        'LLC_multi': df['LLC_multi'].astype(float),
        'log_total_value': df['log_total_value'].astype(float),
        'log_acres': df['log_acres'].astype(float),
        'in_sfha': pd.to_numeric(df.get('in_sfha', 0), errors='coerce').fillna(0).astype(float),
    })
    if dummies is not None:
        X = pd.concat([X, dummies.astype('float32')], axis=1)
    y = df['post'].astype(float)
    Xc = sm.add_constant(X, has_constant='add')
    glm = sm.GLM(y, Xc, family=sm.families.Binomial())
    res = glm.fit(cov_type='HC3')
    # Odds ratio for LLC_multi
    b = res.params.get('LLC_multi', np.nan)
    se = res.bse.get('LLC_multi', np.nan)
    lo, hi = b - 1.96*se, b + 1.96*se
    orr = float(np.exp(b)) if np.isfinite(b) else np.nan
    or_lo = float(np.exp(lo)) if np.isfinite(lo) else np.nan
    or_hi = float(np.exp(hi)) if np.isfinite(hi) else np.nan
    # Marginal effect (AME) for LLC_multi
    try:
        me = res.get_margeff(at='overall', method='dydx')
        ame = float(me.margeff[ list(res.params.index).index('LLC_multi') ]) if 'LLC_multi' in res.params.index else np.nan
        ci_me = me.conf_int(alpha=0.05)
        i = list(res.params.index).index('LLC_multi') if 'LLC_multi' in res.params.index else None
        ame_lo = float(ci_me[i,0]) if i is not None else np.nan
        ame_hi = float(ci_me[i,1]) if i is not None else np.nan
    except Exception:
        ame, ame_lo, ame_hi = np.nan, np.nan, np.nan
    return {
        'results': [{
            'model': 'timing_logit',
            'term': 'LLC_multi',
            'odds_ratio': orr,
            'or_ci_lo': or_lo,
            'or_ci_hi': or_hi,
            'ame_pct_points': ame*100 if np.isfinite(ame) else np.nan,
            'ame_ci_lo_pp': ame_lo*100 if np.isfinite(ame_lo) else np.nan,
            'ame_ci_hi_pp': ame_hi*100 if np.isfinite(ame_hi) else np.nan,
            'p_value': float(res.pvalues.get('LLC_multi', np.nan)),
            'n_obs': int(len(df))
        }],
        'summary': res.summary().as_text()
    }


def main():
    ensure_env()
    args = parse_args()
    df = load_and_build(args)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # Fit models
    val = fit_value_model(df)
    tim = fit_timing_model(df)
    # Collect
    rows = val['results'] + tim['results']
    out_csv = OUT_DIR / 'sales_single_multi_diffs.csv'
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    # Markdown
    lines = []
    lines.append('# Sales Differences: Multi Individuals vs Multi LLCs')
    lines.append(f"Window: ±{parse_args().window_days} days | Event: {parse_args().event_date}")
    lines.append('')
    lines.append('## Value model (OLS, HC3)')
    lines.append('Terms: LLC_multi (pre baseline diff), post (post vs pre for Individuals), LLC_multi×post (differential post change for LLCs vs Individuals).')
    vdf = pd.DataFrame(val['results'])
    if not vdf.empty:
        lines.append(vdf[['term','pct_effect','pct_ci_lo','pct_ci_hi','p_value','n_obs']].to_markdown(index=False, floatfmt='.3f'))
    lines.append('')
    lines.append('## Timing model (Logit, HC3)')
    lines.append('Outcome: post (1=post-flood sale among sold). Report odds ratio and average marginal effect (pp).')
    tdf = pd.DataFrame(tim['results'])
    if not tdf.empty:
        lines.append(tdf[['term','odds_ratio','or_ci_lo','or_ci_hi','ame_pct_points','ame_ci_lo_pp','ame_ci_hi_pp','p_value','n_obs']].to_markdown(index=False, floatfmt='.3f'))
    (OUT_DIR / 'sales_single_multi_diffs.md').write_text("\n".join(lines))
    print('Wrote', out_csv, 'and', OUT_DIR / 'sales_single_multi_diffs.md')


if __name__ == '__main__':
    main()

