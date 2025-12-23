#!/usr/bin/env python3
"""
Analyze pre/post March 2019 housing sales in Douglas County.

Inputs
- results/integration_run/sfr_regression_data.csv (default)

Outputs
- projects/flood_2019_sales_douglas/outputs/pre_post_sales_summary.csv
- projects/flood_2019_sales_douglas/outputs/summary.json
- projects/flood_2019_sales_douglas/outputs/report.md
"""
from __future__ import annotations
import os, sys, json, argparse
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np


DEF_DATA = Path("results/integration_run/sfr_regression_data.csv")
OUT_DIR = Path("projects/flood_2019_sales_douglas/outputs")


def ensure_env():
    # Mirror enforcement pattern used elsewhere in repo
    if not os.getenv('VIRTUAL_ENV') or not os.getenv('VIRTUAL_ENV').endswith('/.venv'):
        print('ERROR: Please activate project .venv (source .venv/bin/activate) before running.', file=sys.stderr)
        sys.exit(1)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pre/Post March 2019 sales summary (Douglas County)")
    p.add_argument("--data", type=Path, default=DEF_DATA, help="Path to sfr_regression_data.csv")
    p.add_argument("--event-date", type=str, default="2019-03-14", help="Flood event date (YYYY-MM-DD)")
    p.add_argument("--window-days", type=int, default=180, help="Half-window size in days for windowed comparison")
    return p.parse_args()


def load(df_path: Path) -> pd.DataFrame:
    if not df_path.exists():
        raise SystemExit(f"Missing data at {df_path}. Generate integration outputs first.")
    df = pd.read_csv(df_path)
    # Coerce date
    df['sale_date'] = pd.to_datetime(df['sale_date'], errors='coerce')
    return df


def get_entity_type(row: pd.Series) -> str:
    if row.get('owner_Corporation', 0) == 1:
        return 'Corporation'
    if row.get('owner_LLC', 0) == 1:
        return 'LLC'
    if row.get('owner_Trust', 0) == 1:
        return 'Trust'
    if row.get('owner_Other', 0) == 1:
        return 'Other'
    return 'Individual'


def pick_sfha_col(df: pd.DataFrame) -> str:
    for c in ('in_sfha', 'in_sfha_correct', 'in_sfha_original'):
        if c in df.columns:
            return c
    return 'in_sfha'  # fallback


def safe_median_log_to_dollars(series: pd.Series) -> float | None:
    s = series.dropna()
    if s.empty:
        return None
    return float(np.exp(s.median()))


def summarize(args: argparse.Namespace) -> dict:
    df = load(args.data)

    # Restrict to rows with a valid sale date where available
    if 'has_valid_sale_date' in df.columns:
        df = df[df['has_valid_sale_date'] == 1].copy()

    event_date = pd.Timestamp(args.event_date)
    df['post_flood'] = (df['sale_date'] >= event_date).astype(int)
    df['year'] = df['sale_date'].dt.year
    df['month'] = df['sale_date'].dt.month
    df['quarter'] = df['sale_date'].dt.quarter

    # Entity labels
    df['entity'] = df.apply(get_entity_type, axis=1)

    # Exposure groups
    sfha_col = pick_sfha_col(df)
    df['in_sfha_flag'] = df[sfha_col] == 1
    df['inund_flag'] = (df['was_inundated'] == 1) if 'was_inundated' in df.columns else False

    # Windowed selection
    wstart = event_date - pd.Timedelta(days=args.window_days)
    wend = event_date + pd.Timedelta(days=args.window_days)
    wmask = df['sale_date'].between(wstart, wend)
    dwin = df[wmask].copy()

    # Helper: median sale value from log_sales_value if present, otherwise use log_total_value
    use_col = 'log_sales_value' if 'log_sales_value' in df.columns else 'log_total_value'

    # March 2019 breakdown (pre vs post within March)
    m2019 = df[(df['year'] == 2019) & (df['month'] == 3)].copy()
    m2019_pre = m2019[m2019['post_flood'] == 0]
    m2019_post = m2019[m2019['post_flood'] == 1]

    # Q1 2019 breakdown
    q1_2019 = df[(df['year'] == 2019) & (df['quarter'] == 1)].copy()
    q1_2019_pre = q1_2019[q1_2019['post_flood'] == 0]
    q1_2019_post = q1_2019[q1_2019['post_flood'] == 1]

    # Windowed counts and ratios
    win_pre = (dwin['post_flood'] == 0).sum()
    win_post = (dwin['post_flood'] == 1).sum()
    win_ratio = (win_post / win_pre) if win_pre else None

    # Entity summary
    ent_rows = []
    ent_summary = {}
    for e, sub in df.groupby('entity'):
        pre = sub[sub['post_flood'] == 0]
        post = sub[sub['post_flood'] == 1]
        rec = {
            'entity': e,
            'pre_count': int(len(pre)),
            'post_count': int(len(post)),
            'post_pre_ratio': (len(post)/len(pre)) if len(pre) else None,
            'pre_median_sales_value': safe_median_log_to_dollars(pre[use_col]) if use_col in pre else None,
            'post_median_sales_value': safe_median_log_to_dollars(post[use_col]) if use_col in post else None,
        }
        ent_summary[e] = rec
        ent_rows.append({**rec, 'group': 'entity'})

    # Exposure summary
    exposure_defs = {
        'Inundated': (df['inund_flag'] == True),
        'SFHA_Not_Inundated': (df['in_sfha_flag'] == True) & (df['inund_flag'] == False),
        'Non_SFHA': (df['in_sfha_flag'] == False)
    }

    exp_summary = {}
    for label, mask in exposure_defs.items():
        sub = df[mask].copy()
        pre = sub[sub['post_flood'] == 0]
        post = sub[sub['post_flood'] == 1]
        rec = {
            'exposure': label,
            'pre_count': int(len(pre)),
            'post_count': int(len(post)),
            'post_pre_ratio': (len(post)/len(pre)) if len(pre) else None,
            'pre_median_sales_value': safe_median_log_to_dollars(pre[use_col]) if use_col in pre else None,
            'post_median_sales_value': safe_median_log_to_dollars(post[use_col]) if use_col in post else None,
        }
        exp_summary[label] = rec
        ent_rows.append({**rec, 'group': 'exposure'})

    # Collect top-level metrics
    payload = {
        'params': {
            'data': str(args.data),
            'event_date': event_date.strftime('%Y-%m-%d'),
            'window_days': args.window_days,
        },
        'march_2019': {
            'pre_count': int(len(m2019_pre)),
            'post_count': int(len(m2019_post)),
            'pre_median_sales_value': safe_median_log_to_dollars(m2019_pre[use_col]) if use_col in m2019_pre else None,
            'post_median_sales_value': safe_median_log_to_dollars(m2019_post[use_col]) if use_col in m2019_post else None,
        },
        'q1_2019': {
            'pre_count': int(len(q1_2019_pre)),
            'post_count': int(len(q1_2019_post)),
            'pre_median_sales_value': safe_median_log_to_dollars(q1_2019_pre[use_col]) if use_col in q1_2019_pre else None,
            'post_median_sales_value': safe_median_log_to_dollars(q1_2019_post[use_col]) if use_col in q1_2019_post else None,
        },
        'window_around_event': {
            'start': wstart.strftime('%Y-%m-%d'),
            'end': wend.strftime('%Y-%m-%d'),
            'pre_count': int(win_pre),
            'post_count': int(win_post),
            'post_pre_ratio': win_ratio,
        },
        'entity_breakdown': ent_summary,
        'exposure_breakdown': exp_summary,
    }

    # Write outputs
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # Row summary as CSV
    out_csv = OUT_DIR / 'pre_post_sales_summary.csv'
    pd.DataFrame(ent_rows).to_csv(out_csv, index=False)

    # JSON payload
    (OUT_DIR / 'summary.json').write_text(json.dumps(payload, indent=2))

    # Simple markdown report
    lines = []
    lines.append(f"# Pre/Post Flood Sales Summary (Douglas County)")
    lines.append("")
    lines.append(f"Event date: {payload['params']['event_date']} | Window: ±{args.window_days} days")
    lines.append("")
    m = payload['march_2019']
    lines.append("## March 2019")
    lines.append(f"- Pre-count: {m['pre_count']} | Post-count: {m['post_count']}")
    if m['pre_median_sales_value'] is not None and m['post_median_sales_value'] is not None:
        lines.append(f"- Median sales value: Pre ${m['pre_median_sales_value']:,.0f} → Post ${m['post_median_sales_value']:,.0f}")
    lines.append("")
    q1 = payload['q1_2019']
    lines.append("## Q1 2019")
    lines.append(f"- Pre-count: {q1['pre_count']} | Post-count: {q1['post_count']}")
    if q1['pre_median_sales_value'] is not None and q1['post_median_sales_value'] is not None:
        lines.append(f"- Median sales value: Pre ${q1['pre_median_sales_value']:,.0f} → Post ${q1['post_median_sales_value']:,.0f}")
    lines.append("")
    win = payload['window_around_event']
    lines.append("## Windowed Comparison")
    lines.append(f"- {win['start']} to {win['end']}")
    lines.append(f"- Pre: {win['pre_count']} | Post: {win['post_count']} | Ratio: {win['post_pre_ratio']:.2f}" if win['post_pre_ratio'] is not None else f"- Pre: {win['pre_count']} | Post: {win['post_count']}")
    lines.append("")
    lines.append("## Entity Breakdown (Post/Pre ratios)")
    for e, rec in payload['entity_breakdown'].items():
        r = rec['post_pre_ratio']
        lines.append(f"- {e}: {r:.2f}x ({rec['pre_count']} → {rec['post_count']})" if r is not None else f"- {e}: n/a")
    lines.append("")
    lines.append("## Exposure Breakdown (Post/Pre ratios)")
    for ex, rec in payload['exposure_breakdown'].items():
        r = rec['post_pre_ratio']
        lines.append(f"- {ex}: {r:.2f}x ({rec['pre_count']} → {rec['post_count']})" if r is not None else f"- {ex}: n/a")
    lines.append("")
    (OUT_DIR / 'report.md').write_text("\n".join(lines))

    # Also print a terse console summary
    print("=== Pre/Post Flood Sales Summary ===")
    print(f"Event: {payload['params']['event_date']} | Window ±{args.window_days}d")
    print(f"March 2019: pre={m['pre_count']}, post={m['post_count']}")
    print(f"Q1 2019: pre={q1['pre_count']}, post={q1['post_count']}")
    if win['post_pre_ratio'] is not None:
        print(f"Windowed: pre={win['pre_count']}, post={win['post_count']}, ratio={win['post_pre_ratio']:.2f}")
    return payload


def main():
    ensure_env()
    args = parse_args()
    summarize(args)


if __name__ == "__main__":
    main()

