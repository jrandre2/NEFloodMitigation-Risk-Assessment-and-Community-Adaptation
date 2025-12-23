#!/usr/bin/env python3
from __future__ import annotations
import os, sys, argparse
from pathlib import Path
import pandas as pd
import numpy as np

DEF_DATA = Path('results/integration_run/sfr_regression_data.csv')
OUT_DIR = Path('projects/flood_2019_sales_douglas/outputs')


def ensure_env():
    if not os.getenv('VIRTUAL_ENV') or not os.getenv('VIRTUAL_ENV').endswith('/.venv'):
        print('ERROR: Please activate project .venv (source .venv/bin/activate) before running.', file=sys.stderr)
        sys.exit(1)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='SFHA vs Non-SFHA sales volumes pre/post flood')
    p.add_argument('--data', type=Path, default=DEF_DATA)
    p.add_argument('--event-date', type=str, default='2019-03-14')
    p.add_argument('--window-days', type=int, default=730)
    return p.parse_args()


def main():
    ensure_env()
    args = parse_args()
    df = pd.read_csv(args.data)
    if 'in_sfha' not in df.columns:
        raise SystemExit('in_sfha column not found')
    df['sale_date'] = pd.to_datetime(df['sale_date'], errors='coerce')
    df['has_valid_sale_date'] = pd.to_numeric(df.get('has_valid_sale_date', 0), errors='coerce').fillna(0).astype(int)
    event = pd.Timestamp(args.event_date)
    start = event - pd.Timedelta(days=args.window_days)
    end = event + pd.Timedelta(days=args.window_days)
    sold = df[(df['has_valid_sale_date'] == 1) & (df['sale_date'].between(start, end))].copy()
    sold['period'] = np.where(sold['sale_date'] < event, 'pre', 'post')

    # Aggregates
    counts = sold.pivot_table(index='period', columns='in_sfha', values='sale_date', aggfunc='count').fillna(0).astype(int)
    for c in [0, 1]:
        if c not in counts.columns:
            counts[c] = 0
    counts = counts[[0, 1]].rename(columns={0: 'Non_SFHA', 1: 'SFHA'})
    totals = sold.groupby('period').size().rename('Total')
    share = sold.groupby('period')['in_sfha'].mean().rename('SFHA_share')

    # Ratios
    sfha_ratio = float(counts.loc['post', 'SFHA']) / float(counts.loc['pre', 'SFHA']) if counts.loc['pre', 'SFHA'] > 0 else np.nan
    non_ratio = float(counts.loc['post', 'Non_SFHA']) / float(counts.loc['pre', 'Non_SFHA']) if counts.loc['pre', 'Non_SFHA'] > 0 else np.nan

    # Monthly series (optional diagnostics)
    sold['ym'] = sold['sale_date'].dt.to_period('M')
    monthly = sold.groupby(['ym', 'in_sfha']).size().unstack(fill_value=0)
    monthly.index = monthly.index.astype(str)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    counts_out = OUT_DIR / 'sfha_sales_counts.csv'
    counts.assign(Total=totals).to_csv(counts_out)
    monthly_out = OUT_DIR / 'sfha_sales_monthly.csv'
    monthly.to_csv(monthly_out)

    # Markdown summary
    md = []
    md.append('# SFHA vs Non-SFHA Sales Volumes')
    md.append(f'Window: ±{args.window_days} days | Event: {args.event_date}')
    md.append('')
    md.append('## Counts (±window)')
    md.append(counts.assign(Total=totals).to_markdown())
    md.append('')
    md.append('## SFHA Share of Sales')
    md.append(share.to_frame().to_markdown())
    md.append('')
    md.append('## Post/Pre Ratios (counts)')
    md.append(f'- SFHA: {sfha_ratio:.2f}')
    md.append(f'- Non-SFHA: {non_ratio:.2f}')
    (OUT_DIR / 'sfha_sales_summary.md').write_text('\n'.join(md))
    print('Wrote', counts_out, 'and', OUT_DIR / 'sfha_sales_summary.md')


if __name__ == '__main__':
    main()

