#!/usr/bin/env python3
"""
Module: event_study.py
Purpose: Run event study estimation for dynamic treatment effects by event time.

This module estimates treatment effects across the event window [-24, +24]
months relative to March 2019, computing sale rate and log-price differences
between inundated (treated) and non-inundated (control) parcels.

Input Files
-----------
- data_work/panel_parcel_month.parquet

Output Files
------------
- data_work/event_study_summary.parquet
- data_work/event_study_summary.md

Functions
---------
ensure_env : Verify virtual environment is activated
main : Execute event study estimation
"""
from __future__ import annotations
import os, sys
from pathlib import Path
import pandas as pd
import numpy as np

PANEL = Path('data_work/panel_parcel_month.parquet')
OUT = Path('data_work/event_study_summary.parquet')
OUT_MD = Path('data_work/event_study_summary.md')

def ensure_env():
    if not os.getenv('VIRTUAL_ENV') or not os.getenv('VIRTUAL_ENV').endswith('/.venv'):
        print('ERROR: Please activate project .venv (source .venv/bin/activate) before running.', file=sys.stderr)
        sys.exit(1)

def main():
    """
    Execute event study estimation.

    Reads panel, computes event time relative to March 2019, aggregates
    sale rates and median log prices by treatment status and event month,
    computes differences, and writes summary outputs.

    Raises
    ------
    SystemExit
        If panel file or required columns do not exist.
    """
    ensure_env()
    if not PANEL.exists():
        raise SystemExit(f'Missing {PANEL}; build panels first')
    df = pd.read_parquet(PANEL)
    # months since event (2019-03)
    def months_since_event(ym: str) -> int:
        y, m = ym.split('-')
        y = int(y); m = int(m)
        return (y - 2019) * 12 + (m - 3)
    df['event_m'] = df['ym'].apply(months_since_event)
    # Restrict to [-24, +24]
    df = df[(df['event_m'] >= -24) & (df['event_m'] <= 24)].copy()
    # Treated vs control
    treated = df.get('inund_201903')
    if treated is None:
        raise SystemExit('inund_201903 not present in panel')
    df['treated'] = (df['inund_201903'] == 1).astype(int)
    # Aggregate sold_this_month and median log_price
    agg = df.groupby(['treated','event_m']).agg(
        sale_rate=('sold_this_month','mean'),
        med_log_price=('log_price', 'median')
    ).reset_index()
    # Wide for diff
    w_rate = agg.pivot(index='event_m', columns='treated', values='sale_rate').rename(columns={0:'control_rate',1:'treated_rate'})
    w_price = agg.pivot(index='event_m', columns='treated', values='med_log_price').rename(columns={0:'control_price',1:'treated_price'})
    out = w_rate.join(w_price, how='outer').reset_index()
    out['rate_diff'] = out['treated_rate'] - out['control_rate']
    out['price_diff'] = out['treated_price'] - out['control_price']
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT, index=False)
    # Markdown summary
    lines = []
    lines.append('# Event Study Summary (aggregated)')
    lines.append('Window: event_m in [-24, +24]; treated = inund_201903 == 1')
    lines.append(out.head(15).to_markdown(index=False, floatfmt='.4f'))
    OUT_MD.write_text('\n'.join(lines))
    print('Wrote', OUT, 'and', OUT_MD)

if __name__ == '__main__':
    main()

