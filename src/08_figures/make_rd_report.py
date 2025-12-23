#!/usr/bin/env python3
"""
Module: make_rd_report.py
Purpose: Create consolidated RD summary report in markdown format.

This module aggregates all RD estimation results, Poisson models, and
figure references into a single markdown report for manuscript review.

Input Files
-----------
- data_work/tab_rd_summary.csv
- data_work/tab_rd_price_summary.csv
- data_work/tab_rd_summary_by_*.csv
- data_work/tab_poisson_ring_models.csv
- data_work/tab_nb_ring_models.csv
- data_work/fig_*.png

Output Files
------------
- data_work/report_rd_summary.md

Functions
---------
ensure_env : Verify virtual environment is activated
maybe_read : Safely read CSV file
main : Generate consolidated markdown report
"""
from __future__ import annotations
import os
from pathlib import Path
import pandas as pd

OUT = Path('data_work/report_rd_summary.md')
BASE = Path('data_work')

def ensure_env():
    if not os.getenv('VIRTUAL_ENV') or not os.getenv('VIRTUAL_ENV').endswith('/.venv'):
        raise SystemExit('Activate .venv first')

def maybe_read(path: Path):
    try:
        return pd.read_csv(path)
    except Exception:
        return None

def main():
    """
    Generate consolidated RD summary markdown report.

    Reads all estimation output files, formats as markdown tables,
    lists available figures, and writes to report file.
    """
    ensure_env()
    lines = []
    lines.append('# Boundary RD Summary')
    lines.append('')
    rd_path = BASE / 'tab_rd_summary.csv'
    if rd_path.exists():
        rd = pd.read_csv(rd_path)
        lines.append('## Sale-rate DiD (inside vs outside; pre vs post)')
        lines.append(rd.to_markdown(index=False, floatfmt='.6f'))
        lines.append('')
    pr_path = BASE / 'tab_rd_price_summary.csv'
    if pr_path.exists():
        pr = pd.read_csv(pr_path)
        lines.append('## Log-price DiD (months with sale only)')
        lines.append(pr.to_markdown(index=False, floatfmt='.6f'))
        lines.append('')
    # Group-level summaries (by form/scale)
    for fname, title in [
        ('tab_rd_summary_by_owner_form.csv','## Sale-rate DiD by Entity Form'),
        ('tab_rd_price_summary_by_owner_form.csv','## Log-price DiD by Entity Form (months with sale only)'),
        ('tab_rd_summary_by_owner_scale.csv','## Sale-rate DiD by Owner Scale (single vs multi)'),
        ('tab_rd_price_summary_by_owner_scale.csv','## Log-price DiD by Owner Scale (months with sale only)'),
        ('tab_rd_summary_by_group.csv','## Sale-rate DiD by Group (all groups)'),
        ('tab_rd_price_summary_by_group.csv','## Log-price DiD by Group (all groups)')
    ]:
        p = BASE / fname
        if p.exists():
            df = pd.read_csv(p)
            lines.append(title)
            lines.append(df.to_markdown(index=False, floatfmt='.6f'))
            lines.append('')
    pois = maybe_read(BASE / 'tab_poisson_ring_models.csv')
    nb = maybe_read(BASE / 'tab_nb_ring_models.csv')
    if pois is not None:
        lines.append('## Ring Poisson models (RRs)')
        lines.append(pois.to_markdown(index=False, floatfmt='.6f'))
        lines.append('')
    if nb is not None:
        lines.append('## Ring Negative Binomial models (RRs)')
        lines.append(nb.to_markdown(index=False, floatfmt='.6f'))
        lines.append('')
    figs = [
        'fig_event_study.png',
        'fig_rd_inund_sale_rate.png',
        'fig_rd_inund_price.png',
        'fig_rd_sfha_sale_rate.png',
        'fig_rd_sfha_price.png',
        'fig_sfha_share_rd.png',
        'fig_sfha_share_county.png'
    ]
    lines.append('## Figures (see data_work)')
    for f in figs:
        p = BASE / f
        if p.exists():
            lines.append(f'- {f}')
    OUT.write_text('\n'.join(lines))
    print('Wrote', OUT)

if __name__ == '__main__':
    main()
