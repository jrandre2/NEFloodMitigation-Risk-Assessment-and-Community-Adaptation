# Flood 2019: Pre/Post Housing Sales (Douglas County)

Purpose: a focused, reproducible space to analyze housing sales before and after the March 2019 flooding in Douglas County.

Scope
- Event: March 14, 2019 (Douglas County flood month)
- Data: uses `results/integration_run/sfr_regression_data.csv` (parcel-level with sales flags and exposure)
- Outputs: concise pre/post counts, medians, and ratios by month/quarter/entity/exposure

Quick Start
- Ensure the project virtualenv is active: `source .venv/bin/activate`
- Run summary: `python projects/flood_2019_sales_douglas/analyze_pre_post_sales.py`
- Optional params:
  - `--data results/integration_run/sfr_regression_data.csv`
  - `--event-date 2019-03-14`
  - `--window-days 180` (for the windowed pre/post comparison)

Outputs
- `projects/flood_2019_sales_douglas/outputs/pre_post_sales_summary.csv`
- `projects/flood_2019_sales_douglas/outputs/summary.json`
- `projects/flood_2019_sales_douglas/outputs/report.md`

Notes
- The repo already contains prior analyses and narratives (see root-level markdown and `run_*` scripts). This subproject consolidates the narrow “pre/post March 2019” slice and keeps it easy to iterate.

