#!/bin/bash
# Run Investigation Suite for Counterintuitive Price Effect
# Generated: 2025-12-23

set -e  # Exit on error

PROJECT_DIR="/Users/jesseandrews/Library/CloudStorage/OneDrive-TexasTechUniversity/Projects/Freeze and Flight"
cd "$PROJECT_DIR"

# Activate virtual environment
source .venv/bin/activate

echo "========================================================================"
echo "RUNNING INVESTIGATION SUITE"
echo "========================================================================"

CALIPER=300

# 1. Decompose Price Effect
echo ""
echo "========================================================================"
echo "1/6: Running decompose_price_effect.py"
echo "========================================================================"
python src/07_estimation/decompose_price_effect.py -c $CALIPER

# 2. Rate-Price Reconciliation
echo ""
echo "========================================================================"
echo "2/6: Running rate_price_reconciliation.py"
echo "========================================================================"
python src/07_estimation/rate_price_reconciliation.py -c $CALIPER

# 3. Seller Selection Analysis
echo ""
echo "========================================================================"
echo "3/6: Running seller_selection_analysis.py"
echo "========================================================================"
python src/07_estimation/seller_selection_analysis.py -c $CALIPER

# 4. Buyer Analysis Extended
echo ""
echo "========================================================================"
echo "4/6: Running buyer_analysis_extended.py"
echo "========================================================================"
python src/07_estimation/buyer_analysis_extended.py -c $CALIPER

# 5. Alternative Trends
echo ""
echo "========================================================================"
echo "5/6: Running alternative_trends.py"
echo "========================================================================"
python src/07_estimation/alternative_trends.py -c $CALIPER --start-year 2015 --end-year 2022

# 6. Investigation Report
echo ""
echo "========================================================================"
echo "6/6: Running investigation_report.py"
echo "========================================================================"
python src/07_estimation/investigation_report.py

echo ""
echo "========================================================================"
echo "INVESTIGATION COMPLETE"
echo "========================================================================"
echo ""
echo "Results saved to:"
echo "  - data_work/diagnostics/*.csv"
echo "  - doc/INVESTIGATION_REPORT.md"
echo ""
