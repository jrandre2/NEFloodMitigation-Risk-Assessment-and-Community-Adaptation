"""
investigation_report.py
=======================
Generate comprehensive report summarizing all investigation findings.

This script aggregates results from:
- decompose_price_effect.py
- rate_price_reconciliation.py
- seller_selection_analysis.py
- buyer_analysis_extended.py
- alternative_trends.py

And produces a unified interpretation of findings.

Author: Claude Code
Date: 2025-12-23
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_WORK = PROJECT_ROOT / "data_work"
DIAGNOSTICS_DIR = DATA_WORK / "diagnostics"
DOC_DIR = PROJECT_ROOT / "doc"


def load_diagnostic_results() -> dict:
    """Load all diagnostic results from CSVs."""

    results = {}

    # List of expected diagnostic files
    files = {
        # Decomposition
        "composition_shift": "composition_shift_test.csv",
        "did_by_age_group": "did_by_age_group.csv",
        "did_by_lot_size": "did_by_lot_size.csv",
        "did_by_value_quartile": "did_by_value_quartile.csv",

        # Rate-price reconciliation
        "rate_vs_price": "rate_vs_price_comparison.csv",
        "conditional_price": "conditional_price_analysis.csv",
        "sold_vs_available": "sold_vs_available_comparison.csv",
        "quality_selection": "quality_selection_test.csv",
        "rate_price_summary": "rate_price_reconciliation_summary.csv",

        # Seller selection
        "seller_propensity": "seller_propensity_model.csv",
        "sellers_vs_holders": "sellers_vs_holders.csv",
        "distress_selection": "distress_selection.csv",
        "selection_did": "selection_did.csv",
        "seller_summary": "seller_selection_summary.csv",

        # Buyer analysis
        "buyer_form_did": "buyer_form_did.csv",
        "portfolio_buyers": "portfolio_buyers.csv",
        "buyer_proximity": "buyer_proximity.csv",
        "price_by_buyer": "price_by_buyer_type.csv",
        "buyer_summary": "buyer_analysis_summary.csv",

        # Trend specifications
        "trend_comparison": "trend_specification_comparison.csv",

        # Original diagnostics
        "pretrends_ftest": "pretrends_ftest.csv",
        "trend_analysis": "trend_analysis.csv",
        "trend_adjusted_did": "trend_adjusted_did.csv",
    }

    for key, filename in files.items():
        path = DIAGNOSTICS_DIR / filename
        if path.exists():
            try:
                results[key] = pd.read_csv(path)
                print(f"  Loaded: {filename}")
            except Exception as e:
                print(f"  Error loading {filename}: {e}")
        else:
            print(f"  Not found: {filename}")

    return results


def summarize_sale_rate_findings(results: dict) -> dict:
    """Summarize findings related to sale rates."""

    findings = {
        "confirmed_freeze": None,
        "sale_rate_did": None,
        "interpretation": ""
    }

    if "rate_vs_price" in results:
        df = results["rate_vs_price"]
        if "did_estimate" in df.columns:
            did = df["did_estimate"].iloc[0]
            findings["sale_rate_did"] = did
            findings["confirmed_freeze"] = did < 0
            findings["interpretation"] = (
                "Sale rates declined inside relative to outside (FREEZE confirmed)"
                if did < 0 else
                "Sale rates did not decline differentially"
            )

    return findings


def summarize_price_findings(results: dict) -> dict:
    """Summarize findings related to prices."""

    findings = {
        "raw_did": None,
        "trend_adjusted_did": None,
        "consistent_across_specs": None,
        "interpretation": ""
    }

    # Raw DiD
    if "trend_adjusted_did" in results:
        df = results["trend_adjusted_did"]
        no_trends = df[df["model"] == "no_trends"]
        with_trends = df[df["model"] == "with_trends"]

        if len(no_trends) > 0:
            findings["raw_did"] = no_trends["did_coef"].values[0]
        if len(with_trends) > 0:
            findings["trend_adjusted_did"] = with_trends["did_coef"].values[0]

    # Consistency across specifications
    if "trend_comparison" in results:
        df = results["trend_comparison"]
        if "did_coef" in df.columns:
            findings["consistent_across_specs"] = (df["did_coef"] > 0).all()

    # Interpretation
    if findings["raw_did"] is not None and findings["raw_did"] > 0:
        findings["interpretation"] = (
            "Prices rose faster inside flood zone (counterintuitive positive effect). "
            f"Consistent across specs: {findings['consistent_across_specs']}"
        )
    elif findings["raw_did"] is not None:
        findings["interpretation"] = "Prices fell inside (supports flight narrative)"

    return findings


def summarize_selection_findings(results: dict) -> dict:
    """Summarize findings related to selection."""

    findings = {
        "selection_detected": False,
        "quality_shift": False,
        "distress_pattern": None,
        "interpretation": ""
    }

    # Seller selection
    if "sellers_vs_holders" in results:
        df = results["sellers_vs_holders"]
        inside_df = df[df["zone"] == "inside"]
        sig_diffs = inside_df[inside_df["pval"] < 0.1] if "pval" in df.columns else pd.DataFrame()
        findings["selection_detected"] = len(sig_diffs) > 0

    # Quality shift
    if "quality_selection" in results:
        df = results["quality_selection"]
        inside_df = df[df["zone"] == "inside"] if "zone" in df.columns else df
        sig_shifts = inside_df[inside_df["pval"] < 0.1] if "pval" in df.columns else pd.DataFrame()
        findings["quality_shift"] = len(sig_shifts) > 0

    # Distress
    if "distress_selection" in results:
        df = results["distress_selection"]
        if "pval" in df.columns:
            sig = df[df["pval"] < 0.1]
            if len(sig) > 0:
                findings["distress_pattern"] = sig["indicator"].tolist()

    # Interpretation
    if findings["selection_detected"] or findings["quality_shift"]:
        findings["interpretation"] = (
            "SELECTION DETECTED: Properties that sold inside flood zone post-flood "
            "are systematically different from those that didn't sell. "
            "This may explain positive price effect."
        )
    else:
        findings["interpretation"] = "No strong selection pattern detected."

    return findings


def summarize_buyer_findings(results: dict) -> dict:
    """Summarize findings related to buyers."""

    findings = {
        "llc_change": None,
        "portfolio_change": None,
        "proximity_change": None,
        "interpretation": ""
    }

    # Buyer form DiD
    if "buyer_form_did" in results:
        df = results["buyer_form_did"]
        llc_row = df[df["owner_form"] == "LLC"] if "owner_form" in df.columns else pd.DataFrame()
        if len(llc_row) > 0 and "did_estimate" in df.columns:
            findings["llc_change"] = llc_row["did_estimate"].values[0]

    # Interpretation
    if findings["llc_change"] is not None:
        if findings["llc_change"] > 0:
            findings["interpretation"] = (
                f"LLC buyer share increased by {findings['llc_change']:.1f}pp inside. "
                "Institutional/investor buyers may be purchasing at higher prices."
            )
        else:
            findings["interpretation"] = (
                f"LLC buyer share decreased by {abs(findings['llc_change']):.1f}pp inside."
            )

    return findings


def determine_resolution(
    rate_findings: dict,
    price_findings: dict,
    selection_findings: dict,
    buyer_findings: dict
) -> dict:
    """Determine resolution based on all findings."""

    resolution = {
        "explanation": None,
        "recommended_narrative": None,
        "recommended_specification": None,
        "confidence": None,
        "caveats": []
    }

    # Check for selection explanation
    if selection_findings["selection_detected"] or selection_findings["quality_shift"]:
        resolution["explanation"] = "SELECTION"
        resolution["recommended_narrative"] = "Freeze with Selection Bias"
        resolution["confidence"] = "MEDIUM"
        resolution["caveats"].append(
            "Positive price effect explained by selection: only higher-quality "
            "properties transact post-flood inside the flood zone."
        )

    # Check for buyer composition explanation
    elif buyer_findings["llc_change"] is not None and buyer_findings["llc_change"] > 2:
        resolution["explanation"] = "BUYER_COMPOSITION"
        resolution["recommended_narrative"] = "Freeze with Investor Entry"
        resolution["confidence"] = "MEDIUM"
        resolution["caveats"].append(
            "Positive price effect may reflect different buyer reservation prices "
            "(investors willing to pay more for flood-risk properties)."
        )

    # If sale rate decline confirmed but price unexplained
    elif rate_findings["confirmed_freeze"]:
        resolution["explanation"] = "UNEXPLAINED"
        resolution["recommended_narrative"] = "Freeze (Price Effect Unexplained)"
        resolution["confidence"] = "LOW"
        resolution["caveats"].append(
            "Sale rate decline confirmed (FREEZE), but positive price effect remains "
            "unexplained. Focus manuscript on sale rate effects; report price results "
            "as auxiliary finding with caveats."
        )

    else:
        resolution["explanation"] = "REQUIRES_MORE_INVESTIGATION"
        resolution["recommended_narrative"] = None
        resolution["confidence"] = "LOW"

    # Specification recommendation
    if price_findings.get("consistent_across_specs"):
        resolution["recommended_specification"] = "Any (results robust)"
    else:
        resolution["recommended_specification"] = "Report range across specifications"

    return resolution


def generate_report(
    results: dict,
    rate_findings: dict,
    price_findings: dict,
    selection_findings: dict,
    buyer_findings: dict,
    resolution: dict
) -> str:
    """Generate the investigation report as markdown."""

    report = []

    # Header
    report.append("# Investigation Report: Counterintuitive Price Effect")
    report.append(f"\n**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    report.append(f"\n**Status**: {resolution['confidence']} CONFIDENCE\n")

    # Executive Summary
    report.append("\n## Executive Summary\n")
    report.append(f"**Explanation**: {resolution['explanation']}")
    report.append(f"\n**Recommended Narrative**: {resolution['recommended_narrative']}")
    report.append(f"\n**Recommended Specification**: {resolution['recommended_specification']}")

    if resolution["caveats"]:
        report.append("\n\n**Key Caveats**:")
        for caveat in resolution["caveats"]:
            report.append(f"\n- {caveat}")

    # Detailed Findings
    report.append("\n\n---\n\n## Detailed Findings")

    # Sale Rate
    report.append("\n\n### Sale Rate Analysis\n")
    report.append(f"- Freeze confirmed: **{rate_findings['confirmed_freeze']}**")
    if rate_findings['sale_rate_did'] is not None:
        report.append(f"\n- DiD estimate: {rate_findings['sale_rate_did']*100:.2f} pp")
    report.append(f"\n- Interpretation: {rate_findings['interpretation']}")

    # Price
    report.append("\n\n### Price Analysis\n")
    if price_findings['raw_did'] is not None:
        report.append(f"- Raw DiD: {price_findings['raw_did']:.4f} "
                     f"({(np.exp(price_findings['raw_did'])-1)*100:+.1f}%)")
    if price_findings['trend_adjusted_did'] is not None:
        report.append(f"\n- Trend-adjusted DiD: {price_findings['trend_adjusted_did']:.4f} "
                     f"({(np.exp(price_findings['trend_adjusted_did'])-1)*100:+.1f}%)")
    report.append(f"\n- Consistent across specs: {price_findings['consistent_across_specs']}")
    report.append(f"\n- Interpretation: {price_findings['interpretation']}")

    # Selection
    report.append("\n\n### Selection Analysis\n")
    report.append(f"- Selection detected: **{selection_findings['selection_detected']}**")
    report.append(f"\n- Quality shift detected: **{selection_findings['quality_shift']}**")
    if selection_findings['distress_pattern']:
        report.append(f"\n- Distress patterns: {', '.join(selection_findings['distress_pattern'])}")
    report.append(f"\n- Interpretation: {selection_findings['interpretation']}")

    # Buyers
    report.append("\n\n### Buyer Analysis\n")
    if buyer_findings['llc_change'] is not None:
        report.append(f"- LLC share change: {buyer_findings['llc_change']:+.1f} pp")
    report.append(f"\n- Interpretation: {buyer_findings['interpretation']}")

    # Recommendations
    report.append("\n\n---\n\n## Recommendations\n")

    if resolution["explanation"] == "SELECTION":
        report.append("""
### For Manuscript

1. **Main finding**: Focus on sale rate decline (FREEZE)
2. **Price results**: Report as secondary with selection caveat
3. **Supplementary**: Full selection analysis in appendix
4. **Title suggestion**: "Freeze, not Flight: Selection Bias in Flood Zone Transactions"

### Specification Choice

Use inundation boundary (passes pre-trends) with:
- Report both with and without trend controls
- Note that price effect is driven by positive selection
- Include Lee bounds for price estimates
""")

    elif resolution["explanation"] == "BUYER_COMPOSITION":
        report.append("""
### For Manuscript

1. **Main finding**: Focus on sale rate decline + buyer composition shift
2. **Interpretation**: Market adjusts through WHO buys, not just WHETHER sales occur
3. **Story**: Investors enter flood zone at higher prices (different risk tolerance)

### Specification Choice

Report main results with discussion of buyer heterogeneity.
""")

    else:
        report.append("""
### For Manuscript

1. **Main finding**: Focus on sale rate decline (FREEZE) - this is robust
2. **Price results**: Report with extensive caveats about counterintuitive direction
3. **Discussion**: Acknowledge limitation that price mechanism unclear
4. **Future work**: Additional investigation needed

### Specification Choice

Report multiple specifications and acknowledge sensitivity.
""")

    # Data Availability Note
    report.append("\n\n---\n\n## Data Files Used\n")
    for key in results.keys():
        report.append(f"\n- `{key}`: ✓")

    return "\n".join(report)


def main():
    """Main function to generate investigation report."""

    print("="*70)
    print("GENERATING INVESTIGATION REPORT")
    print("="*70)

    # Load all results
    print("\nLoading diagnostic results...")
    results = load_diagnostic_results()

    if not results:
        print("\nNo diagnostic results found. Run investigation scripts first:")
        print("  python src/pipeline.py decompose_price_effect")
        print("  python src/pipeline.py rate_price_reconciliation")
        print("  python src/pipeline.py seller_selection")
        print("  python src/pipeline.py buyer_analysis_extended")
        print("  python src/pipeline.py alternative_trends")
        return

    # Summarize findings
    print("\nSummarizing findings...")
    rate_findings = summarize_sale_rate_findings(results)
    price_findings = summarize_price_findings(results)
    selection_findings = summarize_selection_findings(results)
    buyer_findings = summarize_buyer_findings(results)

    # Determine resolution
    print("\nDetermining resolution...")
    resolution = determine_resolution(
        rate_findings, price_findings, selection_findings, buyer_findings
    )

    # Generate report
    print("\nGenerating report...")
    report = generate_report(
        results, rate_findings, price_findings,
        selection_findings, buyer_findings, resolution
    )

    # Save report
    report_path = DOC_DIR / "INVESTIGATION_REPORT.md"
    with open(report_path, "w") as f:
        f.write(report)

    print(f"\nReport saved to: {report_path}")

    # Also print to console
    print("\n" + "="*70)
    print(report)
    print("="*70)

    # Summary output
    summary = {
        "explanation": resolution["explanation"],
        "confidence": resolution["confidence"],
        "freeze_confirmed": rate_findings["confirmed_freeze"],
        "price_positive": price_findings["raw_did"] > 0 if price_findings["raw_did"] else None,
        "selection_detected": selection_findings["selection_detected"],
        "recommended_narrative": resolution["recommended_narrative"]
    }

    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(DIAGNOSTICS_DIR / "investigation_resolution.csv", index=False)

    print(f"\nResolution summary saved to: {DIAGNOSTICS_DIR}/investigation_resolution.csv")


if __name__ == "__main__":
    main()
