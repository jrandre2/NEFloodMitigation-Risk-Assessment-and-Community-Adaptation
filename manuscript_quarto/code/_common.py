"""
Common utilities for Freeze and Rebuild manuscript.
Loads diagnostic CSVs and provides helper functions for table rendering.
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Paths relative to manuscript_quarto/
DATA_DIR = Path(__file__).parent.parent / "data"
FIG_DIR = Path(__file__).parent.parent / "figures"


def load_diagnostic(name: str) -> pd.DataFrame:
    """Load a diagnostic CSV file by name (without .csv extension)."""
    path = DATA_DIR / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Diagnostic file not found: {path}")
    return pd.read_csv(path)


def format_pvalue(p: float, threshold: float = 0.001) -> str:
    """Format p-value for display."""
    if p < threshold:
        return f"<{threshold}"
    return f"{p:.3f}"


def format_ci(lo: float, hi: float, decimals: int = 3) -> str:
    """Format confidence interval as [lo, hi]."""
    return f"[{lo:.{decimals}f}, {hi:.{decimals}f}]"


def format_percent(value: float, decimals: int = 1) -> str:
    """Format value as percentage."""
    return f"{value * 100:.{decimals}f}%"


def add_significance_stars(p: float) -> str:
    """Add significance stars based on p-value."""
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    return ""


def load_rd_summary() -> pd.DataFrame:
    """Load and format the main RD summary results."""
    # Try multiple possible filenames
    for name in ["rd_summary", "tab_rd_summary", "rd_summary_sfr"]:
        try:
            return load_diagnostic(name)
        except FileNotFoundError:
            continue
    raise FileNotFoundError("Could not find RD summary file")


def load_buyer_composition() -> pd.DataFrame:
    """Load buyer composition DiD results."""
    return load_diagnostic("buyer_form_did")


def load_composition_shift() -> pd.DataFrame:
    """Load composition shift analysis."""
    return load_diagnostic("composition_shift_analysis")


def load_pretrends() -> pd.DataFrame:
    """Load pre-trends F-test results."""
    return load_diagnostic("pretrends_ftest")


def load_mccrary() -> pd.DataFrame:
    """Load McCrary density test results."""
    return load_diagnostic("mccrary_density_test")
