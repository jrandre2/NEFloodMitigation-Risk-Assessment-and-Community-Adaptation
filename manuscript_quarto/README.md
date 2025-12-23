# Freeze and Rebuild - Quarto Manuscript

This directory contains the Quarto manuscript for "Freeze and Rebuild: Liquidity Shocks and Composition Changes in Housing Markets After a Major Flood."

## Prerequisites

### Required Software

1. **Quarto** (>= 1.4)
   ```bash
   # macOS (Homebrew)
   brew install quarto

   # Or download from https://quarto.org/docs/get-started/
   ```

2. **Python** (>= 3.9) with required packages:
   ```bash
   pip install pandas numpy matplotlib seaborn tabulate great-tables statsmodels
   ```

3. **LaTeX** (for PDF output):
   ```bash
   # macOS
   brew install --cask mactex-no-gui
   # Or install TinyTeX via Quarto:
   quarto install tinytex
   ```

### Optional

- **R** with `knitr` (if using R code chunks)
- **VS Code** with Quarto extension for editing

## Directory Structure

```
manuscript_quarto/
├── _quarto.yml                    # Project configuration
├── README.md                      # This file
├── references.bib                 # BibTeX bibliography
├── apa.csl                        # Citation style (APA 7th)
├── freeze-rebuild.qmd             # Main manuscript
├── appendix-a-data.qmd            # Appendix A: Data and Study Area
├── appendix-b-identification.qmd  # Appendix B: Identification Diagnostics
├── appendix-c-robustness.qmd      # Appendix C: Robustness Specifications
├── appendix-d-decomposition.qmd   # Appendix D: Price Decomposition
├── appendix-e-mechanisms.qmd      # Appendix E: Mechanism Analysis
├── code/
│   └── _common.py                 # Shared Python utilities
├── figures/ -> ../figures/        # Symlink to project figures
├── data/ -> ../data_work/diagnostics/  # Symlink to diagnostic CSVs
└── _output/                       # Generated outputs (gitignored)
```

## Building the Manuscript

### All Formats (HTML, PDF, DOCX)

```bash
cd manuscript_quarto
quarto render
```

Outputs will be in `_output/`:
- `freeze-rebuild.html` - Interactive HTML with code folding
- `freeze-rebuild.pdf` - Journal-ready PDF
- `freeze-rebuild.docx` - Word document for editing

### Single Format

```bash
# HTML only
quarto render freeze-rebuild.qmd --to html

# PDF only
quarto render freeze-rebuild.qmd --to pdf

# Word only
quarto render freeze-rebuild.qmd --to docx
```

### Preview (Live Reload)

```bash
quarto preview freeze-rebuild.qmd
```

Opens browser with live preview; auto-reloads on file changes.

## Configuration Details

### `_quarto.yml` Settings

| Setting | Value | Purpose |
|---------|-------|---------|
| `project.type` | `manuscript` | Enables manuscript features (notebooks, cross-refs) |
| `execute.freeze` | `auto` | Cache computational results |
| `execute.cache` | `true` | Cache code chunk outputs |
| `format.pdf.geometry` | `margin=1in` | Standard journal margins |
| `crossref.fig-prefix` | `"Figure"` | Cross-reference labels |

### Python Code Chunks

Tables are rendered live from diagnostic CSVs:

```python
#| label: tbl-example
#| tbl-cap: "Example Table"

import pandas as pd
import sys
sys.path.insert(0, 'code')
from _common import load_diagnostic

df = load_diagnostic("pretrends_ftest")
print(df.to_markdown(index=False))
```

### Figure References

Reference figures using Quarto cross-references:

```markdown
See @fig-event-study for the event study results.

![Event Study](figures/fig_event_study_sfha_300m.png){#fig-event-study}
```

### Citations

Use `[@citation_key]` format:

```markdown
Prior work has shown flood discounts of 4-12% [@beltran2018].
```

## Symlinks

The `figures/` and `data/` directories are symlinks to the main project:

```bash
# Recreate if needed
ln -sf ../figures figures
ln -sf ../data_work/diagnostics data
```

## Troubleshooting

### PDF Build Fails

1. Ensure LaTeX is installed: `quarto install tinytex`
2. Check for missing packages in the error log
3. Try `quarto render --to pdf --verbose` for details

### Code Chunk Errors

1. Activate project virtual environment first:
   ```bash
   source ../.venv/bin/activate
   ```
2. Ensure all Python packages are installed
3. Check that symlinks resolve correctly: `ls -la data/`

### Cross-Reference Not Found

1. Ensure the label exists: `{#fig-name}` or `#| label: tbl-name`
2. Run `quarto render` to rebuild cross-reference index
3. Check for typos in `@fig-name` references

## Output Formats

### HTML Features
- Interactive table of contents
- Code folding (click to expand)
- Responsive figures
- Hyperlinked cross-references

### PDF Features
- Letter paper, 1-inch margins
- Numbered sections
- Booktabs table formatting
- Keeps `.tex` source for debugging

### DOCX Features
- Standard Word formatting
- Editable tables
- Track changes compatible
- No code chunks visible

## Freezing Computations

To freeze all computations (for faster rebuilds):

```bash
quarto render --execute-freeze=true
```

To force re-execution of all code:

```bash
quarto render --execute-freeze=false
```

## Version Information

- Quarto: >= 1.4
- Python: >= 3.9
- pandas: >= 2.0
- This manuscript was created: December 2025
