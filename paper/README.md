# PASSAGE Paper

LaTeX source for the PASSAGE dataset paper (NeurIPS 2026 style).

## Structure

```
paper/
├── main.tex           # Main document
├── references.bib     # Bibliography
├── neurips_2026.sty   # NeurIPS style (placeholder - download official)
├── Makefile           # Build automation
├── figures/           # Figures directory (create as needed)
└── README.md          # This file
```

## Building

### Prerequisites

```bash
# Ubuntu/Debian
sudo apt install texlive-full latexmk biber

# Or minimal install
sudo apt install texlive-latex-recommended texlive-latex-extra \
                 texlive-fonts-recommended latexmk biber
```

### Build Commands

```bash
# Full build with latexmk (recommended)
make

# Quick single-pass build (for drafts)
make quick

# Manual build (without latexmk)
make manual

# Watch mode (auto-rebuild on changes)
make watch

# Clean auxiliary files
make clean

# Clean everything including PDF
make distclean
```

### Direct Commands

```bash
# With latexmk
latexmk -pdf main.tex

# Manual
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

## NeurIPS Style

**Important**: The included `neurips_2026.sty` is a placeholder. For submission:

1. Download the official style from the [NeurIPS website](https://neurips.cc/)
2. Replace `neurips_2026.sty` with the official version
3. Follow conference formatting guidelines

## Options

In `main.tex`, change the document class option:

```latex
% For anonymous submission
\usepackage{neurips_2026}

% For preprint (non-anonymous)
\usepackage[preprint]{neurips_2026}

% For camera-ready (final)
\usepackage[final]{neurips_2026}
```

## Figures

Place figures in the `figures/` directory:

```latex
\includegraphics[width=0.8\linewidth]{figures/sample.pdf}
```

Recommended formats:
- **Plots**: PDF (vector, from matplotlib)
- **Diagrams**: PDF (from Inkscape/TikZ)
- **Photos**: PNG/JPG (high resolution)

## Word Count

```bash
make wordcount
# Or directly:
texcount -inc -total main.tex
```

## Tips

1. Use `\dataset{}` macro for consistent formatting of "PASSAGE"
2. Keep figures as PDF for best quality
3. Run `make lint` to check for common LaTeX issues
4. Use `make spellcheck` before submission
