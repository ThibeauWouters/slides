# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is a repository containing LaTeX Beamer presentations for academic talks, primarily focused on gravitational wave physics, binary neutron stars, and related astrophysics topics. Presentations are organized by year (2023/, 2024/, 2025/) with each presentation in its own subdirectory.

## Repository Structure

- **template/** - Base template directory containing reference files for new presentations
  - `main.tex` - Template Beamer presentation structure
  - `preamble.sty` - Shared LaTeX style package with custom commands and packages
  - `references.bib` - Bibliography template
- **YYYY/** (2023, 2024, 2025) - Presentations organized by year
  - Each presentation lives in its own subdirectory
  - Standard structure per presentation:
    - `main.tex` - Main presentation file
    - `preamble.sty` - Custom style package (often copied/modified from template)
    - `references.bib` - Bibliography for that specific talk
    - `Figures/` - PDF, PNG, JPG images
    - `Inkscape/` - SVG source files and their LaTeX exports (PDF + PDF_TEX files)
    - `code/` - Python scripts for generating plots/figures
    - `talk_overview.tex` - Overview slide content (optional)

## Building Presentations

### Standard LaTeX build process
Navigate to the presentation directory, then run:
```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

Or use `biber` if the presentation uses biblatex:
```bash
pdflatex main.tex
biber main
pdflatex main.tex
pdflatex main.tex
```

### Using latexmk (if available)
```bash
latexmk -pdf main.tex
```

## Working with Figures

### Python-generated figures
Python scripts in the `code/` directory generate plots as SVG or other formats. Common pattern:
```bash
cd 2025/AEI/
python gaussian.py          # Generates Figures/gaussian.svg
```

Python scripts typically use:
- `numpy`, `matplotlib` for plotting
- `scipy` for statistical functions
- Beamer blue color: `#4c4cfe` for consistency

### Inkscape figures
SVG files in the `Inkscape/` directory are source files created with Inkscape. Workflow:
1. Create/edit SVG in Inkscape (`Inkscape/*.svg`)
2. Export to PDF + LaTeX using Inkscape's "PDF + LaTeX" export option
3. This creates two files: `filename.pdf` and `filename.pdf_tex`
4. Include in LaTeX using the `\incfig` command

The `\incfig` command is defined in presentations as:
```latex
\newcommand{\incfig}[2][0.75\textwidth]{%
    \def\svgwidth{\columnwidth}
    \resizebox{#1}{!}{\import{Inkscape figs/}{#2.pdf_tex}}
}
```

Usage: `\incfig[0.90\textwidth]{figure_name}` (without .pdf_tex extension)

## LaTeX Structure and Key Commands

### Custom Beamer setup
- Theme: Madrid with circles inner theme
- Custom blue color scheme
- Navigation symbols removed
- Appendix support with `appendixnumberbeamer` package
- Social media icons using `fontawesome` package

### Common custom commands in preamble.sty
- **Math sets**: `\N`, `\Z`, `\Q`, `\R`, `\C` for number sets
- **Colors**: `\red{text}`, `\blue{text}`, `\green{text}`, `\black{text}`, `\grey{text}`
- **Differential**: `\diff` for differential d
- **Citations**: `\smallcite{ref}` for smaller citation format
- **Slide references**: `\slideref{label}` for hyperlinking full frames
- **Comment**: `\comment{text}` for small inline comments

### Social media commands
Defined in main.tex files:
- `\github`, `\linkedin`, `\twitter` (or `\myemail`) for icon links in title slide
- `\ghlink{username/repo}` for GitHub repository links

### Python code highlighting
The template includes custom Python syntax highlighting using the `listings` package with custom colors (deepblue, deepred, deepgreen).

## Git Workflow

The `.gitignore` file excludes LaTeX build artifacts except:
- `main.pdf` (included)
- `main.tex` (included)

All other `main.*` files (aux, log, bbl, etc.) are ignored.

## Dependencies

### LaTeX packages
Core packages used across presentations:
- **Beamer**: Madrid theme, appendixnumberbeamer
- **Bibliography**: biblatex with biber backend (or bibtex)
- **Math**: amsmath, amssymb, physics, mathtools
- **Graphics**: graphicx, tikz, svg, import, xifthen, pdfpages, transparent
- **Figures**: subcaption, caption, mdframed
- **Code**: listings (for Python syntax highlighting)
- **Links**: hyperref with custom colors
- **Fonts**: fontawesome, lmodern
- **Tables**: multirow, multicol, tabularx, booktabs

### Python dependencies
Common packages used in figure generation scripts:
- numpy
- matplotlib
- scipy (scipy.stats)
- jax (for some advanced physics calculations)
- bilby (gravitational wave analysis, in some presentations)

## Creating New Presentations

1. Copy the `template/` directory to `YYYY/presentation_name/`
2. Update title, author, date in `main.tex`
3. Modify `preamble.sty` if custom commands are needed
4. Add figures to `Figures/` directory
5. Create Inkscape figures in `Inkscape/` subdirectory
6. Add Python plotting scripts to `code/` directory if needed
7. Update `references.bib` with citations
8. Build using the standard LaTeX build process above
