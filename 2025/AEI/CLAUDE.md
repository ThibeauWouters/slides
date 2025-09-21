# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Structure

This is a LaTeX Beamer presentation about binary neutron star analysis for AEI (Albert Einstein Institute). The project contains:

- `main.tex` - Main presentation file using Beamer class with Madrid theme
- `preamble.sty` - Custom style package with mathematical notation, physics commands, and color definitions
- `talk_overview.tex` - Overview slide content defining the talk structure
- `references.bib` - Bibliography file with scientific references
- `Figures/` - Directory containing PDF, JPG, and PNG images for the presentation
- `Inkscape/` - Directory containing SVG source files and their LaTeX exports (PDF + PDF_TEX)
- `gaussian.py` - Python script to generate Gaussian curve plot as SVG
- `gw_strain.py` - Python script using Bilby to generate gravitational wave strain plots

## Commands

### Building the presentation
```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

### Generating figures from Python scripts
```bash
python gaussian.py        # Generates Figures/gaussian.svg
python gw_strain.py       # Generates gravitational wave strain plots
```

### Working with Inkscape figures
- SVG files in `Inkscape/` are the source files
- Use Inkscape to export to PDF + LaTeX for inclusion in the presentation
- The PDF_TEX files contain LaTeX commands to overlay text properly

## LaTeX Structure

- The presentation uses a custom theme with blue color scheme
- Mathematical notation is heavily customized in `preamble.sty`
- Color commands for different physics topics (GW, BNS, etc.) are defined
- Bibliography uses scriptsize font for compact citations
- Python code highlighting is configured for listings package

## Key LaTeX Commands

- `\bns{text}` - Binary neutron star colored text
- `\mcmcgreen{text}`, `\mcmcred{text}` - MCMC analysis colors
- `\smallcite{ref}` - Smaller citation format
- `\incfig[width]{filename}` - Include Inkscape figures
- `\slideref{label}` - Reference full frames

## Dependencies

### LaTeX Packages
- Beamer with Madrid theme and circles inner theme
- Physics, mathematics, and graphics packages
- Hyperref for links with custom colors
- Listings for Python code highlighting

### Python Dependencies
- numpy, matplotlib for basic plotting
- scipy.stats for statistical functions
- bilby for gravitational wave analysis (in gw_strain.py)