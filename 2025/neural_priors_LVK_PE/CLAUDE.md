# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with this presentation.

## Repository Overview

This is a LaTeX Beamer presentation for the LVK Parameter Estimation (PE) call on November 12, 2025, presenting work on incorporating neutron star physics into gravitational wave inference using neural priors.

The presentation is based on the paper located in the `paper/` subdirectory.

## Repository Structure

- `main.tex` - Main presentation file (Beamer class, Madrid theme)
- `preamble.sty` - Custom style package with mathematical notation and commands
- `references.bib` - Bibliography file with scientific references
- `Figures/` - PDF, PNG, JPG images for the presentation
- `Inkscape/` - SVG source files and their LaTeX exports (PDF + PDF_TEX files)
- `paper/` - Subdirectory containing the full paper manuscript
  - `paper/main.tex` - Main paper manuscript (RevTeX 4.1 format)
  - `paper/references.bib` - Paper's bibliography (source for citations)
  - `paper/Figures/` - Paper figures (source for presentation figures)

## Building the Presentation

Standard LaTeX build process:
```bash
pdflatex main.tex
biber main
pdflatex main.tex
pdflatex main.tex
```

## Key Information Sources

### For Citations and References

**IMPORTANT**: When adding new citations to the presentation, the bibliography entries should be sourced from `paper/references.bib`.

To find a citation entry:
1. Search for the citation key in `paper/references.bib`
2. Copy the entire `@article{...}` entry
3. Add it to the presentation's `references.bib` file

Example citation keys from the paper:
- Population models: `LIGOScientific:2021qlt`, `Landry:2021hvl`, `Golomb:2024lds`, `Ozel:2016oaf`, `Alsing:2017bbc`, `Shao:2020bzt`
- Normalizing flows: `Kobyzev:2019ydm`, `Papamakarios:2019fms`
- Neural PE applications: `Dax:2021tsq`, `Dax:2022pxd`, `Williams:2021qyt`, `Wouters:2024oxj`
- JESTER code: `Wouters:2025zju`

### For Figures

Figures should be sourced from `paper/Figures/` when available:

**Corner plots (parameter constraints)**:
- GW170817: `paper/Figures/GW170817_corner_gaussian_bns.pdf`
- GW190425: `paper/Figures/GW190425_corner_uniform_bns.pdf`
- GW230529: `paper/Figures/GW230529_corner_gaussian_nsbh.pdf`

**Neural priors figure**:
- All 18 priors: `paper/Figures/bns_nsbh_all_populations_chirp_tilde.pdf` (Figure 2 in paper)

**Schematic figures**:
- Construction workflow: `Figures/Figure1.pdf` (Figure 1 in paper)

### For Content and Explanations

When uncertain about technical content, methodology, or results:
1. Search `paper/main.tex` for relevant sections
2. Key paper sections:
   - Population models: Lines 275-295 (Sec. 2.3.1)
   - EOS constraints: Lines 297-323 (Sec. 2.3.2)
   - Neural prior construction: Lines 325-344 (Sec. 2.3.3)
   - Normalizing flows: Lines 256-269 (Sec. 2.2)
   - Results for each event: Sec. 3 (lines 361-543)

## Slide Organization

### Structure
1. **Title slide** with custom background and logos (Utrecht, Nikhef)
2. **Table of Contents**
3. **Introduction** - Motivation for neural priors
4. **Methods**:
   - NS Population models (Uniform, Gaussian, Double Gaussian)
   - Tidal deformability constraints (PSRs, PSRs+χEFT, PSRs+NICER)
   - Normalizing flows explanation
   - Construction of neural priors
   - All neural priors overview
5. **Results**:
   - GW170817: Source classification + Parameter constraints + Discussion
   - GW190425: Source classification + Parameter constraints + Discussion
   - GW230529: Source classification + Parameter constraints + Discussion
6. **Conclusion**
7. **References**

### Slide Title Convention

Event results slides use the format: **"Event: Topic"** (e.g., "GW170817: Source classification", not "Source classification: GW170817")

## Custom LaTeX Commands

Defined in `preamble.sty` or main.tex:

**Mathematics**:
- `\Msun` - Solar mass symbol
- `\MTOV` - Maximum TOV mass
- `\jaxtwo{text}` - JAX-styled colored text
- `\red{text}`, `\blue{text}`, `\green{text}` - Colored text

**Social media**:
- `\github`, `\linkedin`, `\myemail` - Icon links for author info
- `\ghlink{username/repo}` - GitHub repository link with icon

**Inkscape figures**:
```latex
\incfig[width]{filename}  % Include Inkscape PDF_TEX figures (no extension)
```

**Color scheme**:
- `customblue` - Custom blue for title boxes
- `jeffreysred1` through `jeffreysred5` - Color coding for Bayes factors (from Seaborn's "rocket" palette)

## Figures from Other Presentations

Some slides reuse content from other presentations in parent directories:

**From BeNL_meeting**:
- EOS constraints slide with R14_table figure (copied to `Inkscape/`)

**From AEI**:
- Normalizing flow diagram (NF.pdf and NF.pdf_tex, copied to `Inkscape/`)

## Known Issues and Notes

1. **Bibliography warnings**: Some citations may show warnings during compilation if they haven't been added to `references.bib` yet. Always source them from `paper/references.bib`.

2. **Duplicate warnings**: The paper's references.bib has some duplicate entries (Koehn:2024set, Wong:2022xvh, Gabrie:2021tlu, Pang:2022rzc) - this is expected.

3. **Inkscape figures**: The `\incfig` command imports both the PDF and PDF_TEX files from the Inkscape directory. Both files must be present.

4. **PDF warnings**: "PDF inclusion: multiple pdfs with page group" warnings are harmless and occur with Inkscape figures in Beamer overlays.

## Git Status Note

The presentation is part of the larger slides repository. The main branch is `main`. Uncommitted changes may include:
- Updated figures copied from paper
- New citation entries in references.bib
- Modified content in main.tex

## Related Repositories

- Paper codebase: `paper/` subdirectory in this directory
- JESTER code: https://github.com/ThibeauWouters/jester
- EOS source classification: https://github.com/ThibeauWouters/eos_source_classification

## Common Tasks

### Adding a new citation
1. Find citation key in paper text (e.g., `\cite{AuthorName:2024xxx}`)
2. Search `paper/references.bib` for that key
3. Copy the entire `@article{...}` block
4. Append to presentation's `references.bib`
5. Recompile with biber

### Adding a new figure from the paper
1. Identify figure in `paper/Figures/`
2. Copy to presentation `Figures/` directory
3. Reference with `\includegraphics[width=...]{Figures/filename.pdf}`
4. Recompile

### Adding an Inkscape figure
1. Copy both `.pdf` and `.pdf_tex` files to `Inkscape/` directory
2. Use `\incfig[width]{filename}` (without extension)
3. Recompile

### Updating slide content
1. Check corresponding section in `paper/main.tex` for authoritative content
2. Adapt to slide format (concise bullet points)
3. Ensure citations are present in `references.bib`
4. Recompile and check output

## Compilation Tips

- Use `-interaction=nonstopmode` with pdflatex for non-interactive builds
- Run biber (not bibtex) for bibliography processing
- Expected output: 26 pages, ~5-6 MB PDF file
- Always run pdflatex → biber → pdflatex → pdflatex for clean builds
