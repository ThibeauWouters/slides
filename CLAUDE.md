# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Important instructions

- When recycling old slides, copy over the figures and files needed to make the figures to the new directory.
- When reusing citations, copy over the exact bibtex (do not make up bibtex entries yourself!)

## Repository Overview

LaTeX Beamer presentations for academic talks on gravitational wave physics, binary neutron star mergers, equation of state inference, and related astrophysics. Each year has its own directory (`2023/`, `2024/`, `2025/`, `2026/`), and each talk lives in its own subdirectory within the year.

Every talk subdirectory has a `CLAUDE.md` describing the talk's purpose, topics, and a slide-by-slide breakdown — read those files when looking for specific slides to reuse.

## Repository Structure

```
YYYY/talk_name/
  main.tex          # Main presentation
  preamble.sty      # Custom style/commands
  references.bib    # Bibliography
  Figures/          # PDF, PNG, JPG images
  Inkscape/         # SVG sources + their PDF+LaTeX exports
  code/             # Python scripts to generate figures
  CLAUDE.md         # Talk description and slide index (see below)
```

## Finding and Reusing Slides

Each subdirectory CLAUDE.md contains:
- Talk title, venue, date, and 1-sentence summary
- Slide-by-slide breakdown: what each frame covers and what figures/content it uses

**To build a new talk:** describe the desired narrative, then search subdirectory CLAUDE.md files to find individual frames that cover the needed topics. Copy the relevant `\begin{frame}...\end{frame}` blocks, figures, and bibliography entries.

**Common reusable topics across talks:**
- Parameter estimation / Bayesian inference intro → `2025/AEI`, `2026/neural_priors_GRANDMA`
- Normalizing flows / flowMC → `2024/jim_BNS/junior_colloquium_15_04_2024`, `2025/ET_symposium`
- JAX introduction → `2024/jim_BNS/lorentz`, `2025/AEI`
- EOS parametrization and TOV → `2025/jester_extreme_matter`, `2026/et_div6_roadmap`
- Neural priors construction → `2025/neural_priors_LVK_PE`, `2026/neural_priors_GRANDMA`
- GW170817 / GW190425 / GW230529 results → `2025/BeNL_meeting`, `2025/neural_priors_LVK_EM`
- Einstein Telescope projections → `2025/ET_div6`, `2026/lvk_pisa_jester`
- Source classification → `2025/AIslands`, `2025/BeNL_meeting`
- jester EOS solver → `2026/lvk_pisa_jester`, `2026/et_div6_roadmap`

## Slide Style Guide

This is critical: new slides must match the existing style. Do **not** introduce Claude-style verbosity.

**Do:**
- One main point per slide
- 2–4 short bullet points maximum per slide (often zero — let figures speak)
- Figures are the primary content; text is supporting
- Short, noun-phrase bullets (not full sentences)
- Use `\blue{...}`, `\red{...}`, `\green{...}` to highlight key terms inline
- Use `\smallcite{ref}` for citations, placed inline or at frame bottom
- Keep frame titles short and descriptive (3–6 words)
- Use `\pause` sparingly for reveals; prefer presenting a complete slide

**Do not:**
- Write paragraph-length text on any slide
- Use more than 4–5 bullet points on a single frame
- Explain everything — assume a physics-literate audience
- Add motivational filler ("In this talk, we will show that...")
- Use `\textbf{...}` for emphasis when color commands exist

## Building Presentations

```bash
cd YYYY/talk_name/
pdflatex main.tex && biber main && pdflatex main.tex && pdflatex main.tex
# or
latexmk -pdf main.tex
```

Use `bibtex` instead of `biber` for older talks that use plain BibTeX.

## Key Custom Commands (preamble.sty)

- **Colors**: `\red{text}`, `\blue{text}`, `\green{text}`, `\grey{text}`
- **Citations**: `\smallcite{ref}`
- **Slide link**: `\slideref{label}` — hyperlinks to another frame
- **Math**: `\diff`, `\N`, `\Z`, `\R`, `\C`
- **Inline comment**: `\comment{text}`
- **Figures**: `\incfig[width]{name}` — includes `Inkscape/name.pdf_tex`
- **Social**: `\github`, `\linkedin`, `\ghlink{user/repo}` in title slide

## Beamer Theme

Madrid theme + circles inner theme, custom blue (`#4c4cfe`), no navigation symbols, `appendixnumberbeamer` for appendix slide numbering.

## Figures

Python scripts in `code/` use `numpy`, `matplotlib`, `scipy`, `jax`. Beamer blue for plot colors: `#4c4cfe`. Inkscape SVGs are exported as PDF+LaTeX pairs (`name.pdf` + `name.pdf_tex`) and included via `\incfig`.

## Creating a New Talk

1. Copy `template/` to `YYYY/talk_name/`
2. Update title, author, date, social links in `main.tex`
3. Build the narrative by finding relevant frames in existing talk CLAUDE.md files
4. Copy and adapt frames, figures, and bib entries
5. Create a `CLAUDE.md` in the new directory documenting the slide index
