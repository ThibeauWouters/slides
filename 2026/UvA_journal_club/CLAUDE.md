# CLAUDE.md — 2026/UvA_journal_club

**Title:** GPU-accelerated multimessenger inference: applications and prospects
**Venue:** UvA Journal Club
**Date:** April 2026
**Summary:** 30-min talk for a mixed audience (master students + researchers) covering JAX for GPU-accelerated scientific computing, Jim for BNS parameter estimation, jester for EOS inference, and two results: sampler systematics for ET projections and neutron star pressure anisotropy.

## Narrative

Motivation: scientific computing needs GPU acceleration (like deep learning). JAX enables this in Python. Application: gravitational waves from BNS + EOS inference.

## Slide Index

| Frame title | Content | Source |
|---|---|---|
| Title | Title slide with tintin background | new |
| The computational challenge | GPU revolution in AI; scientific computing lags behind; this talk bridges the gap | new |
| Neutron stars | NS basics: supernova remnants, mass-radius, structure | 2025/AEI |
| Equation of state | EOS and NS observations; Koehn_EOS figure | 2025/AEI |
| Multimessenger astrophysics: GW170817 | GW170817: inspiral → merger → kilonova | 2025/AEI |
| Tidal deformability | Λ–EOS connection; tidal figure | 2025/AEI |
| Parameter estimation: Bayesian inference | Bayes' theorem; likelihood bottleneck | 2025/AEI |
| Connecting multimessenger data to the EOS | Bayesian inference for EOS; TOV equations; NS_likelihood figure | 2025/AIslands adapted |
| Why scalable inference? | Why/how/what: ET, systematics, JAX, ML | 2025/AEI adapted |
| JAX | JAX: GPU acceleration, jit/grad/vmap | 2025/AEI |
| flowMC | Normalizing flow MCMC; flowMCOverview2 figure | 2025/AEI |
| Jim: BNS parameter estimation | Jim: ripple + flowMC; runtime table; ESS figure | 2025/AEI |
| EOS parametrization | Metamodel + speed-of-sound; EOS_parametrizion figure | 2025/AEI |
| jester: differentiable EOS inference | jester overview; GPU speedup; multiple samplers | 2025/AEI + 2026/lvk_pisa |
| Sequential Monte Carlo (SMC) | SMC algorithm; SMC_diagram figure | 2026/lvk_pisa |
| Systematics: GW170817 analysis | SMC vs NS corner plot; consistency check | 2026/lvk_pisa |
| Einstein Telescope projections | 100 BNS events; R14 comparison figure | 2026/lvk_pisa |
| ET projections: EOS model comparison | Lambda14 comparison; different EOS models | 2026/lvk_pisa |
| Pressure anisotropy in neutron stars | What is anisotropy; physical mechanisms; γ model; demo_art figure | new (Pang:2025fes) |
| Measuring anisotropy with jester | Modified TOV; hierarchical inference; model selection | new (Pang:2025fes) |
| Anisotropy results | Bayes factor ≥ 3:1; negative anisotropy preference; individual_gamma_hist figure | new (Pang:2025fes) |
| Conclusion | Summary of all tools and results | new |
| Thanks | Thanks slide with tintin background | new |

## Key Figures

Figures are sourced from multiple directories via `\graphicspath`:
- `resources/` — anisotropy paper figures (pressure_anistropy_demo_art.pdf, individual_gamma_hist_noGaussian.pdf, gamma_mu_sigma_posterior_corner.pdf)
- `../../2025/AEI/Figures/` — NS/JAX/Jim figures
- `../../2026/lvk_pisa_jester/Figures/` — JESTER_logo, SMC_diagram
- `../../2026/lvk_pisa_jester/` — ET comparison PDFs

Inkscape figures use `\incfig` pointing to `../../2025/AEI/Inkscape/`:
- `NS_above_Berlin` — NS cross-section
- `flowMCOverview2` — flowMC diagram
- `EOS_parametrizion` — EOS parametrization
- `tidal` — tidal deformability schematic

## Building

```bash
cd 2026/UvA_journal_club/
pdflatex main.tex && biber main && pdflatex main.tex && pdflatex main.tex
# or
latexmk -pdf main.tex
```
