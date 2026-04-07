# CLAUDE.md — 2026/UvA_journal_club

**Title:** GPU-accelerated multimessenger inference: applications and prospects
**Venue:** UvA Journal Club
**Date:** April 2026
**Summary:** 30-min talk for a mixed audience (master students + researchers) covering JAX for GPU-accelerated scientific computing, Jim for BNS parameter estimation, jester for EOS inference, and two results: ET projections and neutron star pressure anisotropy.

## Narrative

Four sections: (1) Introduction — NS physics, EOS, GW170817, tidal deformability, Bayesian inference, motivation for GPU acceleration. (2) Methods — JAX, SMC, Jim. (3) jester EOS inference overview. (4) Results — Einstein Telescope projections + pressure anisotropy in neutron stars.

## Slide Index

| Frame title | Content | Figures | Source |
|---|---|---|---|
| Title | Title slide; tintin_BNS_2 background; Utrecht + Nikhef logos | tintin_BNS_2.png, utrecht-university.png, Nikhef_logo-transparent.png | new |
| Table of Contents | TOC with sections | — | standard |
| **Section: Introduction** | | | |
| Neutron stars | NS basics: supernova remnants, mass/radius; \incfig NS_above_Berlin | NS_above_Berlin (Inkscape) | 2025/AEI |
| Equation of state | EOS probed by NS; Koehn:2024set | Koehn_EOS.jpg | 2025/AEI |
| Multimessenger astrophysics: GW170817 | GW170817 with \only<>: inspiral → merger → kilonova; \smallcite LIGOScientific:2017vwq | GW170817_inspiral/merger/KN.jpg | 2025/AEI |
| Tidal deformability | Λ–EOS connection; Λ=Λ(m,EOS); GW gives posterior; \incfig tidal | tidal (Inkscape) | 2025/AEI |
| Parameter estimation: Bayesian inference | Bayes' theorem; posterior/prior/likelihood/evidence; likelihood = bottleneck | — | 2025/AEI |
| Connecting multimessenger data to the EOS | EOS Bayesian inference; TOV equations; \incfig NS_likelihood | NS_likelihood (Inkscape) | 2025/AIslands adapted |
| EOS parametrization | Metamodel + speed-of-sound; >26 parameters; \incfig EOS_parametrizion | EOS_parametrizion (Inkscape) | 2025/AEI |
| Motivation: Why GPU-accelerated inference? | Studies bound by compute; GPU vs CPU; tcolorbox question: can we leverage GPU without compromising accuracy? | — | new |
| **Section: Methods** | | | |
| JAX | JAX: GPU acceleration; jit/grad/vmap; gradient descent example | jax.png | 2025/AEI |
| Sequential Monte Carlo (SMC) | SMC: particles through tempered distributions β∈[0,1]; parallelizable; \smallcite Williams:2025szm | SMC_diagram.jpg | 2026/lvk_pisa |
| Jim: GPU-accelerated GW parameter estimation | Jim (GW-JAX-Team/jim); re-analyzes GWTC-3/4; 2×–10× faster than bilby; \smallcite Wong:2023lgb,Wouters:2024oxj | jim_performance_now.jpg | 2025/AEI |
| **Section: jester: EOS inference** | | | |
| jester: JAX-accelerated EOS inference | jester (nuclear-multimessenger-astronomy/jester); EOS models (MM+CSE, spectral); samplers (NS, SMC); likelihoods; \smallcite Wouters:2025zju | JESTER_logo.png | 2025/AEI + 2026/lvk_pisa |
| **Section: Results — ET projections** | | | |
| Einstein Telescope (lambda14) | 100 loudest BNS events, 1 year; ET_ET_lambda14_comparison; \smallcite ET:2019dnz, Branchesi:2023mws | ET_ET_lambda14_comparison.pdf | 2026/lvk_pisa |
| Einstein Telescope (R14) | ET_ET_r14_comparison; different EOS parametrizations → different R14 uncertainties; samplers consistent | ET_ET_r14_comparison.pdf | 2026/lvk_pisa |
| Einstein Telescope: runtimes | Runtime table (1 NVIDIA H100 GPU); scalable inference enables projection studies | ET_runtime_table.tex | 2026/lvk_pisa |
| **Section: Results — Pressure anisotropy** | | | |
| Pressure anisotropy in neutron stars | Isotropic TOV vs anisotropic (p_r≠p_t); γ model σ=γ·2mp_r/r; mechanisms; key question: measure γ≠0? | pressure_anistropy_demo_art.pdf | new (Pang:2025fes) |
| Measuring anisotropy with jester | Modified TOV + tidal with γ; hierarchical inference across stars; GW+NICER+nuclear; model selection | — | new (Pang:2025fes) |
| Anisotropy results | individual_gamma_hist; preference for negative anisotropy; driven by PSR J0740+6620; Bayes factor ≥3:1; \smallcite Pang:2025fes | individual_gamma_hist_noGaussian.pdf | new (Pang:2025fes) |
| **Section: Conclusion** | | | |
| Conclusion | JAX / Jim / jester / Applications (ET projections, anisotropy) | — | new |
| Thanks | Thanks slide; tintin_BNS_2 background | tintin_BNS_2.png | new |
| References | printbibliography | — | standard |

## Key Figures

Figures are sourced from multiple directories via `\graphicspath`:
- `Figures/` — local figures (utrecht-university.png, Nikhef_logo-transparent.png, tintin_BNS_2.png)
- `resources/` — anisotropy paper figures (pressure_anistropy_demo_art.pdf, individual_gamma_hist_noGaussian.pdf)
- `../../2025/AEI/Figures/` — Koehn_EOS.jpg, GW170817_*.jpg, jax.png, jim_performance_now.jpg
- `../../2026/lvk_pisa_jester/Figures/` — JESTER_logo.png, SMC_diagram.jpg
- `../../2026/lvk_pisa_jester/` — ET_ET_lambda14_comparison.pdf, ET_ET_r14_comparison.pdf, ET_runtime_table.tex

Inkscape figures (`\incfig`) point to `../../2025/AEI/Inkscape/`:
- `NS_above_Berlin` — NS cross-section schematic
- `NS_likelihood` — EOS inference pipeline diagram
- `tidal` — tidal deformability schematic
- `EOS_parametrizion` — EOS parametrization diagram

## Building

```bash
cd 2026/UvA_journal_club/
pdflatex main.tex && biber main && pdflatex main.tex && pdflatex main.tex
# or
latexmk -pdf main.tex
```
