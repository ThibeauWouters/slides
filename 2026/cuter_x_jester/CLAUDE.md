# CLAUDE.md — 2026/cuter_jester

**Title:** `jester`: JAX-accelerated equation of state inference and TOV solvers  
**Venue:** CUTER collaboration meeting  
**Date:** 09/04/2026  
**Summary:** Talk motivating jester + CUTER integration for an EOS-expert audience: jester provides scalable GPU-accelerated inference, CUTER provides microscopically motivated nuclear EOS. Covers jester capabilities (Bayesian inference, SMC, gradient-based NEP recovery, ET projections) with recycled slides from lvk_pisa_jester and jester_extreme_matter.

## Slide Index

| Frame title | Content |
|---|---|
| Title | tintin background, Utrecht + Nikhef logos |
| Motivation | CUTER (nuclear physics) + jester (data analysis) → collaboration; missing pieces on each side |
| Inverse problem of neutron stars | TOV bottleneck; Koehn EOS figure |
| `jester`: JAX-accelerated EOS inference | jester overview: differentiable TOV + samplers + EOS models; GitHub link |
| EOS parametrization (1): metamodel | Metamodel Taylor expansion; NEPs; low-density regime |
| EOS parametrization (2): high density | cs2 grid points; linear interpolation above n_break |
| Bayesian EOS inference | GW+NICER+heavy PSR constraints; R14 table; scaling plot |
| Sequential Monte Carlo (SMC) | SMC algorithm; evidence; GPU parallelization; SMC diagram |
| Sampler systematics: GW170817 | R1.4 and runtime tables (SMC vs NS, MM+CSE vs spectral) |
| Gradient-based optimization | Given R(M)/Λ(M): recover NEPs; loss function; Adam optimizer |
| Recovery of nuclear empirical parameters | NEP recovery injection study; money_plot figure |
| Future observing runs | ET simulated study: 100 BNSs, CoBa catalogue, Fabian Gittins PE |
| ET projections: Results | Λ1.4 and R1.4 comparison plots for ET |
| ET projections: Summary | R1.4 and runtime tables for ET |
| Looking ahead | jester+CUTER integration roadmap; other extensions |
| Conclusion | Summary of jester capabilities; CUTER collaboration pitch |
| Thanks | tintin background |

## Key Figures
- `Figures/Koehn_EOS.jpg` — EOS inverse problem overview
- `Figures/R14_table.jpg` — R1.4 results table image
- `Figures/scaling_plot.pdf` — scaling with number of parameters
- `Figures/cs2_sketch.pdf` — cs2(n) EOS parametrization sketch
- `Figures/showcase_variational_inference.pdf` — gradient descent visualization
- `Figures/money_plot.pdf` — NEP recovery results
- `Figures/SMC_diagram.jpg` — SMC particle evolution diagram
- `ET_ET_lambda14_comparison.pdf` — ET Λ1.4 posterior comparison
- `ET_ET_r14_comparison.pdf` — ET R1.4 posterior comparison
- `SS_r14_table_radio.tex`, `SS_runtime_table_radio.tex` — GW170817 tables
- `ET_r14_table.tex`, `ET_runtime_table.tex` — ET projection tables

## Source talks
- `2026/lvk_pisa_jester` — SMC, sampler systematics, ET projections, jester overview
- `2025/jester_extreme_matter` — metamodel/NEPs, gradient-based optimization, Bayesian inference
