# CLAUDE.md — 2025/AEI

**Title:** Towards GPU-accelerated multimessenger inference of neutron star mergers and dense matter physics  
**Venue:** AEI Seminar  
**Date:** 2025  
**Summary:** Long seminar talk covering the full research program: JAX/flowMC, jim (BNS PE), jester (EOS inference), neural priors, and source classification. The most comprehensive overview talk in the collection.

## Slide Index

| Frame title | Content |
|---|---|
| Neutron stars | NS basics: structure, mass–radius, EOS |
| Equation of state | EOS families; pressure–density relation; TOV equation |
| Multimessenger astrophysics: GW170817 | GW170817 overview; simultaneous GW + kilonova detection |
| Future GW detectors: Einstein Telescope | ET sensitivity; rate projections; science case |
| My research focus: why — how — what | Research overview connecting motivation, tools, and applications |
| Parameter estimation: Bayesian inference | Bayes' theorem; likelihood; posterior sampling |
| Parameter estimation: MCMC | MCMC in GW context; computational bottleneck for BNS |
| JAX | JAX transformations (JIT, grad, vmap); GPU acceleration |
| JAX — Function transformations | Detailed JAX transformation examples |
| Normalizing flows | NF architecture; expressiveness; use as proposal |
| flowMC | flowMC: local (MALA) + global (NF) sampler |
| Overview (jim section) | jim pipeline structure |
| ripple | ripple: JAX waveform generation |
| Jim | jim results: GW170817 & GW190425; validation |
| Einstein Telescope | ET signal properties; PE challenges at 3G |
| Overlapping signals | Simultaneous BNS signals in ET; joint PE |
| Open call | Discussion: open problems in 3G PE |
| Electromagnetic counterparts | Kilonova and GRB context for multimessenger PE |
| Overview (EOS section) | EOS inference pipeline structure |
| Equation of state inference — warmup | Toy example of EOS inference |
| Equation of state inference — parametrization | Piecewise polytrope / spectral EOS parametrization |
| Tidal deformability | Λ–EOS connection; how GWs measure NS structure |
| Equation of state | EOS constraint results |
| Jester | jester: differentiable TOV solver; auto-diff EOS inference |
| Anisotropy in neutron stars | Anisotropic pressure in NS matter |
| Auto-differentiable ODE solvers | Differentiable TOV integration via JAX |
| Neutron star data analysis loop | Full loop: GW PE → EOS inference → classification |
| Case study | Example: GW230529 through the full pipeline |
| Equation of state-informed priors | Motivation for EOS-informed priors in PE |
| Source classification | $P(\text{NS})$ classification framework |
| Neural priors | Neural prior construction: NF trained on EOS-conditioned posteriors |
| Application | Applying neural priors to GW events |
| GW170817 — classification | $P(\text{NS})$ for GW170817 components |
| GW170817 — parameter constraints | Tidal / mass posteriors with neural priors |
| GW190425 — classification | $P(\text{NS})$ for GW190425 components |
| GW190425 — parameter constraints | Posteriors with neural priors |
| GW230529 — classification | $P(\text{NS})$ for GW230529 primary |
| GW230529 — parameter constraints | Posteriors with neural priors |
| Conclusion | Full research summary |
| (Appendix: ET/CE projections) | ET and CE radius / Λ constraint projections |

## Key Files
- `gaussian.py` — generates `Figures/gaussian.svg` (Gaussian curve illustration)
- `gw_strain.py` — generates GW strain plots using bilby
- `talk_overview.tex` — overview slide content
- `Figures/` — NS mass–radius diagram, EOS curves, GW170817 corner plots, p-p plot, neural prior figures
- `Inkscape/` — research overview diagram, neutron star data analysis loop schematic
