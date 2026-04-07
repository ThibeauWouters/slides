# CLAUDE.md — 2026/lvk_pisa_jester

**Title:** jester v0.2.0: Scalable inference of the equation of state with multimessenger data  
**Venue:** LVK Meeting, Pisa  
**Date:** March 2026  
**Summary:** LVK meeting talk presenting jester v0.2.0: SMC sampler, sampler systematics study, and future observing run EOS projections.

## Slide Index

| Frame title | Content |
|---|---|
| Motivation | Why scalable EOS inference matters; current limitations |
| jester: JAX-accelerated EOS inference | jester overview: differentiable TOV + SMC; GitHub link |
| Sequential Monte Carlo (SMC) | SMC algorithm; advantages over nested sampling for EOS inference |
| Sampler systematics: GW170817 | SMC vs nested sampling comparison on GW170817 |
| Future observing runs | Projected EOS constraints from O4/O5 BNS detections |
| Looking ahead | Roadmap: multimessenger (GW + NICER + nuclear) joint inference |
| Conclusion | Summary of jester v0.2.0 capabilities |
| $R_{1.4}$ for GW170817 inferences | Appendix: $R_{1.4}$ posterior comparison across sampler/parametrization choices |

## Key Figures
- `ET_ET_lambda14_comparison.pdf` — Λ₁.₄ comparison (ET-ET baseline)
- `ET_ET_r14_comparison.pdf` — R₁.₄ comparison (ET-ET baseline)
- `ET_cornerplot_smc_vs_ns.pdf` — SMC vs nested sampling corner plot
- `SS_lambda14_comparison.pdf` — Λ₁.₄ comparison (Silver Sword variant)
- `SS_r14_comparison.pdf` — R₁.₄ comparison (Silver Sword variant)
- `ET_cornerplot_nep_*.pdf` — NEP corner plots
- `ET_cs2_density.pdf` — Speed of sound vs density
- `ET_pressure_density*.pdf` — Pressure–density EOS bands
- `r14_table.tex` — R₁.₄ summary table
- `runtime_table.tex` — Runtime comparison table
