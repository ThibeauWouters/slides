# CLAUDE.md — 2025/neural_priors_LVK_PE

**Title:** Incorporating neutron star physics into gravitational wave inference with neural priors  
**Venue:** LVK PE Call  
**Date:** November 12, 2025  
**Summary:** LVK PE call presentation of the neural priors paper: EOS-informed normalizing flow priors applied to GW170817, GW190425, GW230529 for improved PE and source classification.

## Slide Index

| Frame title | Content |
|---|---|
| Motivation | Why standard flat priors miss NS physics; information gain potential |
| Key idea | NF trained on EOS-conditioned (m₁, m₂, Λ₁, Λ₂) samples as prior |
| NS population models | Uniform, Gaussian, Double Gaussian NS mass distributions |
| EOS constraints | PSRs, PSRs+χEFT, PSRs+NICER Λ constraints used to build priors |
| Normalizing flows | NF architecture; training procedure |
| Construction of neural priors | Full construction pipeline: EOS samples → NF → prior |
| All neural priors | Overview of all 18 constructed neural prior variants |
| Setup | PE setup: jim + neural prior; event-by-event configuration |
| GW170817: Source classification | $P(\text{NS})$ for GW170817 components with neural priors |
| GW170817: Parameter constraints (Gaussian) | Posterior comparison with/without neural prior |
| GW170817: Discussion | Interpretation; information gain; Bayes factor |
| GW190425: Source classification | $P(\text{NS})$ for GW190425 |
| GW190425: Parameter constraints (Uniform) | Posterior comparison for GW190425 |
| GW190425: Discussion | Interpretation |
| GW230529: Source classification | $P(\text{NS})$ for GW230529 primary |
| GW230529: Parameter constraints (Gaussian) | Posterior comparison for GW230529 |
| GW230529: Discussion | Interpretation |
| Conclusion | Summary: neural priors tighten constraints and improve classification |
| Likelihood distributions | Appendix: likelihood curves |
| Information gain | Appendix: KL divergence / information gain table |
| More posteriors | Appendix: additional posterior comparisons |

## Paper Subdirectory

This talk has a companion paper at `paper/`:
- `paper/main.tex` — authoritative source for all technical content
- `paper/references.bib` — source for all citation entries (use this, not the web)
- `paper/Figures/` — source figures: corner plots, all-priors figure, schematic

**Key paper figures:**
- `paper/Figures/GW170817_corner_gaussian_bns.pdf`
- `paper/Figures/GW190425_corner_uniform_bns.pdf`
- `paper/Figures/GW230529_corner_gaussian_nsbh.pdf`
- `paper/Figures/bns_nsbh_all_populations_chirp_tilde.pdf` (all 18 priors)
- `Figures/Figure1.pdf` (construction schematic)

**Key paper sections:** Population models: §2.3.1; EOS constraints: §2.3.2; NF construction: §2.3.3; Results: §3

## Notes
- Uses biber (not bibtex)
- `jeffreysred1`–`jeffreysred5` colors for Bayes factor coding
- Slide title convention: "GW170817: Source classification" (event first)
