# CLAUDE.md — 2023/ML4GW_Utrecht

**Title:** Accelerating gravitational wave parameter estimation with normalizing flows  
**Venue:** ML4GW Workshop, Utrecht  
**Date:** December 8, 2023  
**Summary:** Introductory talk presenting flowMC and normalizing flows for fast BNS parameter estimation; early results with GW170817.

## Slide Index

| Frame title | Content |
|---|---|
| Parameter estimation | Motivation: why PE is expensive; MCMC bottleneck for BNS signals |
| Overview | Roadmap of the talk: JAX → NFs → flowMC → results |
| JAX? | What JAX is; JIT, grad, vmap transformations; why it matters for PE |
| flowMC — local sampling | MALA/HMC local sampler within flowMC |
| flowMC — normalizing flows | NF architecture used as a global proposal in flowMC |
| flowMC — global sampling | How the NF global move integrates with local sampling |
| flowMC — complete algorithm | Full flowMC algorithm combining local + global steps |
| Results | GW170817 posteriors compared to bilby; runtime comparison |
| Future work | Planned extensions: more waveforms, EM counterparts |
| Conclusion | Summary of contributions and outlook |
| Global acceptance for the NF | Appendix: NF global acceptance rate during sampling |
| (GitHub link) | Appendix: link to flowMC/jim repository |

## Key Figures
- `Figures/`: GW170817 posterior corner plots, runtime comparison table
- Normalizing flow architecture diagram
