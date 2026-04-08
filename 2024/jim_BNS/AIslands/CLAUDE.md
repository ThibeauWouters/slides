# CLAUDE.md — 2024/jim_BNS/AIslands

**Title:** JIM for fast parameter estimation of binary neutron star gravitational waves  
**Venue:** AIslands 2024  
**Date:** 2024  
**Summary:** Conference talk on jim: JAX + normalizing flows for fast BNS parameter estimation; validation results and environmental impact.

## Slide Index

| Frame title | Content |
|---|---|
| Parameter estimation | PE problem; why BNS is expensive (long signals, tidal effects) |
| Overview | Talk roadmap |
| Normalizing flows | NF architecture; change-of-variables; expressiveness |
| flowMC | flowMC algorithm: local (MALA) + global (NF) sampling |
| Results | GW170817 & GW190425 posteriors; JS divergences vs bilby |
| Environmental impact | GPU vs CPU energy/CO₂ comparison |
| Conclusion | Summary and key takeaways |
| Future work/points of discussion | Waveform extensions, EM pipeline, overlapping signals |
| References | Bibliography |
| Normalizing flow details | Appendix: NF architecture details |
| Stopping criterion | Appendix: convergence criterion used |
| Validation — Mismatch waveforms | Appendix: waveform mismatch validation |
| Validation — p-p plot | Appendix: probability–probability plot for coverage |
| Priors | Appendix: prior distributions used |
| GW170817 & GW190425: Jensen-Shannon divergences | Appendix: JS divergence table vs bilby |
| TaylorF2 | Appendix: TaylorF2 waveform results |
| IMRPhenomD_NRTidalv2 | Appendix: IMRPhenomD_NRTidalv2 waveform results |

## Key Figures
- p-p plot
- GW170817 / GW190425 corner plots
- Runtime and CO₂ comparison table
