# CLAUDE.md — 2024/jim_BNS/lvk_sept_2024

**Title:** Fast parameter estimation of gravitational waves from binary neutron stars with JAX and normalizing flows  
**Venue:** LVK Barcelona Meeting  
**Date:** September 2024  
**Summary:** LVK community presentation of jim; covers full validation suite and both waveform models (TaylorF2 and IMRPhenomD_NRTidalv2).

## Slide Index

| Frame title | Content |
|---|---|
| Parameter estimation | BNS PE challenge; MCMC cost |
| Overview | Talk structure |
| JAX | JAX acceleration rationale |
| Normalizing flows | NF architecture |
| flowMC | flowMC algorithm overview |
| Results | GW170817 & GW190425 posteriors; JS divergences |
| Environmental impact | CO₂ / energy footprint comparison |
| Conclusion | Key results summary |
| References | Bibliography |
| Normalizing flow details | Appendix: NF architecture details |
| Stopping criterion | Appendix: convergence criterion |
| Validation — Mismatch waveforms | Appendix: waveform mismatch |
| Validation — p-p plot | Appendix: p-p coverage |
| Priors | Appendix: prior table |
| GW170817 & GW190425: Jensen-Shannon divergences | Appendix: JS divergence table |
| TaylorF2 | Appendix: TaylorF2 results |
| IMRPhenomD_NRTidalv2 | Appendix: IMRPhenomD_NRTidalv2 results |

## Key Figures
- p-p plot
- GW170817 / GW190425 corner plots
- Runtime + CO₂ table
- Waveform model comparison
