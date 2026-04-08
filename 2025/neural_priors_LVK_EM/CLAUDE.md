# CLAUDE.md — 2025/neural_priors_LVK_EM

**Title:** Incorporating neutron star physics into gravitational wave inference with neural priors  
**Venue:** LVK EM Call  
**Date:** November 17, 2025  
**Summary:** LVK EM call version of the neural priors talk; same content as neural_priors_LVK_PE but pitched at the EM follow-up community. Emphasizes source classification for EM targeting.

## Slide Index

| Frame title | Content |
|---|---|
| Motivation | Why standard flat priors miss NS physics; EM follow-up motivation |
| Key idea | NF trained on EOS-conditioned (m₁, m₂, Λ₁, Λ₂) samples as prior |
| NS population models | Uniform, Gaussian, Double Gaussian NS mass distributions |
| EOS constraints | PSRs, PSRs+χEFT, PSRs+NICER Λ constraints |
| Normalizing flows | NF architecture |
| Construction of neural priors | Full construction pipeline |
| All neural priors | All 18 neural prior variants |
| Setup | PE setup with neural priors |
| GW170817: Source classification | $P(\text{NS})$ — relevant for EM follow-up decisions |
| GW170817: Parameter constraints (Gaussian) | Posterior comparison |
| GW170817: Discussion | Interpretation |
| GW190425: Source classification | $P(\text{NS})$ for GW190425 |
| GW190425: Parameter constraints (Uniform) | Posterior comparison |
| GW190425: Discussion | Interpretation |
| GW230529: Source classification | $P(\text{NS})$ for GW230529 primary |
| GW230529: Parameter constraints (Gaussian) | Posterior comparison |
| GW230529: Discussion | Interpretation |
| Conclusion | Summary |
| Likelihood distributions | Appendix |
| Information gain | Appendix: KL divergence table |
| More posteriors | Appendix |

## Paper Subdirectory

Same companion paper as `2025/neural_priors_LVK_PE/paper/` — see that CLAUDE.md for figure paths and paper section references.

## Notes
- Uses biber (not bibtex)
- Shares figures and structure with `2025/neural_priors_LVK_PE`
