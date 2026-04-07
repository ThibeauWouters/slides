# CLAUDE.md — 2026/neural_priors_GRANDMA

**Title:** Incorporating neutron star physics into gravitational wave inference with neural priors  
**Venue:** GRANDMA Call  
**Date:** January 15, 2026  
**Summary:** Extended version of the neural priors talk for the GRANDMA EM follow-up network; includes Bayesian inference warmup and motivation slides not present in shorter versions.

## Slide Index

| Frame title | Content |
|---|---|
| Warm-up example: Motivation | Coin flip / toy Bayesian inference to introduce priors intuitively |
| Bayesian inference | Bayes' theorem; prior, likelihood, posterior; GW context |
| Motivation | Why physics-informed priors matter for NS mergers; EM follow-up connection |
| Tidal deformability | Λ definition; GW measurement; EOS sensitivity |
| Key idea | NF trained on EOS-conditioned (m₁, m₂, Λ₁, Λ₂) samples as prior |
| NS population models | Uniform, Gaussian, Double Gaussian mass distributions |
| Equation of state constraints | PSRs, PSRs+χEFT, PSRs+NICER Λ constraints |
| Normalizing flows | NF architecture; training |
| Construction of neural priors | Pipeline: EOS samples → NF → prior |
| All neural priors | All 18 neural prior variants overview |
| Setup | PE setup with neural priors on O3 events |
| GW170817: Source classification | $P(\text{NS})$ — critical for EM targeting |
| GW170817: Parameter constraints (Gaussian) | Posterior comparison |
| GW170817: Discussion | Interpretation; Bayes factor |
| GW190425: Source classification | $P(\text{NS})$ for GW190425 |
| GW190425: Parameter constraints (Uniform) | Posterior comparison |
| GW190425: Discussion | Interpretation |
| GW230529: Source classification | $P(\text{NS})$ for GW230529 primary |
| GW230529: Parameter constraints (Gaussian) | Posterior comparison |
| GW230529: Discussion | Interpretation |
| Conclusion | Summary |
| Likelihood distributions | Appendix |
| Information gain | Appendix: KL divergence / information gain |
| More posteriors | Appendix: additional posteriors |

## Notes
- Longer than LVK versions — includes Bayesian inference warmup slides
- Audience: EM follow-up community (GRANDMA network)
- Uses biber (not bibtex)
