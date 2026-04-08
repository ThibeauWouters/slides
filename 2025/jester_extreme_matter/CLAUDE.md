# CLAUDE.md — 2025/jester_extreme_matter

**Title:** Leveraging Differentiable Programming in the Inverse Problem of Neutron Stars  
**Venue:** Extreme Matter Call  
**Date:** March 17, 2025  
**Summary:** Methods talk presenting jester: differentiable TOV solver enabling both Bayesian EOS inference and gradient-based recovery of nuclear empirical parameters (NEPs).

## Slide Index

| Frame title | Content |
|---|---|
| Introduction | Inverse problem: infer EOS from NS observables; why differentiability helps |
| Inverting neutron stars with differentiable programming | Overview: jester framework; JAX ODE solver; two use modes |
| Methods — EOS parametrization (1) | Meta-model EOS: nuclear empirical parameters ($E_{\text{sym}}$, $L_{\text{sym}}$, etc.) |
| Methods — EOS parametrization (2) | Connecting NEPs to pressure–density; allowed EOS space |
| Results — Bayesian inference | EOS posterior from GW + NICER data; corner plots |
| Methods — Gradient-based optimization | Auto-diff TOV: recover NEPs via gradient descent |
| Results — Recovery of nuclear empirical parameters | Recovered NEP values vs ground truth; convergence |
| Conclusion | Summary: two-in-one tool for EOS inverse problem |
| More validation results | Appendix: additional injection recovery tests |

## Key Figures
- NEP corner plot (Bayesian posterior)
- Gradient descent convergence curve
- EOS pressure–density bands
