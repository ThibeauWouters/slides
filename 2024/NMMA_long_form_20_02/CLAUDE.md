# CLAUDE.md — 2024/NMMA_long_form_20_02

**Title:** NMMA long form update: JAX  
**Venue:** NMMA Long Form Meeting  
**Date:** February 20, 2024  
**Summary:** Internal group update on integrating JAX-based PE (flowMC/jim) into NMMA for both GW and EM analyses, with a live demo.

## Slide Index

| Frame title | Content |
|---|---|
| Parameter estimation | Motivation for faster PE; bottleneck in MCMC-based inference |
| Overview | Talk roadmap |
| JAX? | JAX transformations (JIT, grad, vmap); acceleration rationale |
| flowMC — local sampling | MALA local sampler |
| flowMC — normalizing flows | NF as global proposal |
| flowMC — global sampling | Global move mechanics |
| flowMC — complete algorithm | Full flowMC pipeline |
| Results — GW | GW170817 posteriors + runtime vs bilby |
| Results — EM | AT2017gfo light curve fits / kilonova results |
| Demo | Live or recorded demo of jim/NMMA in action |

## Key Figures
- GW posterior comparison plots
- EM light curve fitting results (AT2017gfo)
