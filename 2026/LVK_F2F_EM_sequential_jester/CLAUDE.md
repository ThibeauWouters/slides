# CLAUDE.md — 2026/LVK_F2F_EM_sequential_jester

**Title:** Scalable and sequential inference of the neutron star equation of state with the Einstein Telescope
**Venue:** LVK Face-to-Face meeting, EM working group
**Date:** placeholder (`\date{LVK F2F Meeting}` in `main.tex` — update with the actual meeting date)
**Summary:** ~15 min talk (+5 min questions) presenting the paper in `/Users/Woute029/Documents/Code/projects/41_eos_inference_3g/paper` (Wouters et al., in prep). Introduces a hybrid sequential Monte Carlo (SMC) algorithm — combining data tempering and likelihood tempering — for scalable, sequential hierarchical inference of the neutron-star EOS from growing BNS catalogs, demonstrated on a simulated month of Einstein Telescope detections and extrapolated to a full year.

## Slide Index

| Frame title | Content |
|---|---|
| Title | tintin background, Utrecht + Nikhef logos, co-authors credited |
| Neutron stars in the 3G era | Motivation: BNS tidal deformations probe EOS; ET/CE detection rates; hierarchical inference bottleneck |
| Towards scalable, sequential inference | Requirements: GPU-accelerated hyperlikelihood (normalizing flows) + sequential reuse of posteriors → SMC |
| Sequential Monte Carlo (SMC) | Generic SMC (reweight/resample/move); likelihood tempering vs. data tempering; toy-model comparison figure |
| Adaptive hybrid algorithm | Queue new events while ESS is high; update via likelihood tempering when ESS drops; paper's Fig. 1 diagram |
| Simulating one month of ET data | Mock catalog setup: ET-2L, merger rate, SNR cut → 1555 sources; single-event PE settings; SMC hyperparameters |
| SMC diagnostics | Paper's Fig. 2: ESS vs. events, cumulative wall time; 22 update batches, ~2.2h on 1 H100 |
| Equation of state constraints | Paper's Fig. 3: pressure-density and Λ/Λ_inj credible bands across batches |
| Extrapolating to a full year | Paper's Fig. 4: runtime power-law fit; projected ~3h (1 month) / ~1.5 days (1 year) at realistic settings |
| Conclusion & outlook | Summary of recipe (GPU + flows + SMC); outlook to joint EOS+population+cosmology inference |
| Thanks | tintin background |
| References | Full bibliography |

## Key Figures
- `Figures/fig1_diagram.pdf` — hybrid data/likelihood tempering algorithm schematic (paper Fig. 1)
- `Figures/fig2_smc_diagnostics_et2l_n1579_noise_time_order.pdf` — ESS + wall-time diagnostics (paper Fig. 2)
- `Figures/fig3_eos_constraints_et2l_n1579_noise_time_order.pdf` — EOS/Λ constraints per batch (paper Fig. 3)
- `Figures/fig4_extrap_runtime_projection.pdf` — runtime extrapolation to 1 year (paper Fig. 4)
- `Figures/smc_temperings.pdf` / `.py` — self-made toy-model comparison of likelihood vs. data tempering (coin-toss example, blackjax)
- `Figures/tintin_BNS_2.png`, `utrecht-university.png`, `Nikhef_logo-transparent.png` — recycled branding assets

## Source material
- Paper + figures: `/Users/Woute029/Documents/Code/projects/41_eos_inference_3g/paper` (`main.tex`, `supplement.tex`, `Figures/`)
- Style/branding recycled from `2026/cuter_x_jester` and `2026/lvk_pisa_jester`
- `references.bib` here is a trimmed extract (21 entries) of the paper's `references.bib`, containing only the keys cited in these slides
