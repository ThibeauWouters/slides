#!/usr/bin/env python
"""
Side-by-side comparison of the two SMC "tempering paths" discussed in the
SMC section: likelihood tempering (a single combined likelihood, annealed
via a tempering parameter beta: 0 -> 1) vs. data tempering / partial
posteriors (the full likelihood built up one observation at a time).

Toy problem: estimate the probability of success p of a coin toss. The true
coin is fair (TRUE_P = 0.5); N_OBS tosses are drawn once with a fixed numpy
seed so the figure is reproducible. The prior on p is Uniform(0, 1).

Left panel:  blackjax.adaptive_tempered_smc on the combined Bernoulli
             likelihood of all N_OBS tosses, moved with a Gaussian
             random-walk Metropolis kernel.
Right panel: blackjax.partial_posteriors_smc ("data tempering"), where the
             likelihood is grown one toss at a time and particles are
             re-diversified with a few random-walk Metropolis steps after
             each new observation is added.

Run with the project's blackjax-enabled venv:
    /Users/Woute029/Documents/Code/PhD-thesis/code/.venv/bin/python smc_temperings.py
"""
import os
import sys

import jax
import jax.numpy as jnp
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

import blackjax
import blackjax.smc.resampling as resampling
from blackjax.mcmc.random_walk import normal as normal_proposal
from blackjax.smc import extend_params

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import save_fig, set_style

set_style()

# ---------------------------------------------------------------------------
# Tunable parameters
# ---------------------------------------------------------------------------
DATA_SEED = 43          # numpy seed for generating the coin-toss dataset
JAX_SEED_LIKELIHOOD = 0  # jax PRNGKey seed, likelihood-tempering run
JAX_SEED_DATA = 2        # jax PRNGKey seed, data-tempering run

TRUE_P = 0.5             # true probability of success (fair coin)
N_OBS = 100                # number of coin tosses (observations)
SHOW_TRUE_VALUE = False  # toggle the dashed vertical line marking TRUE_P

PRIOR_LOW, PRIOR_HIGH = 0.0, 1.0  # Uniform(0, 1) prior on p
XMIN, XMAX = 0.2, 0.8  # shared x-axis display range for both panels

N_PARTICLES = 10_000

# Left panel -- likelihood tempering (blackjax.adaptive_tempered_smc)
TARGET_ESS = 0.9                 # target ESS fraction driving the adaptive schedule
RW_SIGMA_LIKELIHOOD = 0.05       # std of the Gaussian random-walk proposal
N_MCMC_STEPS_LIKELIHOOD = 50     # random-walk Metropolis steps per SMC iteration

# Right panel -- data tempering (blackjax.partial_posteriors_smc)
RW_SIGMA_DATA = 0.05             # std of the Gaussian random-walk proposal
N_MCMC_STEPS_DATA = 20           # random-walk Metropolis steps per new observation
JUMP_DATA = max(1, N_OBS // 10)  # only plot every JUMP_DATA-th intermediate density
                                  # (otherwise the right panel gets too crowded);
                                  # the final (n = N_OBS) density is always kept

# Colormap for the intermediate distributions: seaborn "flare", consistent
# with the rest of chapter3/ (same palette as smc_steps.py / nested_sampling.py)
CMAP_NAME = "flare"
COLOR_TRUE = "black"

KDE_LINEWIDTH = 1.4
KDE_ALPHA = 0.9                  # alpha of the KDE line itself
KDE_GRID_POINTS = 1_000

FILL_DENSITIES = True            # toggle filling each density curve under its line
FILL_ALPHA = 0.50                # alpha of the fill (only used if FILL_DENSITIES)

Z_CURVE_BASE = 3                 # zorder of the earliest (least-tempered / prior) curve;
                                  # later curves get Z_CURVE_BASE + i so more "recent"
                                  # observations/temperatures are drawn on top
Z_TRUE_LINE = 1_000               # always drawn above every density curve

FIG_W, FIG_H = 11.0, 4.6
WSPACE = 0.4                      # horizontal spacing between the two panels
TITLE_FONTSIZE = 22
LABEL_FONTSIZE = 20
TICK_FONTSIZE = 17
CBAR_LABEL_FONTSIZE = 18
CBAR_TICK_FONTSIZE = 14

# ---------------------------------------------------------------------------
# Dataset: N_OBS coin tosses from a fair coin
# ---------------------------------------------------------------------------
data_rng = np.random.default_rng(DATA_SEED)
data = data_rng.binomial(1, TRUE_P, size=N_OBS)
data_jnp = jnp.asarray(data, dtype=jnp.float32)

# ---------------------------------------------------------------------------
# Model: Uniform(0, 1) prior, Bernoulli likelihood
# ---------------------------------------------------------------------------
def logprior_fn(p):
    in_bounds = (p >= PRIOR_LOW) & (p <= PRIOR_HIGH)
    return jnp.where(in_bounds, 0.0, -jnp.inf)


def bernoulli_logpmf(p, x):
    p = jnp.clip(p, 1e-6, 1.0 - 1e-6)
    return x * jnp.log(p) + (1.0 - x) * jnp.log(1.0 - p)


def loglikelihood_fn(p):
    """Combined likelihood of all N_OBS tosses (likelihood tempering)."""
    return jnp.sum(bernoulli_logpmf(p, data_jnp))


def partial_logposterior_factory(data_mask):
    """logposterior using only the observations flagged in data_mask (data tempering)."""

    def partial_logposterior(p):
        return logprior_fn(p) + jnp.sum(bernoulli_logpmf(p, data_jnp) * data_mask)

    return partial_logposterior


# ---------------------------------------------------------------------------
# Shared move kernel: Gaussian (additive-step) random-walk Metropolis
# ---------------------------------------------------------------------------
_rw_kernel = blackjax.additive_step_random_walk.build_kernel()


def mcmc_step_fn(rng_key, state, logdensity_fn, sigma):
    return _rw_kernel(rng_key, state, logdensity_fn, normal_proposal(sigma))


mcmc_init_fn = blackjax.additive_step_random_walk.init

# ---------------------------------------------------------------------------
# Left panel: likelihood tempering (adaptive_tempered_smc)
# ---------------------------------------------------------------------------
key_init, key_run = jax.random.split(jax.random.PRNGKey(JAX_SEED_LIKELIHOOD))
init_particles_left = jax.random.uniform(
    key_init, shape=(N_PARTICLES,), minval=PRIOR_LOW, maxval=PRIOR_HIGH
)

likelihood_tempering = blackjax.adaptive_tempered_smc(
    logprior_fn,
    loglikelihood_fn,
    mcmc_step_fn,
    mcmc_init_fn,
    extend_params({"sigma": RW_SIGMA_LIKELIHOOD}),
    resampling.systematic,
    TARGET_ESS,
    num_mcmc_steps=N_MCMC_STEPS_LIKELIHOOD,
)

state_left = likelihood_tempering.init(init_particles_left)
# history_left[i] = (beta_t, particles, weights) at each adaptive SMC iteration,
# starting from beta_t = 0 (the prior).
history_left = [
    (float(state_left.tempering_param), np.asarray(state_left.particles), np.asarray(state_left.weights))
]
while state_left.tempering_param < 1.0:
    key_run, subkey = jax.random.split(key_run)
    state_left, _ = likelihood_tempering.step(subkey, state_left)
    history_left.append(
        (float(state_left.tempering_param), np.asarray(state_left.particles), np.asarray(state_left.weights))
    )

particles_left = np.asarray(state_left.particles)
weights_left = np.asarray(state_left.weights)
print(f"Likelihood tempering: converged in {len(history_left) - 1} adaptive SMC steps")

# ---------------------------------------------------------------------------
# Right panel: data tempering (partial_posteriors_smc)
# ---------------------------------------------------------------------------
key_init2, key_run2 = jax.random.split(jax.random.PRNGKey(JAX_SEED_DATA))
init_particles_right = jax.random.uniform(
    key_init2, shape=(N_PARTICLES,), minval=PRIOR_LOW, maxval=PRIOR_HIGH
)

data_tempering = blackjax.partial_posteriors_smc(
    mcmc_step_fn,
    mcmc_init_fn,
    extend_params({"sigma": RW_SIGMA_DATA}),
    resampling.systematic,
    N_MCMC_STEPS_DATA,
    partial_logposterior_factory=partial_logposterior_factory,
)

state_right = data_tempering.init(init_particles_right, N_OBS)
# history_right[i] = (n_included, particles, weights), starting from
# n_included = 0 (the prior, no data added yet).
history_right = [
    (0, np.asarray(state_right.particles), np.asarray(state_right.weights))
]
for n_included in range(1, N_OBS + 1):
    data_mask = jnp.concatenate(
        [jnp.ones(n_included), jnp.zeros(N_OBS - n_included)]
    )
    key_run2, subkey2 = jax.random.split(key_run2)
    state_right, _ = data_tempering.step(subkey2, state_right, data_mask)
    history_right.append(
        (n_included, np.asarray(state_right.particles), np.asarray(state_right.weights))
    )

particles_right = np.asarray(state_right.particles)
weights_right = np.asarray(state_right.weights)
print(f"Data tempering: added all {N_OBS} observations one at a time")

p_grid = np.linspace(0.0, 1.0, KDE_GRID_POINTS)


def weighted_kde_curve(particles, weights, grid):
    kde = gaussian_kde(particles, weights=weights)
    return kde(grid)


def plot_density_curve(ax, grid, curve, color, zorder):
    """Draw one intermediate density, optionally filled, at a given zorder so
    more "recent" curves (higher zorder) are drawn in front of earlier ones."""
    if FILL_DENSITIES:
        ax.fill_between(grid, 0.0, curve, color=color, alpha=FILL_ALPHA, zorder=zorder, linewidth=0)
    ax.plot(grid, curve, color=color, linewidth=KDE_LINEWIDTH, alpha=KDE_ALPHA, zorder=zorder)


# ---------------------------------------------------------------------------
# Figure: each intermediate distribution drawn as a weighted-KDE curve,
# colored along the "flare" colormap by its tempering progress.
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(FIG_W, FIG_H), sharex=True, sharey=False)
ax_left, ax_right = axes
cmap = plt.get_cmap(CMAP_NAME)

# --- Left panel: likelihood tempering, continuous beta_t in [0, 1] --------
norm_left = mcolors.Normalize(vmin=0.0, vmax=1.0)
for i, (beta_t, particles, weights) in enumerate(history_left):
    curve = weighted_kde_curve(particles, weights, p_grid)
    plot_density_curve(ax_left, p_grid, curve, cmap(norm_left(beta_t)), Z_CURVE_BASE + i)

sm_left = plt.cm.ScalarMappable(cmap=cmap, norm=norm_left)
sm_left.set_array([])
cbar_left = fig.colorbar(sm_left, ax=ax_left, pad=0.02)
cbar_left.set_label(r"$\beta_t$", fontsize=CBAR_LABEL_FONTSIZE)
cbar_left.ax.tick_params(labelsize=CBAR_TICK_FONTSIZE)

# --- Right panel: data tempering, one chunky discrete color block per
# JUMP_DATA-sized bin of observations added (matches the plotted curves) ---
n_bins = N_OBS // JUMP_DATA
boundaries_right = np.arange(0, N_OBS + JUMP_DATA, JUMP_DATA)
cmap_right = plt.get_cmap(CMAP_NAME, n_bins)
norm_right = mcolors.BoundaryNorm(boundaries_right, n_bins, clip=True)
plotted_i = 0
for n_included, particles, weights in history_right:
    if n_included % JUMP_DATA != 0 and n_included != N_OBS:
        continue
    curve = weighted_kde_curve(particles, weights, p_grid)
    plot_density_curve(
        ax_right, p_grid, curve, cmap_right(norm_right(n_included)), Z_CURVE_BASE + plotted_i
    )
    plotted_i += 1

sm_right = plt.cm.ScalarMappable(cmap=cmap_right, norm=norm_right)
sm_right.set_array([])
cbar_right = fig.colorbar(
    sm_right, ax=ax_right, pad=0.02,
    boundaries=boundaries_right, ticks=boundaries_right,
    spacing="uniform", drawedges=True,
)
cbar_right.outline.set_edgecolor("black")
cbar_right.outline.set_linewidth(1.2)
cbar_right.dividers.set_color("black")
cbar_right.dividers.set_linewidth(1.2)
cbar_right.set_label(r"$t$", fontsize=CBAR_LABEL_FONTSIZE)
cbar_right.ax.tick_params(labelsize=CBAR_TICK_FONTSIZE)

# --- Reference curves + styling, shared by both panels --------------------
for ax, title in ((ax_left, "Likelihood tempering"), (ax_right, "Data tempering")):
    if SHOW_TRUE_VALUE:
        ax.axvline(TRUE_P, color=COLOR_TRUE, linestyle="--", linewidth=1.4, zorder=Z_TRUE_LINE)
    ax.set_title(title, fontsize=TITLE_FONTSIZE)
    ax.set_xlabel(r"$\theta$", fontsize=LABEL_FONTSIZE)
    ax.set_xlim(XMIN, XMAX)
    ax.set_ylim(bottom=0.0)
    ax.tick_params(labelsize=TICK_FONTSIZE)

ax_left.set_ylabel(r"$p(\{d_1, \dots, d_M\} | \theta)^{\beta_t} \,\pi(\theta)$", fontsize=LABEL_FONTSIZE)
ax_right.set_ylabel(r"$p(\{d_1, \dots, d_t\} | \theta)\,\pi(\theta)$", fontsize=LABEL_FONTSIZE)

fig.tight_layout()
fig.subplots_adjust(wspace=WSPACE)
save_fig(fig, "smc_temperings")
plt.close(fig)
