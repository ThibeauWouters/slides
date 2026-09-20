"""
Schematic figure illustrating one iteration of Sequential Monte Carlo (SMC):
Reweight -> Resample -> Move.

Left panel (Reweight): N particles, drawn (equal weight, small dots on the
x-axis) from the previous target p_{t-1}(x) (dashed, light color). They are
reweighted for the new target p_t(x) (solid, dark color): each particle is
placed on the p_t curve, with marker size proportional to its (self-normalized)
importance weight w_i = p_t(x_i) / p_{t-1}(x_i), up to a tunable maximum size.

Middle panel (Resample): particles are resampled (systematic resampling)
according to their weights. Surviving/duplicated particles are drawn as
vertically stacked, unit-size dots above their x position -- taller stacks
for particles resampled more often.

Right panel (Move): each resampled particle undergoes a short p_t-invariant
random-walk Metropolis run, diversifying duplicate locations. The result is
again a set of equal-weight point masses on the x-axis, ready as input to the
next SMC iteration (dashed curve of the next panel would be today's solid
p_t curve).

Notation follows chapters/chapter3_data_analysis.tex's SMC section:
    - target at iteration t-1, t:  p_{t-1}(theta), p_t(theta)
    - particles / weights:         theta_i, w_i  (i = 1, ..., N)
"""

import sys
import os

import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import set_style, HALO, HALO_THIN, get_color, save_fig

set_style()

# ---------------------------------------------------------------------------
# Tunable parameters
# ---------------------------------------------------------------------------
RNG_SEED = 7
N_PARTICLES = 50
# True: place particles at deterministic, evenly-spaced quantiles of p_{t-1}
# (symmetric, reproducible layout). False: draw them iid at random from p_{t-1}.
SYMMETRIC_PARTICLES = False

# Fontsizes
PANEL_TITLE_FONTSIZE = 32
DENSITY_LABEL_FONTSIZE = 26
PANEL_TITLE_PAD = 14

X_MIN, X_MAX = -4.0, 4.0

# previous target p_{t-1}: broad single Gaussian
PREV_MEAN, PREV_STD = 0.0, 1.35

# current target p_t: bimodal mixture (illustrates tempering towards a
# multimodal posterior)
CURR_MEANS = [-1.4, 1.4]
CURR_STDS = [0.75, 0.75]
CURR_WEIGHTS = [0.5, 0.5]

# color palette: seaborn "flare", two accent colors picked by position in [0, 1]
COLOR_POS_PREV = 0.30   # light: p_{t-1}
COLOR_POS_CURR = 0.75   # dark:  p_t
COLOR_PREV = get_color(COLOR_POS_PREV, palette="flare")
COLOR_CURR = get_color(COLOR_POS_CURR, palette="flare")

# marker sizes (in points^2): the "unit weight" is the reference self-normalized
# weight (mean weight == 1 after normalization) mapped to MARKER_SIZE_UNIT;
# larger weights scale up linearly from there, capped at MARKER_SIZE_MAX.
UNIT_WEIGHT = 1.0
MARKER_SIZE_UNIT = 90.0
MARKER_SIZE_MIN = 22.0
MARKER_SIZE_MAX = 260.0

# equal-weight "point mass" dots (bottom row of reweight panel, and the
# diversified particles of the move panel) -- fixed, unit size
AXIS_DOT_SIZE = 42.0

# resample panel: fixed unit-size dots, stacked vertically per particle
RESAMPLE_DOT_SIZE = 55.0
RESAMPLE_DOT_SPACING = 0.032   # vertical spacing between stacked dots (data units)
RESAMPLE_DOT_X_JITTER = 0.0    # optional horizontal jitter within a stack

# move step: short random-walk Metropolis run, target-invariant (p_t)
MCMC_STEP_STD = 0.45
MCMC_N_STEPS = 100

# axis baseline (data units) where "point mass" dots sit, just below y = 0
AXIS_DOT_Y = -0.018

# p_{t-1} / p_t density labels (reweight panel only): placed above the
# Gaussian's center and above the second density's rightmost mode
DENSITY_LABEL_Y_OFFSET = 0.035

# ---------------------------------------------------------------------------
# Target densities
# ---------------------------------------------------------------------------
def gauss(x, mu, sigma):
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))


def p_prev(x):
    return gauss(x, PREV_MEAN, PREV_STD)


def p_curr(x):
    total = np.zeros_like(np.asarray(x, dtype=float))
    for w, mu, sigma in zip(CURR_WEIGHTS, CURR_MEANS, CURR_STDS):
        total = total + w * gauss(x, mu, sigma)
    return total


x_grid = np.linspace(X_MIN, X_MAX, 600)
curve_prev = p_prev(x_grid)
curve_curr = p_curr(x_grid)
Y_TOP = 1.30 * max(curve_prev.max(), curve_curr.max())

# ---------------------------------------------------------------------------
# Step 1 -- Reweight: draw particles from p_{t-1}, compute importance weights
# for p_t
# ---------------------------------------------------------------------------
rng = np.random.default_rng(RNG_SEED)
if SYMMETRIC_PARTICLES:
    # deterministic, evenly-spaced quantile draws from p_{t-1} -- a clean,
    # symmetric stand-in for an iid sample that still covers both tails
    quantiles = (np.arange(1, N_PARTICLES + 1) - 0.5) / N_PARTICLES
    particles_x = norm.ppf(quantiles, loc=PREV_MEAN, scale=PREV_STD)
else:
    particles_x = np.sort(rng.normal(PREV_MEAN, PREV_STD, size=N_PARTICLES))
particles_x = np.clip(particles_x, X_MIN + 0.15, X_MAX - 0.15)

raw_weights = p_curr(particles_x) / p_prev(particles_x)
weights = raw_weights * N_PARTICLES / raw_weights.sum()  # self-normalized, mean 1

marker_sizes = np.clip(
    MARKER_SIZE_UNIT * (weights / UNIT_WEIGHT), MARKER_SIZE_MIN, MARKER_SIZE_MAX
)

# ---------------------------------------------------------------------------
# Step 2 -- Resample: systematic resampling from the (normalized) weights
# ---------------------------------------------------------------------------
def systematic_resample(probs, n, rng):
    positions = (np.arange(n) + rng.uniform()) / n
    cumsum = np.cumsum(probs)
    cumsum[-1] = 1.0
    return np.searchsorted(cumsum, positions)


probs = weights / weights.sum()
resample_idx = systematic_resample(probs, N_PARTICLES, rng)
counts = np.bincount(resample_idx, minlength=N_PARTICLES)

# ---------------------------------------------------------------------------
# Step 3 -- Move: short p_t-invariant random-walk Metropolis run per particle
# ---------------------------------------------------------------------------
def rw_metropolis(x0, n_steps, step_std, rng):
    x = x0
    fx = p_curr(x)
    for _ in range(n_steps):
        x_prop = x + rng.normal(0.0, step_std)
        if x_prop < X_MIN or x_prop > X_MAX:
            continue
        f_prop = p_curr(x_prop)
        if f_prop >= fx or rng.uniform() < f_prop / fx:
            x, fx = x_prop, f_prop
    return x


moved_x = np.array(
    [
        rw_metropolis(particles_x[i], MCMC_N_STEPS, MCMC_STEP_STD, rng)
        for i in range(N_PARTICLES)
        for _ in range(counts[i])
    ]
)

# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
FIG_W, FIG_H = 14.0, 4.6
fig, axes = plt.subplots(1, 3, figsize=(FIG_W, FIG_H))
ax_rw, ax_rs, ax_mv = axes


def style_curve_axis(ax, title):
    ax.set_xlim(X_MIN, X_MAX)
    ax.set_ylim(AXIS_DOT_Y - 0.02, Y_TOP)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.axhline(0.0, color="black", linewidth=1.2, zorder=3)
    ax.set_xlabel(title, fontsize=PANEL_TITLE_FONTSIZE, labelpad=PANEL_TITLE_PAD)


def plot_curve(ax, y, color, linestyle="-", lw=2.2, fill_alpha=0.15):
    ax.fill_between(x_grid, 0, y, color=color, alpha=fill_alpha, zorder=1)
    (line,) = ax.plot(x_grid, y, color=color, linestyle=linestyle, linewidth=lw, zorder=4)
    line.set_path_effects(HALO_THIN)
    return line


# --- Panel 1: Reweight -----------------------------------------------------
style_curve_axis(ax_rw, "Reweight")
plot_curve(ax_rw, curve_prev, COLOR_PREV, linestyle="--", lw=2.0, fill_alpha=0.10)
plot_curve(ax_rw, curve_curr, COLOR_CURR, linestyle="-", lw=2.2, fill_alpha=0.15)

# p_{t-1} label above the Gaussian's center, p_t label above its rightmost mode
x_curr_label = max(CURR_MEANS)
for x_label, curve_fn, color, text in (
    (PREV_MEAN, p_prev, COLOR_PREV, r"$p_{t-1}$"),
    (x_curr_label, p_curr, COLOR_CURR, r"$p_t$"),
):
    label = ax_rw.text(
        x_label, curve_fn(x_label) + DENSITY_LABEL_Y_OFFSET, text,
        ha="center", va="bottom", fontsize=DENSITY_LABEL_FONTSIZE,
        color=color, zorder=16,
    )
    label.set_path_effects(HALO)

# equal-weight input particles, drawn from p_{t-1}, sitting on the x-axis
ax_rw.scatter(
    particles_x, np.full(N_PARTICLES, AXIS_DOT_Y),
    s=AXIS_DOT_SIZE, color=COLOR_PREV, edgecolors="black", linewidths=0.6,
    zorder=10, clip_on=False,
)

# reweighted particles, riding the p_t curve, sized by importance weight
ax_rw.scatter(
    particles_x, p_curr(particles_x),
    s=marker_sizes, color=COLOR_CURR, edgecolors="black", linewidths=0.7,
    zorder=11,
)

# --- Panel 2: Resample -------------------------------------------------
style_curve_axis(ax_rs, "Resample")
plot_curve(ax_rs, curve_curr, COLOR_CURR, linestyle="-", lw=2.2, fill_alpha=0.15)

for i in range(N_PARTICLES):
    m = counts[i]
    if m == 0:
        continue
    ys = AXIS_DOT_Y + RESAMPLE_DOT_SPACING * (0.5 + np.arange(m))
    xs = np.full(m, particles_x[i])
    ax_rs.scatter(
        xs, ys, s=RESAMPLE_DOT_SIZE, color=COLOR_CURR, edgecolors="black",
        linewidths=0.6, zorder=10, clip_on=False,
    )

# --- Panel 3: Move -----------------------------------------------------
style_curve_axis(ax_mv, "Move")
plot_curve(ax_mv, curve_curr, COLOR_CURR, linestyle="-", lw=2.2, fill_alpha=0.15)

ax_mv.scatter(
    moved_x, np.full(len(moved_x), AXIS_DOT_Y),
    s=AXIS_DOT_SIZE, color=COLOR_CURR, edgecolors="black", linewidths=0.6,
    zorder=10, clip_on=False,
)

# ---------------------------------------------------------------------------
# Arrows between panels (Reweight -> Resample -> Move), aligned with the
# panel titles now sitting below each panel as x-labels
# ---------------------------------------------------------------------------
fig.subplots_adjust(left=0.02, right=0.98, wspace=0.22, top=0.92, bottom=0.26)
fig.canvas.draw()
renderer = fig.canvas.get_renderer()

for i in range(2):
    pos_left = axes[i].get_position()
    pos_right = axes[i + 1].get_position()
    label_bbox = axes[i].xaxis.label.get_window_extent(renderer=renderer).transformed(
        fig.transFigure.inverted()
    )
    y_arrow = label_bbox.y0 + 0.5 * label_bbox.height
    arrow = FancyArrowPatch(
        (pos_left.x1 - 0.006, y_arrow), (pos_right.x0 + 0.006, y_arrow),
        transform=fig.transFigure, arrowstyle="-|>", mutation_scale=20,
        linewidth=1.8, color="black", clip_on=False,
    )
    arrow.set_path_effects(HALO_THIN)
    fig.add_artist(arrow)

save_fig(fig, "smc_steps")
