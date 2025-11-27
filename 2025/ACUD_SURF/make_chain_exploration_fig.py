#!/usr/bin/env python3
"""
Generate a figure illustrating Markov chain Monte Carlo exploration.
Shows density contours of a 2D multimodal distribution and an MCMC chain exploring it.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import seaborn as sns

# Configuration
ALPHA = 1.0
PLOT_SAMPLES = False  # Flag to plot individual sample points
SHOW_COLORBAR = False  # Flag to show colorbar
PLOT_N_CHAINS = 5  # Number of chains to plot (1-5)
# Use a colormap that works on black background (light colors)
CMAP = sns.color_palette("rocket", as_cmap=True)  # Reversed for light on dark

# Set random seed for reproducibility
np.random.seed(42)

params = {"axes.grid": False,
        "text.usetex" : True,
        "font.family" : "serif",
        "ytick.color" : "white",
        "xtick.color" : "white",
        "axes.labelcolor" : "white",
        "axes.edgecolor" : "white",
        "axes.facecolor" : "black",
        "figure.facecolor" : "black",
        "savefig.facecolor" : "black",
        "font.serif" : ["Computer Modern Serif"]
        }

plt.rcParams.update(params)


# Define a bimodal target distribution (mixture of two Gaussians)
# Mode 1: lower left
mean1 = np.array([-2.0, -0.5])
cov1 = np.array([[0.8, 0.4],
                 [0.4, 0.3]])
dist1 = multivariate_normal(mean=mean1, cov=cov1)

# Mode 2: upper right
mean2 = np.array([1.5, 1.0])
cov2 = np.array([[0.5, -0.2],
                 [-0.2, 0.4]])
dist2 = multivariate_normal(mean=mean2, cov=cov2)

# Define mixture distribution
def target_pdf(pos):
    """Bimodal target distribution."""
    return 0.5 * dist1.pdf(pos) + 0.5 * dist2.pdf(pos)

# Create grid for contour plot
x = np.linspace(-5, 4, 300)
y = np.linspace(-4, 4, 300)
X, Y = np.meshgrid(x, y)
pos = np.dstack((X, Y))
Z = target_pdf(pos)

# Define starting locations for up to 5 chains
STARTING_LOCATIONS = [
    np.array([-4.5, -2.0]),   # Lower left, outside mode
    np.array([3.0, 3.0]),     # Upper right, outside mode
    np.array([-3.0, 2.5]),    # Upper left
    np.array([2.5, -3.0]),    # Lower right
    np.array([0.0, -3.5])     # Bottom center
]

# Run Metropolis-Hastings MCMC
def metropolis_hastings(target_pdf, start_pos, n_samples=1000, proposal_std=0.3):
    """Run Metropolis-Hastings algorithm."""
    samples = []
    current = start_pos.copy()

    for i in range(n_samples):
        # Propose new sample
        proposed = current + np.random.normal(0, proposal_std, size=2)

        # Calculate acceptance ratio
        current_prob = target_pdf(current)
        proposed_prob = target_pdf(proposed)
        accept_ratio = proposed_prob / current_prob

        # Accept or reject
        if np.random.rand() < accept_ratio:
            current = proposed

        samples.append(current.copy())

    return np.array(samples)

# Generate MCMC samples for multiple chains
n_samples = 300
chains = []
for i in range(PLOT_N_CHAINS):
    chain = metropolis_hastings(target_pdf, STARTING_LOCATIONS[i],
                                n_samples=n_samples, proposal_std=0.25)
    chains.append(chain)

# Create figure
fig, ax = plt.subplots(figsize=(10, 6))

# Define contour levels that extend far out
# Create levels from low density to high density (must be increasing)
max_Z = np.max(Z)
# Use more levels with smoother spacing for gradual color transitions
contour_levels = max_Z * np.linspace(0.005, 0.999, 15)

# Plot filled contours with colormap
contourf = ax.contourf(X, Y, Z, levels=contour_levels, cmap=CMAP, alpha=ALPHA)

# Plot contour lines (light gray on black background)
contours = ax.contour(X, Y, Z, levels=contour_levels, colors='lightgray',
                      linewidths=0.5, alpha=0.6)

# Define colors for different chains (bright colors for black background)
chain_colors = ['#FF6666', '#66B2FF', '#66FF66', '#FFD700', '#FF66FF']

# Plot all chains
for idx, chain in enumerate(chains):
    color = chain_colors[idx]

    # Plot chain path
    ax.plot(chain[:, 0], chain[:, 1], color=color, linewidth=2,
            alpha=0.9, zorder=10)
    ax.scatter(chain[:, 0], chain[:, 1], color=color, s=10, alpha=0.9)

    # Plot samples (blue dots) - only if flag is True
    if PLOT_SAMPLES:
        ax.scatter(chain[:, 0], chain[:, 1], color=color, s=20, alpha=0.6,
                  zorder=11, edgecolors='none')

    # Mark starting point with a distinct marker
    ax.scatter(chain[0, 0], chain[0, 1], color=color, s=150, marker='o',
              zorder=12, edgecolors='white', linewidths=2)

# Styling
ax.set_xlim(-5, 4)
ax.set_ylim(-4, 4)
ax.set_aspect('equal')
ax.grid(False)
fs_labels = 24
ax.set_xlabel('Parameter 1', fontsize=fs_labels)
ax.set_ylabel('Parameter 2', fontsize=fs_labels)

# Add colorbar to show density scale (if flag is True)
if SHOW_COLORBAR:
    cbar = plt.colorbar(contourf, ax=ax, pad=0.02)
    cbar.set_label('Target', fontsize=12, rotation=90, labelpad=15)
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')

# Clean up spines (white borders)
for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_linewidth(1.5)
    spine.set_color('white')

plt.tight_layout()

# Save figure as PNG for Google Slides (better than JPG for presentation graphics)
output_path = 'Figures/chain_exploration.png'
plt.savefig(output_path, bbox_inches='tight', dpi=300, facecolor='black')
print(f"Figure saved to {output_path}")

# Also save PDF version
pdf_path = 'Figures/chain_exploration.pdf'
plt.savefig(pdf_path, bbox_inches='tight', facecolor='black')
print(f"PDF saved to {pdf_path}")
print(f"DONE")