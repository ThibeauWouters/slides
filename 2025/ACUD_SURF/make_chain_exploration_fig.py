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
# CMAP = 'viridis'      # Colormap for filled contours
CMAP = sns.color_palette("rocket_r", as_cmap=True)
# CMAP = sns.color_palette("crest", as_cmap=True)

# Set random seed for reproducibility
np.random.seed(42)

params = {"axes.grid": False,
        "text.usetex" : True,
        "font.family" : "serif",
        "ytick.color" : "black",
        "xtick.color" : "black",
        "axes.labelcolor" : "black",
        "axes.edgecolor" : "black",
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

# Run Metropolis-Hastings MCMC
def metropolis_hastings(target_pdf, n_samples=1000, proposal_std=0.3):
    """Run Metropolis-Hastings algorithm."""
    samples = []
    current = np.array([-4.5, -2])  # Start from lower left, outside mode

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

# Generate MCMC samples
n_samples = 300
chain = metropolis_hastings(target_pdf, n_samples=n_samples, proposal_std=0.25)

# Create figure
fig, ax = plt.subplots(figsize=(10, 6))

# Define contour levels that extend far out
# Create levels from low density to high density (must be increasing)
max_Z = np.max(Z)
# Use more levels with smoother spacing for gradual color transitions
contour_levels = max_Z * np.linspace(0.005, 0.999, 15)

# Plot filled contours with colormap
contourf = ax.contourf(X, Y, Z, levels=contour_levels, cmap=CMAP, alpha=ALPHA)

# Plot contour lines
contours = ax.contour(X, Y, Z, levels=contour_levels, colors='white',
                      linewidths=0.5, alpha=ALPHA)

# Plot chain path (red line)
ax.plot(chain[:, 0], chain[:, 1], color='#CC3333', linewidth=2,
        alpha=0.9, zorder=10)
ax.scatter(chain[:, 0], chain[:, 1], color='#CC3333', s=10, alpha=0.9)

# Plot samples (blue dots) - only if flag is True
if PLOT_SAMPLES:
    ax.scatter(chain[:, 0], chain[:, 1], color='blue', s=20, alpha=0.6,
              zorder=11, edgecolors='none')

# Mark starting point with a distinct marker
ax.scatter(chain[0, 0], chain[0, 1], color='#CC3333', s=150, marker='o',
          zorder=12, edgecolors='black', linewidths=2)

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

# Clean up spines
for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_linewidth(1.5)

plt.tight_layout()

# Save figure
output_path = 'Figures/chain_exploration.pdf'
plt.savefig(output_path, bbox_inches='tight', dpi=300)
print(f"Figure saved to {output_path}")

# Also save as SVG for editing if needed
svg_path = 'Figures/chain_exploration.svg'
plt.savefig(svg_path, bbox_inches='tight')
print(f"DONE")