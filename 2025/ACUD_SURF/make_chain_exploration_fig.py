#!/usr/bin/env python3
"""
Generate a figure illustrating Markov chain Monte Carlo exploration.
Shows density contours of a 2D multimodal distribution and an MCMC chain exploring it.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import seaborn as sns
import os

FS_LABELS = 42

def setup_plot_style():
    """Configure matplotlib parameters for black background plots."""
    params = {
        "axes.grid": False,
        "text.usetex": True,
        "font.family": "serif",
        "ytick.color": "white",
        "xtick.color": "white",
        "axes.labelcolor": "white",
        "axes.edgecolor": "white",
        "axes.facecolor": "black",
        "figure.facecolor": "black",
        "savefig.facecolor": "black",
        "font.serif": ["Computer Modern Serif"]
    }
    plt.rcParams.update(params)


def create_target_distribution(mean1=None, cov1=None, mean2=None, cov2=None, weight1=0.5):
    """Create a bimodal target distribution (mixture of two Gaussians).

    Args:
        mean1: Mean of first Gaussian (default: [-2.0, -0.5])
        cov1: Covariance of first Gaussian (default: [[0.8, 0.4], [0.4, 0.3]])
        mean2: Mean of second Gaussian (default: [1.5, 1.0])
        cov2: Covariance of second Gaussian (default: [[0.5, -0.2], [-0.2, 0.4]])
        weight1: Weight of first Gaussian (default: 0.5)
    """
    # Default values
    if mean1 is None:
        mean1 = np.array([-2.0, -0.5])
    if cov1 is None:
        cov1 = np.array([[0.8, 0.4], [0.4, 0.3]])
    if mean2 is None:
        mean2 = np.array([1.5, 1.0])
    if cov2 is None:
        cov2 = np.array([[0.5, -0.2], [-0.2, 0.4]])

    dist1 = multivariate_normal(mean=mean1, cov=cov1)
    dist2 = multivariate_normal(mean=mean2, cov=cov2)
    weight2 = 1.0 - weight1

    def target_pdf(pos):
        """Bimodal target distribution."""
        return weight1 * dist1.pdf(pos) + weight2 * dist2.pdf(pos)

    return target_pdf


def metropolis_hastings(target_pdf, start_pos, n_samples=1000, proposal_std=0.3):
    """Run Metropolis-Hastings algorithm.

    Args:
        target_pdf: Target probability density function
        start_pos: Starting position (numpy array)
        n_samples: Number of MCMC samples to generate
        proposal_std: Standard deviation of proposal distribution

    Returns:
        Array of MCMC samples
    """
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


def make_mcmc_figure(
    filename,
    n_chains=1,
    n_samples=300,
    proposal_std=0.25,
    plot_samples=False,
    show_colorbar=False,
    show_true_density=True,
    show_ticks=False,
    alpha=1.0,
    alpha_true=1.0,
    figsize=(10, 6),
    dpi=300,
    random_seed=42,
    mean1=None,
    cov1=None,
    mean2=None,
    cov2=None,
    weight1=0.5
):
    """Generate MCMC exploration figure and save to file.

    Args:
        filename: Output filename (without extension, will save as PNG and PDF)
        n_chains: Number of chains to plot (1-5)
        n_samples: Number of MCMC samples per chain
        proposal_std: Standard deviation of MCMC proposal distribution
        plot_samples: Whether to plot individual sample points
        show_colorbar: Whether to show colorbar
        show_true_density: Whether to show the true density contours
        show_ticks: Whether to show axis tick numbers (labels always shown)
        alpha: Alpha transparency for MCMC chains and markers
        alpha_true: Alpha transparency for true density contours
        figsize: Figure size tuple (width, height)
        dpi: Resolution for PNG output
        random_seed: Random seed for reproducibility
        mean1: Mean of first Gaussian in mixture
        cov1: Covariance of first Gaussian in mixture
        mean2: Mean of second Gaussian in mixture
        cov2: Covariance of second Gaussian in mixture
        weight1: Weight of first Gaussian (0 to 1)
    """
    # Set random seed
    np.random.seed(random_seed)

    # Setup plot style
    setup_plot_style()

    # Create target distribution with custom parameters
    target_pdf = create_target_distribution(mean1, cov1, mean2, cov2, weight1)

    # Create grid for contour plot
    x = np.linspace(-5, 4, 300)
    y = np.linspace(-4, 4, 300)
    X, Y = np.meshgrid(x, y)
    pos = np.dstack((X, Y))
    Z = target_pdf(pos)

    # Define starting locations for up to 5 chains
    starting_locations = [
        np.array([-4.5, -2.0]),   # Lower left, outside mode
        np.array([3.0, 3.0]),     # Upper right, outside mode
        np.array([-3.0, 2.5]),    # Upper left
        np.array([2.5, -3.0]),    # Lower right
        np.array([0.0, -3.5])     # Bottom center
    ]

    # Generate MCMC samples for multiple chains (skip if n_samples = 0)
    chains = []
    if n_samples > 0:
        for chain_idx in range(n_chains):
            chain = metropolis_hastings(
                target_pdf,
                starting_locations[chain_idx],
                n_samples=n_samples,
                proposal_std=proposal_std
            )
            chains.append(chain)
    else:
        # Just store starting positions if no evolution
        for chain_idx in range(n_chains):
            chains.append(starting_locations[chain_idx].reshape(1, 2))

    # Create figure
    _, ax = plt.subplots(figsize=figsize)

    # Plot true density if requested
    if show_true_density:
        # Use colormap that works on black background (light colors)
        cmap = sns.color_palette("rocket", as_cmap=True)

        # Define contour levels
        max_Z = np.max(Z)
        contour_levels = max_Z * np.linspace(0.005, 1.0, 15)

        # Plot filled contours with colormap (higher zorder than chains)
        contourf = ax.contourf(X, Y, Z, levels=contour_levels, cmap=cmap,
                               alpha=alpha_true, zorder=20)

        # Plot contour lines (light gray on black background, higher zorder)
        ax.contour(X, Y, Z, levels=contour_levels, colors='lightgray',
                   linewidths=0.5, alpha=alpha_true * 0.6, zorder=21)
    else:
        contourf = None

    # Define colors for different chains (bright colors for black background)
    chain_colors = ['#FF6666', '#66B2FF', '#66FF66', '#FFD700', '#FF66FF']

    # Plot all chains
    for idx, chain in enumerate(chains):
        color = chain_colors[idx]

        # Only plot chain path if there are samples to connect
        if n_samples > 0:
            # Plot chain path
            ax.plot(chain[:, 0], chain[:, 1], color=color, linewidth=2,
                    alpha=alpha, zorder=10)
            ax.scatter(chain[:, 0], chain[:, 1], color=color, s=10, alpha=alpha)

            # Plot samples (individual points) - only if flag is True
            if plot_samples:
                ax.scatter(chain[:, 0], chain[:, 1], color=color, s=20, alpha=alpha * 0.6,
                          zorder=11, edgecolors='none')

        # Always mark starting point with a distinct marker
        ax.scatter(chain[0, 0], chain[0, 1], color=color, s=150, marker='o',
                  zorder=12, edgecolors='white', linewidths=2, alpha=alpha)

    # Styling
    ax.set_xlim(-5, 4)
    ax.set_ylim(-4, 4)
    ax.set_aspect('equal')
    ax.grid(False)

    # Always show axis labels
    ax.set_xlabel('Parameter 1', fontsize=FS_LABELS)
    ax.set_ylabel('Parameter 2', fontsize=FS_LABELS)

    if not show_ticks:
        # Remove tick numbers but keep labels
        ax.set_xticks([])
        ax.set_yticks([])

    # Add colorbar (if flag is True and density is shown)
    if show_colorbar and show_true_density and contourf is not None:
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

    # Ensure Figures directory exists
    os.makedirs('Figures', exist_ok=True)

    # Save figure as PNG for Google Slides
    png_path = f'Figures/{filename}.png'
    plt.savefig(png_path, bbox_inches='tight', dpi=dpi, facecolor='black')
    print(f"PNG saved to {png_path}")

    # Also save PDF version
    pdf_path = f'Figures/{filename}.pdf'
    plt.savefig(pdf_path, bbox_inches='tight', facecolor='black')
    print(f"PDF saved to {pdf_path}")

    plt.close()
    print("DONE")


if __name__ == "__main__":
    # Example usage - MCMC exploration figures

    make_mcmc_figure(
        filename="chain_exploration_start",
        n_chains=5,
        n_samples=0,
        proposal_std=0.25,
        plot_samples=False,
        show_colorbar=False,
        show_true_density=False,
        alpha=1.0,
        alpha_true=1.0,
        figsize=(10, 6),
        dpi=300,
        random_seed=42
    )

    make_mcmc_figure(
        filename="chain_exploration_5_chains",
        n_chains=5,
        n_samples=300,
        proposal_std=0.25,
        plot_samples=False,
        show_colorbar=False,
        show_true_density=False,
        alpha=1.0,
        alpha_true=1.0,
        figsize=(10, 6),
        dpi=300,
        random_seed=42
    )

    make_mcmc_figure(
        filename="chain_exploration_5_chains_with_density",
        n_chains=5,
        n_samples=300,
        proposal_std=0.25,
        plot_samples=False,
        show_colorbar=False,
        show_true_density=True,
        alpha=0.5,
        alpha_true=0.8,
        figsize=(10, 6),
        dpi=300,
        random_seed=42
    )

    # Generate 5 density-only plots with different configurations
    # Configuration 1: Original bimodal (default)
    make_mcmc_figure(
        filename="density_1",
        n_chains=0,
        n_samples=0,
        show_true_density=True,
        alpha_true=1.0,
        figsize=(10, 6),
        dpi=300,
        random_seed=42
    )

    # Configuration 2: Well-separated modes
    make_mcmc_figure(
        filename="density_2",
        n_chains=0,
        n_samples=0,
        show_true_density=True,
        alpha_true=1.0,
        figsize=(10, 6),
        dpi=300,
        random_seed=42,
        mean1=np.array([-3.0, -1.5]),
        cov1=np.array([[0.5, 0.0], [0.0, 0.5]]),
        mean2=np.array([2.5, 2.0]),
        cov2=np.array([[0.6, 0.1], [0.1, 0.6]]),
        weight1=0.5
    )

    # Configuration 3: Unequal weights (mode 1 dominant)
    make_mcmc_figure(
        filename="density_3",
        n_chains=0,
        n_samples=0,
        show_true_density=True,
        alpha_true=1.0,
        figsize=(10, 6),
        dpi=300,
        random_seed=42,
        mean1=np.array([-1.5, 0.0]),
        cov1=np.array([[1.0, 0.3], [0.3, 0.8]]),
        mean2=np.array([2.0, 1.5]),
        cov2=np.array([[0.4, -0.1], [-0.1, 0.3]]),
        weight1=0.7
    )

    # Configuration 4: Elongated correlated modes
    make_mcmc_figure(
        filename="density_4",
        n_chains=0,
        n_samples=0,
        show_true_density=True,
        alpha_true=1.0,
        figsize=(10, 6),
        dpi=300,
        random_seed=42,
        mean1=np.array([-2.5, -1.0]),
        cov1=np.array([[1.2, 0.8], [0.8, 0.6]]),
        mean2=np.array([1.0, 1.5]),
        cov2=np.array([[0.8, -0.6], [-0.6, 0.5]]),
        weight1=0.5
    )

    # Configuration 5: Close modes with different shapes
    make_mcmc_figure(
        filename="density_5",
        n_chains=0,
        n_samples=0,
        show_true_density=True,
        alpha_true=1.0,
        figsize=(10, 6),
        dpi=300,
        random_seed=42,
        mean1=np.array([-1.0, 0.5]),
        cov1=np.array([[0.6, 0.2], [0.2, 0.4]]),
        mean2=np.array([1.0, -0.5]),
        cov2=np.array([[1.0, 0.0], [0.0, 0.3]]),
        weight1=0.6
    )