#!/usr/bin/env python3
"""
Plot three neutron star mass populations:
- Uniform
- Gaussian
- Double Gaussian
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Font size constants
TICK_FONTSIZE = 24
LABEL_FONTSIZE = 26
LEGEND_FONTSIZE = 26

# Line width constant
LINE_WIDTH = 3.5
LABELPAD = 20

params = {"axes.grid": False,
        "text.usetex" : True,
        "font.family" : "serif",
        "ytick.color" : "black",
        "xtick.color" : "black",
        "axes.labelcolor" : "black",
        "axes.edgecolor" : "black",
        "font.serif" : ["Computer Modern Serif"],
        "xtick.labelsize": TICK_FONTSIZE,
        "ytick.labelsize": TICK_FONTSIZE,
        "axes.labelsize": LABEL_FONTSIZE,
        "legend.fontsize": LEGEND_FONTSIZE,
        "legend.title_fontsize": LEGEND_FONTSIZE,
        "figure.titlesize": LEGEND_FONTSIZE}

plt.rcParams.update(params)

# Improved corner kwargs
default_corner_kwargs = dict(bins=40, 
                        smooth=1., 
                        show_titles=False,
                        label_kwargs=dict(fontsize=16),
                        title_kwargs=dict(fontsize=16), 
                        color="blue",
                        # quantiles=[],
                        # levels=[0.9],
                        plot_density=True, 
                        plot_datapoints=False, 
                        fill_contours=True,
                        max_n_ticks=4, 
                        min_n_ticks=3,
                        truth_color = "red",
                        save=False)

# JAX color palette
JAX_LIGHT_BLUE = '#60b4ff'
JAX_DARK_GREEN = '#2d8f3f'
JAX_PURPLE = '#9d5de5'

# Population colors
population_colors = {
    'uniform': JAX_LIGHT_BLUE,
    'gaussian': JAX_DARK_GREEN,
    'double_gaussian': JAX_PURPLE
}

# Solar mass constant (for labels)
M_sun = 1.0
MASS_XLIM_LOWER = 0.90
MASS_XLIM_UPPER = 2.45  

# Maximum NS mass - sample from Gaussian with M_TOV = 2.25 +0.42/-0.22 M_sun
# Approximate as Gaussian with mean 2.25 and std ~ 0.3 (average of asymmetric errors)
np.random.seed(42)  # For reproducibility
M_TOV = np.random.normal(2.25, 0.3)
print(f"Sampled M_TOV = {M_TOV:.2f} M_sun")

# Mass range for plotting
mass_range = np.linspace(0.8, 3.0, 1000)

# 1. Uniform distribution
def uniform_population(m, m_min=1.0, m_max=M_TOV):
    """Uniform distribution between m_min and M_TOV"""
    pdf = np.where((m >= m_min) & (m <= m_max), 1.0 / (m_max - m_min), 0.0)
    return pdf

# 2. Gaussian distribution
def gaussian_population(m, mean=1.33, std=0.09):
    """Gaussian distribution with given mean and std"""
    return stats.norm.pdf(m, loc=mean, scale=std)

# 3. Double Gaussian distribution
def double_gaussian_population(m,
                                mean1=1.34, std1=0.07, weight1=0.65,
                                mean2=1.80, std2=0.21):
    """Weighted mixture of two Gaussian distributions"""
    gaussian1 = stats.norm.pdf(m, loc=mean1, scale=std1)
    gaussian2 = stats.norm.pdf(m, loc=mean2, scale=std2)
    weight2 = 1.0 - weight1
    return weight1 * gaussian1 + weight2 * gaussian2

# Calculate PDFs
uniform_pdf = uniform_population(mass_range)
gaussian_pdf = gaussian_population(mass_range)
double_gaussian_pdf = double_gaussian_population(mass_range)

# Create the plot
plt.figure(figsize=(10, 6))

# Plot with fill_between for light shading
plt.plot(mass_range, uniform_pdf, label='Uniform', linewidth=LINE_WIDTH,
         color=population_colors['uniform'])
plt.fill_between(mass_range, uniform_pdf, alpha=0.15, color=population_colors['uniform'])

plt.plot(mass_range, gaussian_pdf, label='Gaussian', linewidth=LINE_WIDTH,
         color=population_colors['gaussian'])
plt.fill_between(mass_range, gaussian_pdf, alpha=0.15, color=population_colors['gaussian'])

plt.plot(mass_range, double_gaussian_pdf, label='Double Gaussian', linewidth=LINE_WIDTH,
         color=population_colors['double_gaussian'])
plt.fill_between(mass_range, double_gaussian_pdf, alpha=0.15, color=population_colors['double_gaussian'])

plt.xlabel(r'Neutron star mass [$M_\odot$]', labelpad=LABELPAD)
plt.ylabel('Probability Density', labelpad=LABELPAD)
plt.legend()
plt.xlim(MASS_XLIM_LOWER, MASS_XLIM_UPPER)
plt.ylim(0, None)

plt.tight_layout()
plt.savefig('./Figures/populations_overview.pdf', dpi=300, bbox_inches='tight')
plt.close()
print("Plot saved as './Figures/populations_overview.pdf'")
