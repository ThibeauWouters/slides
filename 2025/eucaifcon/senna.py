"""Making some masterplots for the ET symposium"""

import numpy as np 
np.random.seed(0)
import matplotlib.pyplot as plt

fs = 22
params = {"axes.grid": True,
        "text.usetex" : True,
        "font.family" : "serif",
        "ytick.color" : "black",
        "xtick.color" : "black",
        "axes.labelcolor" : "black",
        "axes.edgecolor" : "black",
        "font.serif" : ["Computer Modern Serif"],
        "xtick.labelsize": fs,
        "ytick.labelsize": fs,
        "axes.labelsize": fs,
        "legend.fontsize": fs,
        "legend.title_fontsize": fs,
        "figure.titlesize": fs}

plt.rcParams.update(params)

# JAX_BLUE = "#2a56c6" # jax blue, darker
JAX_GREEN = "#00695c" # jax green, darker
JAX_PURPLE = "#6a1b9a" # jax purple, darker

JAX_BLUE = "#5e97f6" # jax blue, lighter
# JAX_GREEN = "#26a69a" # jax green, lighter
# JAX_PURPLE = "#ea80fc" # jax purple, lighter, pink really

N_inj = 2
numbers = np.array([-7.17, 0.0, -0.08, -1.17, -2.95]) # set 3, N_inj = 2
indices = [1, 2, 3, 4, 5]

plt.figure(figsize = (6, 4))
zorder = 1e4
plt.scatter(indices, numbers, color=JAX_PURPLE, s=100, zorder=zorder)
plt.plot(indices, numbers, color=JAX_PURPLE, linewidth=2, zorder=zorder)

plt.axhline(0.0, color="black")
plt.axvline(2.0, linestyle="--", color="black")
fs = 24
plt.xlabel(r"$N_{\rm{recovery}}$", fontsize=fs)
plt.ylabel(r"$\ln \mathcal{B}^{N_{\rm{injection}}}_{N_{\rm{recovery}}}$", fontsize=fs)
plt.grid(False)
plt.xticks(indices, labels=[str(i) for i in indices])
plt.ylim(top = 0.5)

plt.savefig("./Figures/senna_plot.pdf", bbox_inches="tight")
plt.close()