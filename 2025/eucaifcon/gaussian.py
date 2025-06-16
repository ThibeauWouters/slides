import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

BEAMER_COLOR = "#4c4cfe"

width = 5
N_samples = 10_000
x = np.linspace(-width, width, N_samples)
y = multivariate_normal.pdf(x, mean=0, cov=1)

plt.plot(x, y, color=BEAMER_COLOR, lw=3)
# Remove everything from the plot to only have the Gaussian
plt.axis("off")
plt.xlim(-3, 3)
plt.ylim(0, 0.45)
plt.savefig("./Figures/gaussian.svg", bbox_inches="tight", transparent=True)
plt.close()