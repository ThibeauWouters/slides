import numpy as np
import matplotlib.pyplot as plt 

params = {"axes.grid": True,
        "text.usetex" : True,
        "font.family" : "serif",
        "ytick.color" : "black",
        "xtick.color" : "black",
        "axes.labelcolor" : "black",
        "axes.edgecolor" : "black",
        "font.serif" : ["Computer Modern Serif"],
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "axes.labelsize": 16,
        "legend.fontsize": 16,
        "legend.title_fontsize": 16,
        "figure.titlesize": 16}

plt.rcParams.update(params)

def plot_cs2_sketch():
    
    plt.figure(figsize = (6, 6))
    n = np.linspace(2.0, 4.0, 1_000)
    n_sat = 1
    
    n_grid_points = np.array([2.2, 2.76, 3.1, 3.3, 3.9])
    cs2_grid_points = np.array([0.4, 0.2, 0.7, 0.8, 0.1])
    
    idx_annot = 3
    
    def cs2(n_array):
        return np.interp(n_array, n_grid_points, cs2_grid_points)
    
    plt.plot(n, cs2(n), color = "blue", zorder = 1e9)
    plt.scatter(n_grid_points, cs2_grid_points, color = "blue", zorder = 1e9)
    fs = 32
    plt.xlabel(r"$n$ [$n_{\rm sat}$]", fontsize = fs)
    plt.ylabel(r"$c_s^2$", fontsize = fs)
    plt.xlim(2.0, 4.0)
    plt.ylim(0.0, 1.0)
    
    # Annotate at the idx point
    n_annot = n_grid_points[idx_annot]
    cs2_annot = cs2_grid_points[idx_annot]
    
    text = r"$\left(n^{(i)}, (c_s^2)^{(i)}\right)$"
    plt.text(n_annot, cs2_annot + 0.1, text, fontsize = 24, ha = 'center', bbox=dict(facecolor='white', edgecolor='none', alpha=1.0))
    
    plt.savefig("./Figures/cs2_sketch.pdf", bbox_inches = "tight")

def main():
    plot_cs2_sketch()

if __name__ == "__main__":
    main()