import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

params = {"axes.grid": False,
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

filename = "runtime_status.csv"
df = pd.read_csv(filename)

columns = df.columns.tolist()
print("columns", columns)

# Drop those where jim_runtime_seconds has ERROR in the string
jim_runtime_seconds = df["jim_runtime_seconds"].to_numpy()
valid_idx = np.where(np.array(["ERROR:" not in str(x) for x in jim_runtime_seconds]))[0]
print("valid_idx", valid_idx)

# Get data 
df = df.iloc[valid_idx]

bilby_runtime_seconds = df["bilby_runtime_seconds"].to_numpy()

jim_runtime_seconds = df["jim_runtime_seconds"].to_numpy()

# convert to floats
jim_runtime_seconds = np.array([float(x) for x in jim_runtime_seconds])

print("jim_runtime_seconds")
print(jim_runtime_seconds)

bilby_runtime_seconds = np.array([float(x) for x in bilby_runtime_seconds])

print("bilby_runtime_seconds")
print(bilby_runtime_seconds)

jim_gpu_info = df["jim_gpu_info"].to_numpy()

# Make a new column called speedup which is bilby_runtime_seconds / jim_runtime_seconds
speedup = bilby_runtime_seconds / jim_runtime_seconds
df["speedup"] = speedup


# Organize data per GPU type
gpu_types = np.unique(jim_gpu_info)
print("gpu_types", gpu_types)

# Now put the data in a dictionary and save per GPU type
data_per_gpu = {}
for gpu_type in gpu_types:
    idx = np.where(jim_gpu_info == gpu_type)[0]
    gpu_speedup = speedup[idx]
    data_per_gpu[gpu_type] = {
        "bilby_runtime_seconds": bilby_runtime_seconds[idx],
        "jim_runtime_seconds": jim_runtime_seconds[idx],
        "speedup": gpu_speedup,
        "avg_speedup": np.mean(gpu_speedup),
        "n_events": len(idx)
    }
    print(f"gpu_type {gpu_type} has {len(idx)} entries")
    
# Now make a separate bar chart for each GPU
for i, gpu_type in enumerate(gpu_types):
    data = data_per_gpu[gpu_type]

    # Print statistics to screen
    avg_speedup = data["avg_speedup"]
    n_events = data["n_events"]
    print(f"{gpu_type} - Average: {avg_speedup:.1f}x faster ({n_events} events)")

    # Create individual figure for each GPU
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    sns.barplot(x=np.arange(len(data["speedup"])), y=data["speedup"], ax=ax)

    # Set title to just GPU name
    if "A800" in gpu_type:
        ax.set_title("", fontsize=20)
    else:
        ax.set_title(gpu_type, fontsize=20)

    ax.set_ylabel("Speedup (Bilby / Jim)", fontsize=18)
    ax.set_xlabel("Event index", fontsize=18)

    # Increase tick label sizes
    ax.tick_params(axis='both', which='major', labelsize=16)

    # Add horizontal line at speedup = 1
    ax.axhline(y=1, color='red', linestyle='--', alpha=1.0, linewidth=4, label='No speedup')
    ax.legend(fontsize=26)

    ax.set_ylim(0, np.max(data["speedup"]) * 1.1)

    plt.tight_layout()

    # Save each GPU plot separately
    gpu_filename = gpu_type.replace(' ', '_').replace('/', '_')
    plt.savefig(f"speedup_{gpu_filename}.pdf")
    plt.savefig(f"./Figures/speedup_{gpu_filename}.pdf")
    plt.close()
