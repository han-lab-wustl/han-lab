import numpy as np
import matplotlib.pyplot as plt
import scipy.stats

# File paths
filenames = [
    'dff_grabda_e241.npy',
    'dff_grabda_e242.npy',
    'dff_grabda_e243.npy',
    'dff_grabda_e277.npy',
    'dff_grabne_e274.npy'
]

# Load data
arrays = [np.load(f) for f in filenames]
group1 = np.concatenate(arrays[:4], axis=1)  # GRABDA: shape (4, N, 100)
group2 = arrays[4]  # GRABNE: shape (4, M, 100)

# Plotting parameters
range_val = 10
binsize = 0.2
time_bins = group1.shape[2]
time_axis = np.linspace(-range_val, range_val, time_bins)

planelut = ['SLM', 'SR', 'SP', 'SO']
colors = {'GRABDA': '#1f77b4', 'GRABNE': '#E24A33'}

# Per-plane plots
for pln in range(4):
    plt.figure(figsize=(6, 3))
    for group, data, label in zip(['GRABDA', 'GRABNE'], [group1, group2], ['GRABDA', 'GRABNE']):
        mean = np.nanmean(data[pln], axis=0)
        sem = scipy.stats.sem(data[pln], axis=0, nan_policy='omit')
        plt.plot(time_axis, mean, label=label, color=colors[group])
        plt.fill_between(time_axis, mean - sem, mean + sem, alpha=0.3, color=colors[group])
    plt.axvline(0, linestyle='--', color='k')
    plt.title(f'Plane {planelut[pln]}')
    plt.xlabel('Time from CS (s)')
    plt.ylabel('dFF')
    plt.legend()
    plt.grid(False)
    plt.tight_layout()
    plt.show()

# Deep vs. Superficial
deep_idx = [0, 1, 2]
superficial_idx = 3

def plot_combined(name, idxs):
    plt.figure(figsize=(6, 3))
    for group, data in [('GRABDA', group1), ('GRABNE', group2)]:
        selected = data[idxs] if isinstance(idxs, list) else data[[idxs]]
        mean = np.nanmean(selected, axis=(0, 1))
        sem = scipy.stats.sem(selected.reshape(-1, selected.shape[-1]), axis=0, nan_policy='omit')
        plt.plot(time_axis, mean, label=group, color=colors[group])
        plt.fill_between(time_axis, mean - sem, mean + sem, alpha=0.3, color=colors[group])
    plt.axvline(0, linestyle='--', color='k')
    plt.title(f'{name} Planes')
    plt.xlabel('Time from CS (s)')
    plt.ylabel('dFF')
    plt.legend()
    plt.grid(False)
    plt.tight_layout()
    plt.show()

plot_combined('Superficial ', deep_idx)
plot_combined('Deep ', superficial_idx)
