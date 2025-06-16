import os

import matplotlib.pylab as pl
import numpy as np

from matplotlib import pyplot as plt
from saveable_objects.extensions import SaveableWrapper
from thread_chunks import Checkpoint
from tqdm import tqdm

from tasks import setup_matplotlib
from src.setup.get_dir import DATA_DIR, FIG_DIR

data_dir = os.path.join(DATA_DIR, "figure_3_and_4")
fig_path = os.path.join(FIG_DIR, "figure_3.png")

SaveableList = SaveableWrapper[list];


paths = [os.path.join(data_dir, f"{qubits}_qubits.pkl") for qubits in [1, 2, 3, 4]]

# Initialise figure
fig, axes_all = plt.subplots(2, 5, width_ratios = [1, 1, 1, 1, 0.08])#, height_ratios=[3, 2.7, 2.7])
axes_all = np.array(axes_all)
axes_all = np.concatenate([axes_all, [axes_all[-1]]])
caxes = axes_all[:, 4:].flatten()
caxes[0].remove()
axes_all = axes_all[:, :4]

for i, axes, Tmax, path in zip(range(4), axes_all.T, [500, 290, 100, 60], paths):
    # Set plot title
    axes[0].set_title(f"{i+1} Qubit" + ("s" if i != 0 else ""), fontsize=10)
    # Load data
    E = np.array(SaveableList.load(path, strict_typing=False))
    E = np.flip(np.array(E), axis=1)

    # We will truncate the samples so that all plots have the same number. To
    #   make this fair we will first shuffle them.
    np.random.shuffle(E)
    E = E[:1024]

    num_t=Tmax//2
    ts = np.linspace(0, Tmax, num_t)

    def get_hist(data, bins, min_val=0, max_val=1):
        E_bins = np.linspace(min_val, max_val, bins)
        hists = []
        for vals in data.T:
            hist, bin_edges = np.histogram(vals, E_bins)
            hists.append(hist)
        hists = np.array(hists)
        hists = hists/np.expand_dims(np.sum(hists, axis=1), axis=1)
        return hists, bin_edges

    # Clean the data
    np.nan_to_num(E, False, 1e-10, 1, 1e-10)
    E = np.clip(E, 1e-10, 1)
    E_min = np.nanmin(E)
    E_max = np.nanmax(E)
    E_min = np.nanmin(E)
    E_max = np.nanmax(E)

    # Get the distribution of the infidelities
    hist, bins = get_hist(np.log(E), 10000, min_val=np.log(E_min), max_val=0)
    bins = np.exp(bins)

    # Computing cumulative distribution
    cumulative = np.cumsum(hist, axis=1)
    # Plotting the cumulative infidelity distribution
    im1 = axes[2].pcolor(bins, list(ts)+[ts[-1]+ts[1]], cumulative, cmap="cividis_r", vmax=1, vmin=0)
    if i >= 2:
        # Fill uncomputed data with extrapolated values
        axes[2].pcolor(bins, [ts[-1]+ts[1], 200*np.sqrt(2)], np.expand_dims(np.mean(cumulative[-10:], axis=0), axis=0), cmap="cividis_r", vmax=1, vmin=0)
        axes[2].axhline(ts[-1]+ts[1], linestyle="--", color="gray", linewidth=1, alpha=0.75)
    # Add contours
    con1 = axes[2].contour(bins[:-1], ts, cumulative, levels=[0.25, 0.5, 0.75], origin="lower", colors=["k", "k", "white", "white"], extent=[E_min, E_max, 0, Tmax], linewidths=0.5)
    if i == 3:
        cbar1 = fig.colorbar(im1, cax=caxes[2], label=r"Cumulative Probability of $\Delta$ given $T$")
        cbar1.add_lines(con1)

    # Set the axes properties
    axes[2].set_xscale("log")
    axes[2].set_xlim([1e-10, 1])
    axes[2].set_xticks([1e-10, 1e-5, 1])
    axes[2].set_ylim([0, 200*np.sqrt(2)])
    axes[0].set_xticks([0, 0.5, 1])
    axes[0].set_xticklabels(["0", "0.5", "1"])

    # Restrict to data below the MET
    argMET = np.max(np.nanargmin(np.divide(np.expand_dims(ts, axis=-1), cumulative>=0.9), axis=0))
    ts = ts[:argMET]
    E = E[:, :argMET]

    # Plot the infidelity histograms
    cmap = pl.colormaps['viridis_r']
    norm = plt.Normalize(vmin=ts[0], vmax=ts[-1])
    indices = np.flip(np.linspace(0, len(ts)-1, 7, dtype=np.int64))
    weights  = np.ones_like(E[:,indices[0]].T)/float(len(E[:,indices[0]].T))
    for t, y in zip(ts[indices], E[:,indices].T):
        weights  = np.ones_like(y)/float(len(y))
        x, bins, p = axes[0].hist(y, np.linspace(0,1,25), color=cmap(norm(t)), weights = weights, alpha=0.25)
        x, bins, p = axes[0].hist(y, np.linspace(0,1,25), histtype='step', color=cmap(norm(t)), weights = weights)
        axes[0].plot([], [], color=cmap(norm(t)), label=int(np.round(t, 0)))
    axes[0].legend(title=r"$T$/ns", fontsize=8, title_fontsize=8, labelspacing=0)
    axes[0].set_xlim([0, 1])

    # Label plots
    if i != 0:
        for axis in axes:
            axis.set_yticks([])
    else:
        axes[0].set_ylabel(r"Probability of $\Delta$ given $T$")
        axes[2].set_ylabel(r"$T$ / ns")

    axes[0].set_xlabel(r"Infidelity $\Delta$")
    axes[2].set_xlabel(r"Infidelity $\Delta$")

caxes[2].set_ylabel(r"Cumulative Probability"+"\n"+r"of $\Delta$ given $T$")

fig.align_ylabels(caxes)
fig.align_ylabels(axes_all.T[0])

# Adjusting the figure size
width = 7.05826
height = width/2
fig.set_size_inches(width, height)
fig.subplots_adjust(right=0.9,
                    top=0.945,
                    left=0.075,
                    bottom=0.115,
                    hspace=0.45,
                    wspace=0.4)
# Save the figure
plt.savefig(fig_path, dpi=1200)