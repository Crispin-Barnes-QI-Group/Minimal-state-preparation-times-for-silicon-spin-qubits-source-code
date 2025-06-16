import os
import pickle as pkl

from functools import reduce

import matplotlib.colors as colors
import matplotlib.patheffects as pe
import matplotlib.pylab as pl
import numpy as np

from matplotlib import pyplot as plt
from matplotlib.cbook import get_sample_data
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from tqdm import tqdm

from tasks import setup_matplotlib
from src.setup.get_dir import DATA_DIR, FIG_DIR
from src.setup.hamiltonian import Ham, get_FCI_energies
from src.setup.molecules import h2, hehp, lih

figure_1_data_dir = os.path.join(DATA_DIR, "figure_1")
figure_2_data_dir = os.path.join(DATA_DIR, "figure_2")
fig_path = os.path.join(FIG_DIR, "figure_2.png")

def factors(n: int) -> set[int]:
    return set(reduce(list.__add__,
                  ([i, n//i] for i in range(1, int(n**0.5) + 1) if n % i == 0)))

E = []

image_files = ["H2.png",
               "HeH.png",
               "LiH.png"]
image_files = [os.path.join(FIG_DIR, path) for path in image_files]

data_files = [
    os.path.join(figure_1_data_dir, "H2.pkl"),
    os.path.join(figure_2_data_dir, "HeH.pkl"),
    os.path.join(figure_2_data_dir, "LiH.pkl")
]

molecules = [h2, hehp, lih]
T_maxs = [24, 10, 50]
r_maxs = [3.3, 3.3, 3.3]
r_mins = [0.3, 0.65, 1.2]
List_of_number_of_lines_to_skip = [2, 4, 4]


images = [plt.imread(get_sample_data(path)) for path in image_files]

# Initialise figure
fig, axes_all = plt.subplots(2, 4, width_ratios=[1, 1, 1, 0.05])
axes_all = np.array(axes_all).T
cax = axes_all[-1, -1]
axes_all[-1, 0].remove()
axes_all = axes_all[:-1]

Emax = -np.inf
Es = []
E0s = []
for i, axes in enumerate(axes_all):
    # Load data
    with open(data_files[i], "rb") as file:
        Es.append(pkl.load(file))
    # Initialising parameters
    num_t, num_r = Es[i].shape
    r_min = r_mins[i]
    r_max = r_maxs[i]
    rs = np.linspace(r_min, r_max, num_r)
    mol = molecules[i]

    # Compute FCI energies
    E0 = get_FCI_energies(mol, rs)

    # Find min and max errors
    Emax = np.nanmax([np.nanmax(np.abs(Es[i]-E0)), Emax])
    E0s.append(E0)

for i, axes in enumerate(axes_all):
    # Initialising parameters
    num_t, num_r = Es[i].shape
    T_max = T_maxs[i]
    r_min = r_mins[i]
    r_max = r_maxs[i]
    number_of_lines_to_skip = List_of_number_of_lines_to_skip[i]
    rs = np.linspace(r_min, r_max, num_r)
    ts = np.linspace(0, T_max, num_t)
    mol = molecules[i]

    # Removing numerical errors. If we can solve the problem in a time T then we
    #   can also solve it in a time T' > T.
    E = np.minimum.accumulate(Es[i], axis=0)

    E0 = E0s[i]

    # Plot the energy error
    im = axes[1].imshow(np.abs(E-E0), extent=[r_min, r_max, 0, T_max], aspect="auto", origin="lower", cmap="cividis", norm=colors.LogNorm(vmin=1e-10, vmax=Emax))

    # Compute the MET
    MET = np.nanmin(np.divide(np.expand_dims(ts, axis=-1), E-E0<=1e-7), axis=0)
    argmax_MET = np.max(np.nanargmin(np.divide(np.expand_dims(ts, axis=-1), E-E0<=1e-7), axis=0))

    # Add Transmon MET lines
    # Data from:
    # Meitei, O.R., Gard, B.T., Barron, G.S. et al. Gate-free state preparation
    # for fast variational quantum eigensolver simulations. npj Quantum Inf 7,
    # 155 (2021). https://doi.org/10.1038/s41534-021-00493-0
    if i == 0:
        bond_distance = [0.3, 0.35, 0.4, 0.45, 0.49, 0.52, 0.55, 0.58, 0.61, 0.66, 0.7, 0.75, 0.8, 0.85, 0.9, 0.96, 1, 1.2, 1.4, 1.6, 1.8, 1.9, 2, 2.1, 2.55, 2.8, 3.05, 3.55]
        time = [9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 10, 10, 10, 14, 14, 14, 18, 22, 21, 21, 24, 24, 24, 20, 22, 22, 20]
        axes[1].plot(bond_distance, time, "gray", label="Transmon MET", linestyle="--")
    if i == 1:
        bond_distance = [0.55, 0.58, 0.61, 0.63, 0.65, 0.67, 0.7, 0.73, 0.75, 0.77, 0.79, 0.82, 0.85, 0.88, 0.91, 0.94, 0.97, 1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2, 2.1, 2.3, 2.5, 2.7, 3, 3.3]
        time = [10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 9, 10, 10, 9, 9, 8, 8, 9, 8, 6, 6, 1, 1, 1, 1, 1]
        axes[1].plot(bond_distance, time, "gray", linestyle="--")

    # Add MET line
    axes[1].plot(rs, MET, color="white", linewidth=1, path_effects=[pe.Stroke(linewidth=3, foreground='k'), pe.Normal()], label="Si MET")

    # Add colour bar
    cbar = fig.colorbar(im, cax = cax, extend='min')
    cbar.ax.set_ylabel("Energy Error / Hartrees")

    # Set axes limits
    axes[0].set_xlim([rs[0], rs[-1]])
    axes[1].set_xlim([r_min, r_max])
    if i == 1:
        hehpEmax = -2.78
        axes[0].set_ylim([np.min(E)-0.05*(hehpEmax-np.min(E)), hehpEmax])
        axes[0].set_yticks([-2.8, -2.82, -2.84, -2.86])
    else:
        axes[0].set_ylim([np.min(E)-0.05*(np.max(E)-np.min(E)), np.max(E)])

    max_MET = np.max(MET)

    # Restrict the data to the points below the MET.
    E_sub = E[:argmax_MET+1]
    ts_sub = ts[:argmax_MET+1]

    # Plot dissociation curves
    cmap = pl.colormaps['viridis']
    norm = plt.Normalize(vmin=0, vmax=max_MET)
    for Et, t in tqdm(zip(np.flip(E_sub, axis=0)[::number_of_lines_to_skip], np.flip(ts_sub)[::number_of_lines_to_skip]), total=len(ts_sub[::number_of_lines_to_skip])):
        axes[0].plot(rs, Et, color=cmap(1-norm(t)), linewidth=1.5)

    # Add labels
    if i == 0:
        axes[0].set_ylabel("Energy / Hartrees")
        axes[1].set_ylabel("Evolution Time $T$ / ns")
    plt.rcParams['axes.titley'] = 1
    if i == 0:
        axes[0].set_title(r"$\textrm{H}_2$", loc="left")
    elif i == 1:
        axes[0].set_title(r"$\textrm{HeH}^+$", loc="left")
    elif i == 2:
        axes[0].set_title(r"$\textrm{LiH}$", loc="left")
    axes[1].set_xlabel(r"Bond Distance (\AA)")

    # Add colour bars
    cbar_insert_axes = inset_axes(axes[0], width = "45%", height = "5%", loc="upper left")
    cbar = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap), cax=cbar_insert_axes, orientation="horizontal", label=r"$T$/ns")
    cbar.ax.xaxis.set_ticks_position("bottom")
    cbar.ax.xaxis.set_label_position("bottom")
    cbar.ax.invert_xaxis()
    try:
        # Set colour bar ticks
        fac = np.concatenate([np.array(list(factors(np.floor(max_MET)))), np.array(list(factors(np.ceil(max_MET))))])
        n_ticks = 2

        ticks = np.unique(np.concatenate([np.linspace(0, np.floor(max_MET), n_ticks+1, dtype=int)[:-1], [max_MET]]))
        
        cbar.ax.set_xticks(1-ticks/max_MET)
        cbar.ax.set_xticklabels(['%g'%(tick) for tick in ticks[:-1]]+['%g'%(np.round(ticks[-1], 2))])
        n_ticks = 1

        for l in cbar.ax.xaxis.get_ticklabels():
            l.set_verticalalignment('top') 
    except: pass    
    # Plot FCI energy
    axes[0].plot(rs, E0, "k--")

fig.align_ylabels(axes_all[0])

# Add legend in custom order
handles, _ = axes_all[0, 1].get_legend_handles_labels()
fig.legend(handles = [handles[1], handles[0]], loc="lower center", ncols = 2)

# Set the figure size and adjust layout
width = 7.05826
height = width/1.9
fig.set_size_inches(width, height)
fig.subplots_adjust(right=0.96,
                    top=0.92,
                    hspace=0.07,
                    left=0.085,
                    bottom=0.22,
                    wspace=0.4)

# Position colour bar
pos = cax.get_position()
pos.x0 -=0.05
pos.x1 -=0.05
cax.set_position(pos)

# Add insets with molecule images
for axes, im in zip(axes_all, images):
    newax = axes[0].inset_axes([0.7, 0.975, 0.3, 0.3])
    newax.imshow(im)
    newax.axis('off')

# Save the figure
fig.savefig(fig_path, dpi=1200)
