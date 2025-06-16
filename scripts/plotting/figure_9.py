import os

import numpy as np

from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from saveable_objects.extensions import SaveableWrapper
from thread_chunks import Checkpoint
from tqdm import tqdm

from tasks import setup_matplotlib
from src.setup.get_dir import DATA_DIR, FIG_DIR
from tasks.bootstrap import bootstrap

figure_3_data_dir = os.path.join(DATA_DIR, "figure_3_and_4")
figure_9_data_dir = os.path.join(DATA_DIR, "figure_9")
fig_path = os.path.join(FIG_DIR, "figure_9.pdf")

Es = [os.path.join(figure_9_data_dir, "non_linear_zeeman_splittings.pkl"),
      os.path.join(figure_3_data_dir, "4_qubits.pkl")]

# Load the data
for i in range(len(Es)):
    SaveableList = SaveableWrapper[list];
    Es[i] = np.array(SaveableList.load(Es[i], strict_typing=False)).T
    Es[i] = np.flip(np.array(Es[i]), axis=0)

Tmaxs=[50, 60]
num_ts=[50, 30]
colors = ["#fe6100", "#2e6e8e"]
labels = ["Non-Linear Detunings", "Linear Detunings"]
qubits = 4
do_bootstrap = True
number_bootstrap_samples = 100000
detuning = 30
splittings = [np.linspace(detuning, -detuning, 4)[[1, 3, 0, 2]]+np.array([-2, 0, 0, 1]),
             np.array([0]) if qubits == 1 else np.linspace((-detuning), (+detuning), qubits)]

fig, axis = plt.subplots(1, 1, constrained_layout=True)
inset_ax = inset_axes(axis,
                      width="33%",
                      height="25%",
                      loc="upper left",
                      bbox_to_anchor=(0.18,-0.01,1,1), bbox_transform=axis.transAxes)
inset_ax.set_xlabel(r"Qubit Index $i$")
inset_ax.set_ylabel(r"$\Delta B_i$/MHz")
inset_ax.set_xticks(np.arange(1, qubits+1))
inset_ax.set_yticks([-detuning, 0, detuning])
inset_ax.set_xlim([0.5, qubits+0.5])
inset_ax.set_ylim([-detuning*1.3, detuning*1.3])

for Tmax, num_t, E, splitting, color, label in zip(Tmaxs, num_ts, Es, splittings, colors, labels):
    inset_ax.plot(np.arange(1, qubits+1), splitting, color=color, marker="o")
    ts = np.linspace(0, Tmax, num_t)

    MET = np.nanmin(np.divide(np.expand_dims(ts, axis=-1), -np.log10(np.clip(E, 0, 1))>=7), axis=0)

    hist, bins = np.histogram(MET, ts)
    hist = hist/len(MET)
    cumulative = np.cumsum(hist)

    if do_bootstrap:
        lower_bound, upper_bound, _ = bootstrap(MET, ts, number_bootstrap_samples, 99.99)
        axis.fill_between(ts, [0]+list(lower_bound), [0]+list(upper_bound), color=color, edgecolor="none", alpha = 0.2)

    axis.plot(ts, [0]+list(cumulative), color=color, linewidth=0.5, label=label)

# Set axis properties
axis.set_xlim([0, 50])
axis.set_xlabel("$T$ / ns")
axis.set_ylabel(r"$\mathbb P\left(\textrm{MET}\le T\right)$")
axis.set_ylim([0, 1])

# Adding legend
fig.legend(loc="outside lower center")

# Setting the figure size
width = 3.40457
height = 3
fig.set_size_inches(width, height)

# Saving the figure
fig.savefig(fig_path, dpi=600)