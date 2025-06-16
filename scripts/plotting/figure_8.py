import os

import numpy as np

from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from saveable_objects.extensions import SaveableWrapper
from thread_chunks import Checkpoint
from tqdm import tqdm

from tasks import setup_matplotlib
from src.setup.file_names import file_name_parameters
from src.setup.get_dir import DATA_DIR, FIG_DIR
from tasks.bootstrap import bootstrap

figure_3_data_dir = os.path.join(DATA_DIR, "figure_3_and_4")
figure_7_data_dir = os.path.join(DATA_DIR, "figure_7")
figure_8_data_dir = os.path.join(DATA_DIR, "figure_8")
fig_path = os.path.join(FIG_DIR, "figure_8.pdf")

Es = [os.path.join(figure_3_data_dir, "2_qubits.pkl"),
      os.path.join(figure_8_data_dir, "drive_angled.pkl"),
      os.path.join(figure_7_data_dir, f"{file_name_parameters(2*np.pi, (np.pi*0.02/(2*np.sqrt(2)))/8, 0.03)}.pkl")]

# Load the data
for i in range(len(Es)):
    SaveableList = SaveableWrapper[list];
    Es[i] = np.array(SaveableList.load(Es[i], strict_typing=False)).T
    Es[i] = np.flip(np.array(Es[i]), axis=0)

Tmaxs=[290, 1200, 1400]
num_ts=[145, 50, 50]
colors = ["lightgrey", "#fe6100", "#2e6e8e"]
labels = ["On-Axis Drive", r"Off-Axis Drive", "On-Axis 1/8$^\\textrm{th}$ Drive"]
do_bootstrap = True
number_bootstrap_samples = 100000
drive_vecs = [[0,1],
              [np.arccos(1/8), 1],
              [0, 1/8]]

# Initialise figure
fig, axis = plt.subplots(1, 1, constrained_layout=True)

inset_ax = inset_axes(axis,
                      width="25%",
                      height="50%",
                      loc="upper center",
                      bbox_to_anchor=(0.045,0.01,1,1),
                      bbox_transform=axis.transAxes,
                      axes_kwargs={"zorder": 1})
inset_ax.set_aspect('equal')
inset_ax.spines['left'].set_position('zero')
inset_ax.spines['right'].set_visible(False)
inset_ax.spines['bottom'].set_position('zero')
inset_ax.spines['top'].set_visible(False)
inset_ax.xaxis.set_ticks_position('bottom')
inset_ax.yaxis.set_ticks_position('left')
inset_ax.plot((1), (0), ls="", marker=">", ms=2, color="k",
            transform=inset_ax.get_yaxis_transform(), clip_on=False)
inset_ax.plot((0), (1), ls="", marker="^", ms=2, color="k",
        transform=inset_ax.get_xaxis_transform(), clip_on=False)
inset_ax.set_xticks([0, 1/8, 1])
inset_ax.set_xticklabels([0, r"$\frac{1}{8}$", 1])
inset_ax.set_yticks([0, np.sqrt(1-np.square(1/8))])
inset_ax.set_yticklabels([0, r"$\sqrt{63}/8$"])
inset_ax.set_xlim([0, 1.05])
inset_ax.set_ylim([-0.025, 1.04])
axis.patch.set_alpha(0)
inset_ax.tick_params(axis='y', which='major', pad=9)
inset_ax.tick_params(axis='x', which='major', pad=9)

theta = np.linspace(0, np.arccos(1/8), 10000)

inset_ax.plot(1/4*np.cos(theta), 1/4*np.sin(theta), color="lightgrey", linestyle="--", linewidth=0.5)
t=plt.text(5/16*np.cos(theta[10000//2]), 5/16*np.sin(theta[10000//2]),  r"$\cos^{-1}(1/8)$", zorder=-4)
plt.text(0, 1.05,  r"$z$", horizontalalignment='center', verticalalignment='bottom')
plt.text(1.1, 0,  r"$x$", horizontalalignment='left', verticalalignment='center')

for Tmax, num_t, E, drive_vec, color, label in zip(Tmaxs, num_ts, Es, drive_vecs, colors, labels):
    inset_ax.arrow(0, 0, (drive_vec[1]-0.025)*np.cos(drive_vec[0]), (drive_vec[1]-0.025)*np.sin(drive_vec[0]), head_width = 0.025, head_length=0.025,
                  edgecolor = color, facecolor = color, lw = 1, zorder = 5)
    plt.plot([0, drive_vec[1]*np.cos(drive_vec[0])], [drive_vec[1]*np.sin(drive_vec[0])]*2, zorder=-5, color="lightgrey", linestyle="--", linewidth=0.5)
    plt.plot([drive_vec[1]*np.cos(drive_vec[0])]*2, [0, drive_vec[1]*np.sin(drive_vec[0])], zorder=-5, color="lightgrey", linestyle="--", linewidth=0.5)
    ts = np.linspace(0, Tmax, num_t)

    MET = np.nanmin(np.divide(np.expand_dims(ts, axis=-1), -np.log10(np.clip(E, 0, 1))>=7), axis=0)

    hist, bins = np.histogram(MET, ts)
    hist = hist/len(MET)
    cumulative = np.cumsum(hist)


    if do_bootstrap:
        lower_bound, upper_bound, _ = bootstrap(MET, ts, number_bootstrap_samples, 99.99)
        axis.fill_between(ts, [0]+list(lower_bound), [0]+list(upper_bound), color=color, edgecolor="none", alpha = 0.2, zorder=4)

    axis.plot(ts, [0]+list(cumulative), color=color, linewidth=0.5, label=label, zorder=5)

# Set axis properties
axis.set_xlim([10, 1400])
axis.set_xlabel("$T$ / ns")
axis.set_ylabel(r"$\mathbb P\left(\textrm{MET}\le T\right)$")
axis.set_ylim([0, 1])
axis.set_xscale("log")

# Adding legend
handles, _ = axis.get_legend_handles_labels()
fig.legend(handles=list(np.array(handles)[[1,2,0]]),loc="outside lower center", ncol=2)

# Adjusting the figure size
width = 3.40457
height = 3
fig.set_size_inches(width, height)

# Saving the figure
fig.savefig(fig_path, dpi=600)