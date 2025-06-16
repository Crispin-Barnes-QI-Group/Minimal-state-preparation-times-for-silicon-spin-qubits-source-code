import os

import matplotlib.colors as colors
import matplotlib.pylab as pl
import numpy as np

from matplotlib import pyplot as plt
from saveable_objects.extensions import SaveableWrapper
from thread_chunks import Checkpoint
from tqdm import tqdm

from src.setup.file_names import file_name_parameters
from src.setup.get_dir import DATA_DIR, FIG_DIR
from tasks import setup_matplotlib
from tasks.bootstrap import bootstrap

figure_3_data_dir = os.path.join(DATA_DIR, "figure_3_and_4")
figure_7_data_dir = os.path.join(DATA_DIR, "figure_7")
fig_path = os.path.join(FIG_DIR, "figure_7.pdf")

fig, axes = plt.subplots(3, 2, width_ratios = [1, 0.05])
axes = np.array(axes).T
caxs = axes[1]
axes = axes[0]

# Default values
default_J_max = 2*np.pi
default_IQ_max = np.pi*0.02/(2*np.sqrt(2))
default_detuning=0.03

# Values to scan
multiples = [0.125, 0.25, 0.5, 1,  2, 4, 8]

IQpaths = []
for multiple in multiples:
    if multiple == 1:
        IQpaths.append(os.path.join(figure_3_data_dir, "2_qubits.pkl"))
    else:
        IQpaths.append(os.path.join(figure_7_data_dir, f"{file_name_parameters(default_J_max, multiple*default_IQ_max, default_detuning)}.pkl"))
Zpaths = []
for multiple in multiples:
    if multiple == 1:
        Zpaths.append(os.path.join(figure_3_data_dir, "2_qubits.pkl"))
    else:
        Zpaths.append(os.path.join(figure_7_data_dir, f"{file_name_parameters(default_J_max, default_IQ_max, multiple*default_detuning)}.pkl"))
Jpaths = []
for multiple in multiples:
    if multiple == 1:
        Jpaths.append(os.path.join(figure_3_data_dir, "2_qubits.pkl"))
    else:
        Jpaths.append(os.path.join(figure_7_data_dir, f"{file_name_parameters(multiple*default_J_max, default_IQ_max, default_detuning)}.pkl"))
cmap = pl.colormaps['viridis_r']
norm = colors.LogNorm(vmin=0.125, vmax=8)

multiples = [0.125, 0.25, 0.5, 1, 2, 4, 8]
qubits = 2

do_bootstrap = True
number_bootstrap_samples = 100000

for axis in axes:
    axis.set_xscale("log")
    axis.set_xlabel("$T$ / ns")
    axis.set_ylim([0, 1])
    axis.set_ylabel(r"$\mathbb P\left(\textrm{MET}\le T\right)$")

def add_curve(path, axis, Tmax, color):
    SaveableList = SaveableWrapper[list];
    E = SaveableList.load(path, strict_typing=False)
    if len(E[0]) == 3:
        E = list(zip(*E))[0]
        E = SaveableList(E, path=path)

    E = np.array(E).T
    E = np.flip(np.array(E), axis=0)

    num_t=len(E)

    ts = np.linspace(0, Tmax, num_t)
    MET = np.nanmin(np.divide(np.expand_dims(ts, axis=-1), -np.log10(np.clip(E, 0, 1))>=7), axis=0)
        
    hist, _ = np.histogram(MET, ts)
    hist = hist/len(MET)
    cumulative = np.cumsum(hist)

    y = np.array([0]+list(cumulative))
    i1 = np.argmax(y>0)-1
    i2 = np.argmax(y==1)+1
    if i2 < i1:
        i2 = -1
    if do_bootstrap:
        lower_bound, upper_bound, _ = bootstrap(MET, ts, number_bootstrap_samples, 99)
        alpha = 0.2
        axis.fill_between(ts[i1:i2], np.array([0]+list(lower_bound))[i1:i2], np.array([0]+list(upper_bound))[i1:i2], color=color, edgecolor="none", label=f"99\% CI", alpha=alpha)

    axis.plot(ts[i1:i2], y[i1:i2], color=color, linewidth=0.5, label="Numerical Data", zorder=2)

def add_curve_and_cax(path, axis, cax, Tmax, parameter):
    add_curve(path, axis, Tmax, cmap(1-norm(parameter)))
    cax.axhline(parameter, color=cmap(1-norm(parameter)))

Tmaxs = [1400, 400, 200, 290, 100, 100, 100]
for multiple, Tmax, path in zip(multiples, Tmaxs, IQpaths):
    axes[0].axvline(200/(qubits*multiple), color=cmap(1-norm(multiple)), linewidth=0.5, linestyle="--", zorder=-1)
    add_curve_and_cax(path, axes[0], caxs[0], Tmax, multiple)

Tmaxs = [800, 400, 200, 290, 100, 100, 150]
for multiple, Tmax, path in zip(multiples, Tmaxs, Zpaths):
    axes[2].axvline(200/(qubits), color="#2E6E8E", linewidth=0.5, linestyle="--", zorder=-1)
    add_curve_and_cax(path, axes[2], caxs[2], Tmax, multiple)

Tmaxs = [100]*3+[290]+[100]*3
for multiple, Tmax, path in zip(multiples, Tmaxs, Jpaths):
    axes[1].axvline(200/(qubits), color="#2E6E8E", linewidth=0.5, linestyle="--", zorder=-1)
    add_curve_and_cax(path, axes[1], caxs[1], Tmax, multiple)

# 10 MHz line
path = os.path.join(figure_7_data_dir, f"{file_name_parameters(0.01*default_J_max, default_IQ_max, default_detuning)}.pkl")
add_curve(path, axes[1], 300, "k")

for cax in caxs:
    cax.yaxis.tick_right()
    cax.set_yscale("log", base=2)
    cax.yaxis.set_label_position("right")
    cax.set_yticks(multiples)
    cax.set_xticks([])
    cax.spines['top'].set_color('none')
    cax.spines['bottom'].set_color('none')
    cax.spines['left'].set_color('none')
    cax.spines['right'].set_color('none')
    cax.tick_params(length=0)
caxs[0].set_ylabel(r"$\textrm{IQ}_{\max}$ / $\frac{10}{\sqrt{2}}$ MHz")
caxs[1].set_ylabel(r"$J_{\max}$ / GHz")
caxs[2].set_ylabel(r"$\Delta B$ / 60 MHz")

axes[1].set_xlim([10,None])

width = 3.40457
height = width*1.25
fig.set_size_inches(width, height)
fig.subplots_adjust(right=0.85, top=0.985, left=0.15, bottom=0.1, hspace=0.5)
fig.savefig(fig_path, dpi=1200)