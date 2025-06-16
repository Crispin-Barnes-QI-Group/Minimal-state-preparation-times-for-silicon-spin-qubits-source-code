import os
import pickle as pkl

import matplotlib.colors as colors
import matplotlib.pylab as pl
import numpy as np

from matplotlib import pyplot as plt
from matplotlib.cbook import get_sample_data

from tasks import setup_matplotlib
from src.setup.hamiltonian import get_FCI_energies
from src.setup.molecules import h2, hehp
from src.setup.get_dir import DATA_DIR, FIG_DIR
from src.setup.file_names import file_name_parameters

figure_1_data_dir = os.path.join(DATA_DIR, "figure_1")
figure_2_data_dir = os.path.join(DATA_DIR, "figure_2")
figure_6_data_dir = os.path.join(DATA_DIR, "figure_6")
fig_path = os.path.join(FIG_DIR, "figure_6.pdf")

molecules = [h2, hehp]

# Default values
default_J_max = 2*np.pi
default_IQ_max = np.pi*0.02/(2*np.sqrt(2))
default_detuning=0.03

# Values to scan
multiples = [0.125, 0.25, 0.5, 1,  2, 4, 8]

#H2 data
H2_J_scan_paths = []
for multiple in multiples:
    if multiple == 1:
        H2_J_scan_paths.append(os.path.join(figure_1_data_dir, "H2.pkl"))
    else:
        H2_J_scan_paths.append(os.path.join(figure_6_data_dir, f"H2{file_name_parameters(multiple*default_J_max, default_IQ_max, default_detuning)}.pkl"))
H2_J_scan_num_ts = [256]+[128]*2+[307]+[128]*3
H2_J_scan_T_maxs = [20]+[10]*2+[24]+[10]*3

H2_MW_scan_paths = []
for multiple in multiples:
    if multiple == 1:
        H2_MW_scan_paths.append(os.path.join(figure_1_data_dir, "H2.pkl"))
    else:
        H2_MW_scan_paths.append(os.path.join(figure_6_data_dir, f"H2{file_name_parameters(default_J_max, multiple*default_IQ_max, default_detuning)}.pkl"))
H2_MW_scan_num_ts = [320, 256, 128, 307, 256, 64, 64]
H2_MW_scan_T_maxs = [25, 20, 10, 24, 20, 5, 5]

H2_Z_scan_paths = []
for multiple in multiples:
    if multiple == 1:
        H2_Z_scan_paths.append(os.path.join(figure_1_data_dir, "H2.pkl"))
    else:
        H2_Z_scan_paths.append(os.path.join(figure_6_data_dir, f"H2{file_name_parameters(default_J_max, default_IQ_max, multiple*default_detuning)}.pkl"))
H2_Z_scan_num_ts = [256, 128, 128, 307, 128, 128, 128]
H2_Z_scan_T_maxs = [20, 10, 10, 24, 10, 10, 10]

#HeH+ data
HeHp_J_scan_paths = []
for multiple in multiples:
    if multiple == 1:
        HeHp_J_scan_paths.append(os.path.join(figure_2_data_dir, "HeH.pkl"))
    else:
        HeHp_J_scan_paths.append(os.path.join(figure_6_data_dir, f"HeH{file_name_parameters(multiple*default_J_max, default_IQ_max, default_detuning)}.pkl"))
HeHp_J_scan_num_ts = [128]*7
HeHp_J_scan_T_maxs = [10]*7

HeHp_MW_scan_paths = []
for multiple in multiples:
    if multiple == 1:
        HeHp_MW_scan_paths.append(os.path.join(figure_2_data_dir, "HeH.pkl"))
    else:
        HeHp_MW_scan_paths.append(os.path.join(figure_6_data_dir, f"HeH{file_name_parameters(default_J_max, multiple*default_IQ_max, default_detuning)}.pkl"))
HeHp_MW_scan_num_ts = [512, 256, 128, 128, 256, 64, 64]
HeHp_MW_scan_T_maxs = [40, 20, 10, 10, 20, 5, 5]

HeHp_Z_scan_paths = []
for multiple in multiples:
    if multiple == 1:
        HeHp_Z_scan_paths.append(os.path.join(figure_2_data_dir, "HeH.pkl"))
    else:
        HeHp_Z_scan_paths.append(os.path.join(figure_6_data_dir, f"HeH{file_name_parameters(default_J_max, default_IQ_max, multiple*default_detuning)}.pkl"))
HeHp_Z_scan_num_ts = [256]*2+[128]*5
HeHp_Z_scan_T_maxs = [20]*2+[10]*5

J_scan_paths = [H2_J_scan_paths, HeHp_J_scan_paths]
MW_scan_paths = [H2_MW_scan_paths, HeHp_MW_scan_paths]
Z_scan_paths = [H2_Z_scan_paths, HeHp_Z_scan_paths]

J_scan_num_ts = [H2_J_scan_num_ts, HeHp_J_scan_num_ts]
MW_scan_num_ts = [H2_MW_scan_num_ts, HeHp_MW_scan_num_ts]
Z_scan_num_ts = [H2_Z_scan_num_ts, HeHp_Z_scan_num_ts]

J_scan_T_maxs = [H2_J_scan_T_maxs, HeHp_J_scan_T_maxs]
MW_scan_T_maxs = [H2_MW_scan_T_maxs, HeHp_MW_scan_T_maxs]
Z_scan_T_maxs = [H2_Z_scan_T_maxs, HeHp_Z_scan_T_maxs]

from functools import reduce

def factors(n: int) -> set[int]:
    return set(reduce(list.__add__,
                  ([i, n//i] for i in range(1, int(n**0.5) + 1) if n % i == 0)))

image_files = ["H2.png",
               "HeH.png",
               "LiH.png"]
image_files = [os.path.join(FIG_DIR, path) for path in image_files]
images = [plt.imread(get_sample_data(path)) for path in image_files]

r_mins = [0.3, 0.65]
r_maxs = [3.3, 3.3]
num_rs = [80, 80]

# Initialising figure
fig, axes = plt.subplots(3, 3, width_ratios = [1, 1, 0.1])
axesJ = axes[1]
axesIQ = axes[0]
axesD = axes[2]

cmap = pl.colormaps['viridis_r']
norm = colors.LogNorm(vmin=0.125, vmax=8)

multiples = [0.125, 0.25, 0.5, 1, 2, 4, 8]
Js = [2*np.pi*v for v in multiples]
drives = [v*np.pi*0.02/(2*np.sqrt(2)) for v in multiples]
detunings = [v*0.03 for v in multiples]

for i, axisJ, axisIQ, axisD in zip(range(2), axesJ[:2], axesIQ[:2], axesD[:2]):
    r_min = r_mins[i]
    r_max = r_maxs[i]
    num_r = num_rs[i]
    rs = np.linspace(r_min, r_max, num_r)
    mol = molecules[i]

    # Computing the FCI energies
    E0 = get_FCI_energies(mol, rs)

    # Computing the J METs
    MET_list = []
    for path, num_t, T_max in zip(J_scan_paths[i], J_scan_num_ts[i], J_scan_T_maxs[i]):
        with open(path, "rb") as file:
            E = pkl.load(file)
        ts = np.linspace(0, T_max, num_t)
        MET = np.nanmin(np.divide(np.expand_dims(ts, axis=-1), E-E0<=1e-7), axis=0)
        MET_list.append(MET)
    MET_list = np.array(MET_list)
    # Removing numerical errors. If we can solve the problem in a time T then we
    #   can also solve it in a time T' > T.
    MET_list = np.minimum.accumulate(MET_list, axis=0)

    # Plotting the J curves
    for J_max, MET in zip(Js, MET_list):
        axisJ.plot(rs, MET, color=cmap(1-norm(J_max/(2*np.pi))))

    # Computing the IQ METs
    MET_list = []
    for path, num_t, T_max in zip(MW_scan_paths[i], MW_scan_num_ts[i], MW_scan_T_maxs[i]):
        with open(path, "rb") as file:
            E = pkl.load(file)
        ts = np.linspace(0, T_max, num_t)
        MET = np.nanmin(np.divide(np.expand_dims(ts, axis=-1), E-E0<=1e-7), axis=0)
        MET_list.append(MET)
    MET_list = np.array(MET_list)
    # Removing numerical errors. If we can solve the problem in a time T then we
    #   can also solve it in a time T' > T.
    MET_list = np.minimum.accumulate(MET_list, axis=0)

    # Plotting the IQ curves
    for drive, MET in zip(drives, MET_list):
        axisIQ.plot(rs, MET, color=cmap(1-norm(drive/(np.pi*0.02/(2*np.sqrt(2))))))

    # Computing the detuning METs and plotting the detuning curves
    for detuning, path, num_t, T_max in zip(detunings, Z_scan_paths[i], Z_scan_num_ts[i], Z_scan_T_maxs[i]):
        with open(path, "rb") as file:
            E = pkl.load(file)
        ts = np.linspace(0, T_max, num_t)
        MET = np.nanmin(np.divide(np.expand_dims(ts, axis=-1), E-E0<=1e-7), axis=0)
        axisD.plot(rs, MET, color=cmap(1-norm(detuning/0.03)))

    # Adding the J=10MHz MET lines
    if i == 0:
        path = os.path.join(figure_6_data_dir, f"H2{file_name_parameters(2*np.pi*0.01, default_IQ_max, default_detuning)}.pkl")
        num_t = 256
        T_max=100
    else:
        path = os.path.join(figure_6_data_dir, f"HeH{file_name_parameters(2*np.pi*0.01, default_IQ_max, default_detuning)}.pkl")
        num_t = 256
        T_max=20
    ts = np.linspace(0, T_max, num_t)
    with open(path, "rb") as file:
        E = pkl.load(file)
    MET = np.nanmin(np.divide(np.expand_dims(ts, axis=-1), E-E0<=1e-7), axis=0)
    print(f"J=10MHz MET={np.max(MET)}")
    axisJ.plot(rs, MET, color="k", linestyle="--")

    # Setting axis properties
    if i == 0:
        axisJ.set_ylabel("MET / ns")
        axisIQ.set_ylabel("MET / ns")
        axisD.set_ylabel("MET / ns")
    axisJ.set_xlim([r_min, r_max])
    axisIQ.set_xlim([r_min, r_max])
    axisD.set_xlim([r_min, r_max])
    axisJ.set_yscale("log")
    axisIQ.set_yscale("log")
    axisD.set_yscale("log")

# Setting axis properties
ymax = max([ax.get_ylim()[-1] for ax in axesJ[:2]])
ymin = min([ax.get_ylim()[0] for ax in axesJ[:2]])
for ax in axesJ[:2]: ax.set_ylim([ymin, ymax])
ymax = max([ax.get_ylim()[-1] for ax in axesIQ[:2]])
ymin = min([ax.get_ylim()[0] for ax in axesIQ[:2]])
for ax in axesIQ[:2]: ax.set_ylim([ymin, ymax])
ymax = max([ax.get_ylim()[-1] for ax in axesD[:2]])
ymin = min([ax.get_ylim()[0] for ax in axesD[:2]])
for ax in axesD[:2]: ax.set_ylim([ymin, ymax])

fig.text(0.5, 0.04, r"Bond Distance / \AA", va='center', ha='center', fontsize=plt.rcParams['axes.labelsize'])

for J in Js:
    axesJ[2].axhline(J/(2*np.pi), color=cmap(1-norm(J/(2*np.pi))))
    axesIQ[2].axhline(J/(2*np.pi), color=cmap(1-norm(J/(2*np.pi))))
    axesD[2].axhline(J/(2*np.pi), color=cmap(1-norm(J/(2*np.pi))))
axesJ[2].yaxis.tick_right()
axesJ[2].set_yscale("log", base=2)
axesJ[2].set_ylabel(r"$J_{\max}$ / GHz")
axesJ[2].yaxis.set_label_position("right")
axesJ[2].set_yticks(np.array(Js)/(2*np.pi))
axesJ[2].set_xticks([])

axesIQ[2].yaxis.tick_right()
axesIQ[2].set_yscale("log", base=2)
axesIQ[2].set_ylabel(r"$\textrm{IQ}_{\max}$ / $\frac{10}{\sqrt{2}}$ MHz")
axesIQ[2].yaxis.set_label_position("right")
axesIQ[2].set_yticks((2*np.pi)/np.array(Js))
axesIQ[2].set_xticks([])

axesD[2].yaxis.tick_right()
axesD[2].set_yscale("log", base=2)
axesD[2].set_ylabel(r"$\Delta B$ / 60 MHz")
axesD[2].yaxis.set_label_position("right")
axesD[2].set_yticks((2*np.pi)/np.array(Js))
axesD[2].set_xticks([])

axesJ[1].set_yticklabels([])
axesIQ[1].set_yticklabels([])
axesD[1].set_yticklabels([])

for axisIQ, im in zip(axesIQ[:2], images):
    newax = axisIQ.inset_axes([0.7, 0.975, 0.3, 0.3])
    newax.imshow(im)
    newax.axis('off')
plt.rcParams['axes.titley'] = 1
axesIQ[0].set_title(r"$\textrm{H}_2$", loc="left")
axesIQ[1].set_title(r"$\textrm{HeH}^+$", loc="left")

axesIQ[2].spines['top'].set_color('none')
axesIQ[2].spines['bottom'].set_color('none')
axesIQ[2].spines['left'].set_color('none')
axesIQ[2].spines['right'].set_color('none')
axesJ[2].spines['top'].set_color('none')
axesJ[2].spines['bottom'].set_color('none')
axesJ[2].spines['left'].set_color('none')
axesJ[2].spines['right'].set_color('none')
axesD[2].spines['top'].set_color('none')
axesD[2].spines['bottom'].set_color('none')
axesD[2].spines['left'].set_color('none')
axesD[2].spines['right'].set_color('none')
axesIQ[2].tick_params(length=0)
axesJ[2].tick_params(length=0)
axesD[2].tick_params(length=0)
axesJ[0].set_xticklabels([""]*3)
axesIQ[0].set_xticklabels([""]*3)
axesJ[1].set_xticklabels([""]*3)
axesIQ[1].set_xticklabels([""]*3)

fig.align_ylabels([axesJ[2], axesIQ[2], axesD[2]])

# Adjusting figure size
width = 3.40457
height = width*1.25
fig.set_size_inches(width, height)
fig.subplots_adjust(right=0.83, top=0.94, left=0.17, bottom=0.13, hspace=0.2)

# Saving the figure
fig.savefig(fig_path, dpi=1200)