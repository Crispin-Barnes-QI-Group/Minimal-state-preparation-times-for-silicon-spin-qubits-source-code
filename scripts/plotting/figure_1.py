import os
import pickle as pkl

import matplotlib as mpl
import matplotlib.colors as colors
import matplotlib.patches as mpatches
import numpy as np

from matplotlib import pyplot as plt
from matplotlib.cbook import get_sample_data
from matplotlib.collections import LineCollection
from qugradlab.pulses.sampling import SampleTimes

from tasks import setup_matplotlib
from src.setup.devices import get_device
from src.setup.get_dir import DATA_DIR, FIG_DIR
from src.setup.hamiltonian import Ham
from src.setup.molecules import h2 as mol
from src.setup.pulse_form import get_pulse_form

data_dir = os.path.join(DATA_DIR, "figure_1")
fig_path = os.path.join(FIG_DIR, "figure_1.pdf")

T_min = 0
T_max = 24
num_t = 307
n = 0
tau = 1
length =  10
initial_state_index =  1
r_min = 0.3
r_max = 3.3
num_r =  80
rs =  np.linspace(r_min, r_max, num_r)
ts =  np.linspace(T_min, T_max, num_t)

# Loading data
with open(os.path.join(data_dir, "H2.pkl"), "rb") as file:
    E = pkl.load(file)
with open(os.path.join(data_dir, "H2_pulse_parameters.pkl"), "rb") as file:
    xs = pkl.load(file)

for i, r in enumerate(rs):
    if r >= 0.735:
        break

E0 = Ham(mol(r)).min_eigenvalue
MET = np.nanmin(np.divide(ts, E[..., i]-E0 <= 3E-9))
argMET = np.nanargmin(np.divide(ts, E[..., i]-E0 <= 3E-9))

device = get_device(2, J_min=0, use_graph=False)
initial_state = device.hilbert_space.basis_vector(initial_state_index)

# Initialise figure
fig, axis = plt.subplots()

# Plot the energy error
# Cleaning the data
y = np.clip(E[..., i]-E0, 3E-9, None)
# Generating the coloured line segments
points = np.array([ts, y]).T.reshape(-1, 1, 2)
m=100
points = np.concatenate([np.stack([np.linspace(points[k, :, 0], points[k+1, :, 0], m), np.power(10, np.linspace(np.log10(points[k, :, 1]), np.log10(points[k+1, :, 1]), m))], axis=-1) for k in range(len(points)-1)], axis=0)
segments = np.concatenate([points[:-1], points[1:]], axis=1)
cmap = mpl.cm.get_cmap('cividis')
norm = colors.LogNorm(vmin=1e-10, vmax=0.29478983879696596)
lc = LineCollection(segments, cmap=cmap, norm=norm)
lc.set_array(points[..., 0, 1])
axis.add_collection(lc)

# Mark MET
axis.axvline(MET, linestyle="--", color="lightgrey", zorder=-1)

# Set the axis properties
axis.set_xlim([0, 3*MET])
axis.set_xlabel(r"Evolution Time $T$ / ns")
axis.set_yscale("log")
axis.set_ylim([1E-9,1E10])
axis.set_yticks([1E-9, 1E-7, 1E-5, 1E-3, 1E-1])
axis.set_ylabel("Energy Error\n/ Hartrees", loc="bottom")

# Add H2 image
imax = axis.inset_axes([-0.11, 0.8, 0.1, 0.2])
im = plt.imread(get_sample_data(os.path.join(FIG_DIR, "H2.png")))
imax.imshow(im)
imax.axis('off')

# Setup gradients for pulses
bottom_color = colors.to_rgb("#2E6E8E")
top_color = colors.to_rgb("#1E9C89")
potential_cmap = colors.LinearSegmentedColormap('new_cmap',segmentdata={
    'red': ((0.0, bottom_color[0], bottom_color[0]),(1.0, top_color[0], top_color[0])),
    'green': ((0.0, bottom_color[1], bottom_color[1]),(1.0, top_color[1], top_color[1])),
    'blue': ((0.0, bottom_color[2], bottom_color[2]),(1.0, top_color[2], top_color[2])),
    })

# Set times for the pulse annotations
X = [0.5*MET, 1.5*MET, 2.5*MET]

for j, T_index in enumerate([argMET-5, argMET, int(1.3*argMET)]):
    # Generate the pulse
    samples_per_point = (1+10000//length)
    sample_times = SampleTimes(T=ts[T_index], number_sample_points=samples_per_point*(length)+1)
    generate_pulse_form = get_pulse_form(length, ts[T_index], sample_times.dt, samples_per_point, n=n, tau=tau)
    driven_device = device.pulse_form(generate_pulse_form)
    pulses = driven_device.get_driving_pulses(xs[T_index, i], initial_state, 2, length)[0].real/(2*np.pi)

    # Plot line to inset
    axis.plot([ts[T_index], X[j]], [y[T_index], 1], "--", color=cmap(norm(y[T_index])))
    axis.scatter([ts[T_index]], [y[T_index]], color=cmap(norm(y[T_index])))
    # Create the insets for the pulse
    box = axis.inset_axes([j/3+0.035, 0.45, 1/3-0.065, 0.51])
    box.axis('off')
    box.add_artist(
        mpatches.FancyBboxPatch((0.06, 0.06),
                                0.88,
                                0.88,
                                edgecolor="k",
                                facecolor="lightgrey",
                                boxstyle=mpatches.BoxStyle("Round", pad=0.05)))
    J_axis = axis.inset_axes([j/3+0.095, 0.78, 1/3-0.15, 0.14])
    IQ_axis = axis.inset_axes([j/3+0.095, 0.58, 1/3-0.15, 0.14])

    # Plot the J pulse with colour gradient
    points = np.array([sample_times.t[:-1], pulses[:, 1]]).T.reshape(-1, 1, 2)
    m=0.1
    points = points[np.logical_or(np.concatenate([[True], np.diff(pulses[:, 1]) != 0]), np.concatenate([np.diff(pulses[:, 1]) != 0, [True]]))]
    points = np.concatenate([np.linspace(points[k], points[k+1], np.clip(int(np.abs(points[k+1,0, 1]-points[k, 0, 1])/m), 2, None)) for k in range(len(points)-1)], axis=0)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, cmap=potential_cmap, linewidth=1)
    lc.set_array(points[..., 0, 1])
    J_axis.add_collection(lc)
    # Add a baseline to the J puls
    J_axis.axhline(0, color=potential_cmap(1), linewidth=1)
    # Fill the area under the J pulse with a colour gradient
    polygon = J_axis.fill_between(sample_times.t[:-1], np.zeros_like(sample_times.t[:-1]),  pulses[:, 1], lw=0, color='none')
    verts = np.vstack([p.vertices for p in polygon.get_paths()])
    gradient = J_axis.imshow(np.flip(np.linspace(0, 1, 256).reshape(-1, 1)), cmap=potential_cmap, aspect='auto',
                          extent=[verts[:, 0].min(), verts[:, 0].max(), verts[:, 1].min(), verts[:, 1].max()], alpha=0.75)
    gradient.set_clip_path(polygon.get_paths()[0], transform=J_axis.transData)
    
    # Plot the IQ pulse
    IQ_axis.plot(sample_times.t[:-1], pulses[:, 0]*1E3, linewidth=0.5, color="#FE6100")

    # Set the axis properties
    J_axis.set_ylim([-0.5/(2*np.pi), 1+0.5/(2*np.pi)])
    J_axis.set_xlim([0, sample_times.t[-1]])
    IQ_axis.set_ylim([-10,10])
    IQ_axis.set_xlim([0, sample_times.t[-1]])
    IQ_axis.set_xlabel(r"Time $t$ / ns")
    IQ_axis.xaxis.set_label_coords(0.5,-0.36)
    if j == 0:
        J_axis.set_ylabel(r"$J_1$/GHz")
        IQ_axis.set_ylabel(r"$g$/MHz")
        J_axis.yaxis.set_label_coords(-0.35,0.6)
        IQ_axis.yaxis.set_label_coords(-0.35,0.1)
    J_axis.set_xticks([0, sample_times.t[-1]])
    J_axis.set_xticklabels(["", ""])
    IQ_axis.set_xticks([0, sample_times.t[-1]])
    IQ_axis.set_xticklabels(["0", np.round(sample_times.t[-1], 2)])

# Set figure size
width = 7.05826
height = width/3
fig.set_size_inches(width, height)
fig.subplots_adjust(right=0.998, top=0.99, left=0.105, bottom=0.2)

# Save the figure
fig.savefig(fig_path, dpi=1200)