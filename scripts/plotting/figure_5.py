import os

import matplotlib.pyplot as plt
import numpy as np

from matplotlib import cm, colors
from matplotlib import pyplot as plt
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
from qugradlab.pulses.sampling import SampleTimes
from thread_chunks import Checkpoint
from tqdm import tqdm

from src import config

from src.setup.devices import get_device
from src.setup.get_dir import DATA_DIR, FIG_DIR
from src.setup.pulse_form import get_pulse_form
from tasks import setup_matplotlib

data_path = os.path.join(DATA_DIR, "figure_5/Bloch_sphere_scan_checkpoint.pkl")
fig_path = os.path.join(FIG_DIR, "figure_5.png")

# Loding data
data = Checkpoint.load(data_path)

class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))

        return np.min(zs)

Tmax=290
num_t=290
length=40
n=0
tau=1
qudits=1
levels=2
xs = np.array([np.zeros((290, 81)) if o is None else o[1] for o in  tqdm(data.output, desc="Unpacking")]).T
xs = np.flip(np.array(xs), axis=1)
E = np.array([np.zeros(num_t) if o is None else o[0] for o in  tqdm(data.output, desc="Unpacking")]).T
E = np.flip(np.array(E), axis=0)
theta_res = 30
phi_res = 30
ts = np.linspace(0, Tmax, num_t)
MET = np.nanmin(np.divide(np.expand_dims(ts, axis=-1), -np.log10(np.clip(E, 0, 1))>=7), axis=0)
MET = MET.reshape(theta_res, phi_res)
fMET = np.flip(MET, axis=1)
MET = np.concatenate([MET, fMET]*4, axis=1)

fig = plt.figure(layout="constrained")
gs = fig.add_gridspec(ncols=5, nrows=1, width_ratios=[15, 1, 15, 15, 1])
ax = fig.add_subplot(gs[0], projection='3d')
ax.set_axis_off()
ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), np.diag([s:=1.4, s, 1.1*s, 1]))

u, v = np.mgrid[0:np.pi:theta_res*1j, 0:2*np.pi:phi_res*8j]


class CustomNorm(colors.Normalize):
    def __init__(self, MET, a, clip=None):
        colors.Normalize.__init__(self, None, None, clip)
        self._mid = np.nanmax(MET)/2
        self._func=lambda x: np.tanh(a*x/self._mid)
        self._inv_func=lambda x: self._mid*np.arctanh(x)/a
        self._lower = np.nanmin(self._func(MET-self._mid))
        self._upper = np.nanmax(self._func(MET-self._mid))

    def __call__(self, x, clip=None):
        return  (self._func(x-self._mid)-self._lower)/(self._upper-self._lower)

    def inverse(self, x):
        return self._inv_func((self._upper-self._lower)*x+self._lower)+self._mid

norm = CustomNorm(MET=MET, a=4)
cax = fig.add_subplot(gs[1])
y = np.expand_dims(norm(np.linspace(0, np.nanmax(MET), 1000)), axis=-1)
cax.imshow(y, extent=[0, 1, 0, np.nanmax(MET)], aspect="auto")
cax.set_xticks([])
cax.yaxis.tick_right()
cax.yaxis.set_label_position("right")
cax.set_ylabel("MET / ns")



x = np.sin(u) * np.cos(-v)
y = np.sin(u) * np.sin(-v)
z = np.cos(u)

a = Arrow3D([0.975, 1.4], [0, 0], 
                [0, 0], mutation_scale=10, 
                lw=1, arrowstyle="->", color="k")

ax.text(1.4, 0, 0, "$X$", horizontalalignment="right", verticalalignment="center")

ax.add_artist(a)

a = Arrow3D([0, 0], [0.975, 1.4], 
                [0, 0], mutation_scale=10, 
                lw=1, arrowstyle="->", color="k")

ax.text(0, 1.4, 0, "$Y$", horizontalalignment="left", verticalalignment="center")

ax.add_artist(a)

ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap="viridis_r",
                       linewidth=0, antialiased=False,
                       facecolors=cm.viridis_r(norm(MET)))

a = Arrow3D([0, 0], [0, 0], 
                [0.975, 1.4], mutation_scale=10, 
                lw=1, arrowstyle="->", color="k")

ax.text(0, 0, 1.4, "$Z$", horizontalalignment="center")
ax.text2D(0, 1, "a)", verticalalignment="top", transform=ax.transAxes)

ax.add_artist(a)

ax.set_zlim([-1, 1.4])

ax.view_init(10, 45, 0)


MET = np.nanmin(np.divide(np.expand_dims(ts, axis=-1), -np.log10(np.clip(E, 0, 1))>=7), axis=0)
argMET = np.nanargmin(np.divide(np.expand_dims(ts, axis=-1), -np.log10(np.clip(E, 0, 1))>=7), axis=0)

ax2 = fig.add_subplot(gs[2], projection='3d')
ax2.set_axis_off()
ax2.get_proj = lambda: np.dot(Axes3D.get_proj(ax2), np.diag([s:=1.2, s, s, 1]))

device = get_device(qudits, J_min=0, use_graph=False)
initial_state = device.hilbert_space.basis_vector(0)
for i, a, m, output in tqdm(zip(range(len(MET)), argMET, MET, data.output)):
    x = xs[:,a,i]
    if output is not None:
        samples_per_point = (1+10000//length)
        sample_times = SampleTimes(T=m, number_sample_points=samples_per_point*(length))
        generate_pulse_form = get_pulse_form(length, m, sample_times.dt, samples_per_point, n=n, tau=tau)
        driven_device = device.pulse_form(generate_pulse_form)
        psis = driven_device.propagate_all(x, initial_state, qudits, length)[:, 1:]
        psis = np.einsum("ijt,jt->it",np.expand_dims(np.identity(2), axis=-1)*np.exp(1j*np.multiply.outer(device.H0,sample_times.t)), psis) #can do as H0 is diagonal
        X = np.einsum("it,ij,jt->t", psis.conj(), np.array([[0, 1], [1, 0]]), psis).real
        Y = np.einsum("it,ij,jt->t", psis.conj(), np.array([[0, -1j], [1j, 0]]), psis).real
        Z = np.einsum("it,ij,jt->t", psis.conj(), np.array([[1, 0], [0, -1]]), psis).real

        ax2.plot(X, Y, Z, alpha=0.01, color="k", linewidth=2)
        ax2.plot(Y, -X, Z, alpha=0.01, color="k", linewidth=2)
        ax2.plot(-X, -Y, Z, alpha=0.01, color="k", linewidth=2)
        ax2.plot(-Y, X, Z, alpha=0.01, color="k", linewidth=2)
        ax2.plot(-X, Y, Z, alpha=0.01, color="k", linewidth=2)
        ax2.plot(-Y, -X, Z, alpha=0.01, color="k", linewidth=2)
        ax2.plot(X, -Y, Z, alpha=0.01, color="k", linewidth=2)
        ax2.plot(Y, X, Z, alpha=0.01, color="k", linewidth=2)
        

a = Arrow3D([0.975, 1.4], [0, 0], 
                [0, 0], mutation_scale=10, 
                lw=1, arrowstyle="->", color="k")

ax2.text(1.4, 0, 0, "$X$", horizontalalignment="left", verticalalignment="center")

ax2.add_artist(a)

a = Arrow3D([0, 0], [0.975, 1.4], 
                [0, 0], mutation_scale=10, 
                lw=1, arrowstyle="->", color="k")

ax2.text(0, 1.4, 0, "$Y$", horizontalalignment="center")

ax2.add_artist(a)

ax2.text2D(0, 1, "b)", verticalalignment="top", transform=ax2.transAxes)

ax2.set_xlim([-1, 1.4])
ax2.set_ylim([-1, 1.4])

ax2.view_init(90, -90, 0)


class CustomNorm2(colors.Normalize):
    def __init__(self, MET, a, clip=None):
        colors.Normalize.__init__(self, None, None, clip)
        self._mid = np.nanmax(MET)
        self._func=lambda x: np.arctan(a*x/self._mid)+np.pi/2
        self._inv_func=lambda x: self._mid*np.tan(x-np.pi/2)/a
        self._lower = self._func(-self._mid)
        self._upper = np.nanmax(self._func(MET-self._mid))

    def __call__(self, x, clip=None):
        return  (self._func(x-self._mid)-self._lower)/(self._upper-self._lower)

    def inverse(self, x):
        return self._inv_func((self._upper-self._lower)*x+self._lower)+self._mid

ax3 = fig.add_subplot(gs[3], projection='3d')
ax3.set_axis_off()
ax3.get_proj = lambda: np.dot(Axes3D.get_proj(ax3), np.diag([s:=1.25, s, s, 1]))
norm = CustomNorm2(MET=MET[420:450], a=4.5)
cax3 = fig.add_subplot(gs[4])
y = np.expand_dims(norm(np.linspace(0, np.nanmax(MET[420:450]), 1000)), axis=-1)
cax3.imshow(y, extent=[0, 1, 0, np.nanmax(MET[420:450])], aspect="auto", cmap=cm.viridis_r, origin="lower")
cax3.set_xticks([])
cax3.yaxis.tick_right()
cax3.yaxis.set_label_position("right")
cax3.set_ylabel("$t$ / ns")

for i, a, m, output in tqdm(zip(range(len(MET)), argMET, MET, data.output)):
    if i < 420:
        continue
    if i >= 450:
        break
    x = xs[:,a,i]
    if output is not None:
        samples_per_point = (1+10000//length)
        sample_times = SampleTimes(T=m, number_sample_points=samples_per_point*(length))
        generate_pulse_form = get_pulse_form(length, m, sample_times.dt, samples_per_point, n=n, tau=tau)
        driven_device = device.pulse_form(generate_pulse_form)
        psis = driven_device.propagate_all(x, initial_state, qudits, length)[:, 1:]
        psis = np.einsum("ijt,jt->it",np.expand_dims(np.identity(2), axis=-1)*np.exp(1j*np.multiply.outer(device.H0,sample_times.t)), psis) #can do as H0 is diagonal
        # psis = psis[:,::40]
        X = np.einsum("it,ij,jt->t", psis.conj(), np.array([[0, 1], [1, 0]]), psis).real
        Y = np.einsum("it,ij,jt->t", psis.conj(), np.array([[0, -1j], [1j, 0]]), psis).real
        Z = np.einsum("it,ij,jt->t", psis.conj(), np.array([[1, 0], [0, -1]]), psis).real
        for a in [1, -1]:
            for b in [1, -1]:
                points = np.expand_dims(np.stack([a*X, b*Y, Z], axis=-1), axis=1)
                points = np.flip(np.flip(points, axis=0)[::40], axis=0)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)
                for ii in range(len(segments)):
                    segii=segments[ii]
                    lii,=ax3.plot(segii[:,0],segii[:,1],segii[:,2],color=cm.viridis_r(norm(np.flip(np.flip(sample_times.t, axis=0)[::40], axis=0)[ii])),linewidth=0.8, alpha=0.2)
                    lii.set_solid_capstyle('round')
            for a in [1, -1]:
                for b in [1, -1]:
                    points = np.expand_dims(np.stack([a*Y, b*X, Z], axis=-1), axis=1)
                    points = np.flip(np.flip(points, axis=0)[::40], axis=0)
                    segments = np.concatenate([points[:-1], points[1:]], axis=1)
                    for ii in range(len(segments)):
                        segii=segments[ii]
                        lii,=ax3.plot(segii[:,0],segii[:,1],segii[:,2],color=cm.viridis_r(norm(np.flip(np.flip(sample_times.t, axis=0)[::40], axis=0)[ii])),linewidth=0.8, alpha=0.2)
                        lii.set_solid_capstyle('round')

a = Arrow3D([0.975, 1.4], [0, 0], 
                [0, 0], mutation_scale=10, 
                lw=1, arrowstyle="->", color="k")

ax3.text(1.4, 0, 0, "$X$", horizontalalignment="left", verticalalignment="center")

ax3.add_artist(a)

a = Arrow3D([0, 0], [0.975, 1.4], 
                [0, 0], mutation_scale=10, 
                lw=1, arrowstyle="->", color="k")

ax3.text(0, 1.4, 0, "$Y$", horizontalalignment="center")

ax3.add_artist(a)

ax3.text2D(0, 1, "c)", verticalalignment="top", transform=ax3.transAxes)

ax3.set_xlim([-1, 1.4])
ax3.set_ylim([-1, 1.4])

ax3.view_init(90, -90, 0)

# Set the figure size
width = 7.05826
height = width/3.5
fig.set_size_inches(width, height)

# Save the figure
plt.savefig(fig_path, dpi=600)