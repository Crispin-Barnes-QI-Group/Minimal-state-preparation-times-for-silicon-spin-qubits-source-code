import os

import json
import matplotlib as mpl
import matplotlib.pylab as pl
import numpy as np

from matplotlib import pyplot as plt
from saveable_objects.extensions import SaveableWrapper
from scipy.optimize import curve_fit
from thread_chunks import Checkpoint
from tqdm import tqdm

from tasks.bootstrap import bootstrap
from tasks import setup_matplotlib
from src.setup.get_dir import DATA_DIR, FIG_DIR

data_dir = os.path.join(DATA_DIR, "figure_3_and_4")
fig_path = os.path.join(FIG_DIR, "figure_4.pdf")

paths = [os.path.join(data_dir, f"{qubits}_qubits.pkl") for qubits in [1, 2, 3, 4]]

# Fitting functions for the cumulative distribution of the MET.
def f(x, n):
    return np.power(np.sin(x), n)
def F(x, n):
    assert n == int(n)
    assert n >= 1
    if n == 1:
        return 1 - np.cos(x)
    if n == 2:
        return 0.5*(x - 0.5*np.sin(2*x))
    return ((n-1) * F(x,n-2) - f(x, n-1)*np.cos(x))/n
def composite(x, d, no_derivative, *args):
    v = args[0]
    c = np.array(args[1:])
    Fs = []
    fs = []
    Fn = []
    if no_derivative:
        fn = []
    for n in range(2*d-3, 2*d-3+len(c)+1+no_derivative):
        Fs.append(F(v*x/2, n))
        fs.append(f(v*x/2, n))
        Fn.append(F(np.pi/2, n))
        if no_derivative:
            fn.append(f(np.pi/2, n))
    fs = np.array(fs)
    Fs = np.array(Fs)
    Fn = np.array(Fn)
    F1 = Fn[0]
    if no_derivative:
        fn = np.array(fn)
        f1 = fn[0]
        f2 = fn[1]
        F2 = Fn[1]
        fn = fn[2:]
        Fn = Fn[2:]
        a = c@fn
        A = c@Fn
        c = np.concatenate([[(1-A+a*F2/f2)/(F1-f1*F2/f2),
                             (1-A+a*F1/f1)/(F2-f2*F1/f1) ],
                            c
                           ])
    else:
        Fn = Fn[1:]
        A = c@Fn
        c = np.concatenate([[(1-A)/F1], c])
    # add a constant offset so f is positive. This is a linear offset to F which will clearly be a bad fit and so will penialise the exploration of this space.
    return (c@Fs+np.min(np.concatenate([[0], c@fs*np.logical_and(0 <= x, x<=np.pi/v)]))*x) * np.logical_and(0 <= x, x<=np.pi/v) + (x>np.pi/v)

# Initialise figure
fig, axes = plt.subplots(1, 4)

# Initialise lists for storing fitting performance data
chis = []
terms = []

for qubits, Tmax, path, axis in zip([1, 2, 3, 4], [500, 290, 100, 60], paths, axes):
    # Load data
    SaveableList = SaveableWrapper[list];
    E = np.array(SaveableList.load(path, strict_typing=False)).T
    E = np.flip(np.array(E), axis=0)

    num_t=int(Tmax/2)
    do_bootstrap = True
    number_bootstrap_samples = 100000

    ts = np.linspace(0, Tmax, num_t)

    # Add gate times for benchmarking
    if qubits != 1:
        axis.axvline(0.25, color="#2E6E8E", linewidth=0.5)
    axis.axvline(200/qubits*np.sqrt(2), color="#2E6E8E", linewidth=0.5, label="Gate Benchmarks")
    axis.axvline(200/qubits, color="#2E6E8E", linewidth=0.5)

    # Compute the METs
    MET = np.nanmin(np.divide(np.expand_dims(ts, axis=-1), -np.log10(np.clip(E, 0, 1))>=7), axis=0)

    # Generate the distribution of the METs
    hist, bins = np.histogram(MET, ts)
    hist = hist/len(MET)
    cumulative = np.cumsum(hist)
    d = 2**qubits

    if do_bootstrap:
        lower_bound, upper_bound, sigmas = bootstrap(MET, ts, number_bootstrap_samples, 99.99)

        alpha = 0.25
        color = tuple(np.array([1, 1, 1])*(1-alpha)+alpha*np.array(mpl.colors.hex2color("#2E6E8E")))
        # Plot confidence intervals
        axis.fill_between(ts, [0]+list(lower_bound), [0]+list(upper_bound), color=color, edgecolor="none", label=f"99.99\% CI")

    # Plot the cumulative distribution  
    axis.plot(ts, [0]+list(cumulative), color="k", linewidth=0.5, label="Numerical Data")

    # Add fitted curves
    cmap = pl.cm.get_cmap('viridis')
    v = np.array(qubits*[np.pi/40])
    # Recursively fit initialising the parameters for the next order with those of the previous fit.
    N = 11
    for i in range(1, N):
        # The fit is better at the start of the distribution so for the first fit just use the first half of the data
        if i <=1:
            filter = np.logical_and(0 < cumulative, cumulative <= 0.5)
            quartile = cumulative[filter]
        else:
            filter = np.logical_and(0 < cumulative, cumulative < 1)
            quartile = cumulative[filter]
        try:
            if bootstrap:
                # Performing fit using the errors estimated via bootstrapping
                v_new, _ = curve_fit(lambda x, *args: composite(x, d, i!=1, *args), ts[1:][filter], quartile, v, sigma=sigmas[filter])
                chisq_new = np.sum(np.square((composite(ts[1:][filter], d, i!=1, *v)-quartile)/sigmas[filter]))/np.clip(len(quartile)-len(v[v!=0]), 0, None)
                if i <= 1 or chisq_new < chisq:
                    chisq = chisq_new
                    v = v_new
                else:
                    break
                if (chisq <= 1) and i > 1:
                    break
            else:
                v, _ = curve_fit(lambda x, *args: composite(x, d, i!=1, *args), ts[1:1+len(quartile)], quartile, v)
        except RuntimeError:
            print("Warning a runtime error occurred during the fit. Continuing plot.")
        print(f"i={i}, v={v}")
        if i == 1:
            # Plot the fit
            axis.plot(t_new := np.linspace(0, ts[-1], 1000), composite(t_new, d, i!=1, *v), linestyle=":", color= "#FE6100" if i != 1 else cmap(1), linewidth=1, label="HI Fit")
        else:
            v = np.concatenate([[v[0]], v[1:], [0]])
    axis.plot(t_new := np.linspace(0, ts[-1], 10030), composite(t_new, d, i!=1, *v), linestyle=":", color= "#FE6100" if i != 1 else cmap(1), linewidth=1, label="Higher Order Fit" if i != 1 else "HI Fit")
    terms.append(len(v[v!=0]))
    chis.append(chisq)

    # Set axis limits
    axis.set_xlim([0 if qubits == 1 else -200/qubits*np.sqrt(2)*0.04, 200/qubits*np.sqrt(2)+200/qubits*np.sqrt(2)*0.04])
    axis.set_ylim([0, 1])
    # Label axes
    axis.xaxis.set_ticks([0, int(np.round(200/qubits, -1))])
    axis.text(200/qubits+5/qubits, 0.5, "$\\frac{1}{\sqrt{2}}\left(X\pm Y\\right)$", rotation=90, horizontalalignment='left', verticalalignment='center')
    axis.text(200/qubits*np.sqrt(2), 0.5, "$X$ or $Y$", rotation=90, horizontalalignment='right', verticalalignment='center')
    if qubits != 1:
        axis.text(0.25+5/qubits, 0.5, r"$\sqrt{\textrm{SWAP}}$", rotation=90, horizontalalignment='left', verticalalignment='center')
    axis.set_xlabel("$T$ / ns")
    if qubits == 1:
        axis.set_ylabel(r"$\mathbb P\left(\textrm{MET}\le T\right)$")
    else:
        axis.yaxis.set_ticks([])
    axis.set_title(f"{qubits} Qubit" + ("s" if qubits != 1 else ""), fontsize=10, y=1)

# Add legeng
handles, _ = axes[0].get_legend_handles_labels()
handles = [handles[2], handles[1], handles[3], handles[4], handles[0]]
fig.legend(handles = handles, loc="lower center", ncol=5, handlelength=1)

# Set the figure size and adjust layout
width = 7.05826
height = width/3
fig.set_size_inches(width, height)
fig.subplots_adjust(right=0.955, top=0.9, left=0.08, bottom=0.32, wspace=0.06)

# Save the fit data
with open(os.path.join(data_dir, "fit_parameters.json"), "w") as file:
    json.dump({"chis":chis, "terms":terms}, file)

# Save the figure
fig.savefig(fig_path, dpi=1200)