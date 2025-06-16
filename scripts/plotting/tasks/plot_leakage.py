import os

import matplotlib.cm as cm
import numpy as np

from matplotlib import pyplot as plt
from saveable_objects.extensions import SaveableWrapper

from . import setup_matplotlib

SaveableDict = SaveableWrapper[dict];

def plot(data_path, fig_path):
    data = SaveableDict.load(data_path, strict_typing=False)

    fig, ax = plt.subplots(constrained_layout=True)

    num_colors = 5
    colors = cm.viridis(np.linspace(0, 1, num_colors))

    ax.plot(data["sample_times"], data["computational_occupation"], label="Computational", color=colors[0])
    ax.plot(data["sample_times"], data["valley_occupation"], label="Numerical Valley", color=colors[1])
    ax.plot(data["sample_times"], data["P_leakage"], color="k", alpha=0.35, label="Analytical Valley")
    ax.plot(data["sample_times"], data["double_occupation"], label="Double Occupation", color=colors[2])
    ax.plot(data["sample_times"], data["triple_occupation"], label="Triple Occupation", color=colors[3])
    ax.plot(data["sample_times"], data["quad_occupation"], label="Quad Occupation", color=colors[4])
    fig.legend(loc="outside lower center", ncol=2)
    ax.set_xlabel("Time (ns)")
    ax.set_yscale("log")
    ax.set_ylim([1E-18, 2])
    ax.set_ylabel("Population Probability")

    width = 3.40457
    height = 3.40457
    fig.set_size_inches(width, height)
    fig.savefig(fig_path, dpi=600)