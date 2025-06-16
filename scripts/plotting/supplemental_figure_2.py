import os

from src.setup.get_dir import DATA_DIR, FIG_DIR
from tasks.plot_leakage import plot

data_path = os.path.join(DATA_DIR, "supplemental_figure_2", "SiGe_off_resonance_leakage.pkl")
fig_path = os.path.join(FIG_DIR, "supplemental_figure_2.png")

plot(data_path, fig_path)