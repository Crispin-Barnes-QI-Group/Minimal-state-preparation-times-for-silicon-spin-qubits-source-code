import os

import numpy as np

from tasks.leakage import calculate_leakage
from src.setup.get_dir import DATA_DIR

data_dir = os.path.join(DATA_DIR, "supplemental_figure_1")
figure_2_data_dir = os.path.join(DATA_DIR, "figure_2")
if not os.path.exists(os.path.join(figure_2_data_dir, "LiH.pkl")):
    import figure_2

qudits = 4
detuning = 0.03
zeeman_splittings = (np.array([2*np.pi*28]) if qudits == 1 else np.linspace(2*np.pi*(28+detuning), 2*np.pi*(28-detuning), qudits))
calculate_leakage(np.max(zeeman_splittings), 2*np.pi*1400, "SiGe_on_resonance", os.path.join(data_dir, "SiGe_on_resonance_leakage.pkl"))