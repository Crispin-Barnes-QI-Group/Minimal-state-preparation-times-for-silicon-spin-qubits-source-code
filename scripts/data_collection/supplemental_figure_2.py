import os

import numpy as np

from tasks.leakage import calculate_leakage
from src.setup.get_dir import DATA_DIR

data_dir = os.path.join(DATA_DIR, "supplemental_figure_2")
figure_2_data_dir = os.path.join(DATA_DIR, "figure_2")
if not os.path.exists(os.path.join(figure_2_data_dir, "LiH.pkl")):
    import figure_2

calculate_leakage(2*np.pi*30, 2*np.pi*1400, "SiGe_off_resonance", os.path.join(data_dir, "SiGe_off_resonance_leakage.pkl"))