import os

import numpy as np

from tasks.leakage import calculate_leakage
from src.setup.get_dir import DATA_DIR

data_dir = os.path.join(DATA_DIR, "figure_10")
figure_2_data_dir = os.path.join(DATA_DIR, "figure_2")
if not os.path.exists(os.path.join(figure_2_data_dir, "LiH.pkl")):
    import figure_2

calculate_leakage(2*np.pi*100, 2*np.pi*2400, "SiMOS", os.join(data_dir, "SiMOS_leakage.pkl"))