import os

import numpy as np

from src.setup.molecules import hehp, lih
from tasks.molecule_scan import molecule_scan
from src.setup.get_dir import DATA_DIR

data_dir = os.path.join(DATA_DIR, "figure_2")
figure_1_data_dir = os.path.join(DATA_DIR, "figure_1")

if not os.path.exists(os.path.join(figure_1_data_dir, "H2.pkl")):
    import figure_1

J_max = 2*np.pi
drive = np.pi*0.02/(2*np.sqrt(2))
detuning=0.03

molecule_scan(hehp, 10, 128, drive, detuning, J_max, 1, 0.65, 3.3, 80, False,
              os.path.join(data_dir, "HeH.pkl"),
              os.path.join(data_dir, "HeH_pulse_parameters.pkl"),
              os.path.join(data_dir, "HeH_checkpoint.pkl"))
molecule_scan(lih, 50, 128, drive, detuning, J_max, 3, 1.2, 3.3, 80, True,
              os.path.join(data_dir, "LiH.pkl"),
              os.path.join(data_dir, "LiH_pulse_parameters.pkl"),
              os.path.join(data_dir, "LiH_checkpoint.pkl"))