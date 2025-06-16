import os

import numpy as np

from src.setup.molecules import h2
from tasks.molecule_scan import molecule_scan
from src.setup.get_dir import DATA_DIR

data_dir = os.path.join(DATA_DIR, "figure_1")

J_max = 2*np.pi
drive = np.pi*0.02/(2*np.sqrt(2))
detuning=0.03

molecule_scan(h2, 24, 307, drive, detuning, J_max, 1, 0.3, 3.3, 80, True,
              os.path.join(data_dir, "H2.pkl"),
              os.path.join(data_dir, "H2_pulse_parameters.pkl"),
              os.path.join(data_dir, "H2_checkpoint.pkl"))