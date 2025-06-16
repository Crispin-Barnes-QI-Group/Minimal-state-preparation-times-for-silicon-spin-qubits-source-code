import os

import numpy as np

from tasks.haar_sampling import haar_sample
from src.setup.get_dir import DATA_DIR
from src.setup.devices import get_device

data_dir = os.path.join(DATA_DIR, "figure_3_and_4")

J_max = 2*np.pi
IQ_max = np.pi*0.02/(2*np.sqrt(2))
detuning =  0.03

T_maxs = [500, 290, 100, 60]
num_ts = [250, 145, 50, 30]

for qubits, T_max, num_t in zip([1, 2, 3, 4], T_maxs, num_ts):
    haar_sample(get_device, qubits, T_max, num_t, IQ_max, detuning, J_max,
                os.path.join(data_dir, f"{qubits}_qubits.pkl"))