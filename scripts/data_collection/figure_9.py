import os

from functools import partial

import numpy as np

from src.setup.get_dir import DATA_DIR
from tasks.haar_sampling import haar_sample
from src.setup.devices import get_non_linear_zeeman_device

data_dir = os.path.join(DATA_DIR, "figure_9")
figure_3_data_dir = os.path.join(DATA_DIR, "figure_3_and_4")

if not os.path.exists(os.path.join(figure_3_data_dir, "4_qubit.pkl")):
    import figure_3

J_max = 2*np.pi
IQ_max = (np.pi*0.02/(2*np.sqrt(2)))
detuning = 0.03

get_device = partial(get_non_linear_zeeman_device, zeeman_splittings=2*np.pi*(np.linspace(28+0.03, 28-0.03, 4)[[1, 3, 0, 2]]+np.array([-0.002, 0, 0, 0.001])))
haar_sample(get_device, 4, 50, 50, IQ_max, detuning, J_max,
            os.path.join(data_dir, f"non_linear_zeeman_splittings.pkl"))