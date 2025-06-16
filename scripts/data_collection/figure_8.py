import os

from functools import partial

import numpy as np

from src.setup.file_names import file_name_parameters
from src.setup.get_dir import DATA_DIR
from tasks.haar_sampling import haar_sample
from src.setup.devices import get_drive_angle_device

data_dir = os.path.join(DATA_DIR, "figure_8")
figure_3_data_dir = os.path.join(DATA_DIR, "figure_3_and_4")
figure_7_data_dir = os.path.join(DATA_DIR, "figure_7")

J_max = 2*np.pi
IQ_max = (np.pi*0.02/(2*np.sqrt(2)))
detuning= 0.03

if not os.path.exists(os.path.join(figure_7_data_dir, f"{file_name_parameters(J_max, IQ_max/8, detuning)}.pkl")):
    import figure_7
elif not os.path.exists(os.path.join(figure_3_data_dir, "2_qubit.pkl")):
    # elif as figure_7.py will compute the figure_3 data if not found.
    import figure_3

get_device = partial(get_drive_angle_device, zeeman_splittings=0.5*np.array([[[1/8, 0, np.sqrt(1-np.square(1/8))]]*2]))
haar_sample(get_device, 2, 1200, 50, IQ_max, detuning, J_max,
            os.path.join(data_dir, f"drive_angled.pkl"))