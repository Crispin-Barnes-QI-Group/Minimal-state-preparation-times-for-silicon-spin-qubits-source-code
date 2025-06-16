import os

import numpy as np

from src.setup.devices import get_device
from src.setup.file_names import file_name_parameters
from src.setup.get_dir import DATA_DIR
from tasks.haar_sampling import haar_sample

data_dir = os.path.join(DATA_DIR, "figure_7")
figure_3_data_dir = os.path.join(DATA_DIR, "figure_3_and_4")

if not os.path.exists(os.path.join(figure_3_data_dir, "2_qubit.pkl")):
    import figure_3

# Default values
default_J_max = 2*np.pi
default_IQ_max = np.pi*0.02/(2*np.sqrt(2))
default_detuning=0.03

# Values to scan
multiples = [0.125, 0.25, 0.5, 2, 4, 8]
J_maxs = [2*np.pi*0.01] + [default_J_max*v for v in multiples]
IQ_maxs = [default_IQ_max*v for v in multiples]
detunings = [default_detuning*v for v in multiples]

T_maxs = [300]+[100]*6
num_ts = [50]*7
for J_max, T_max, num_t in zip(J_maxs, T_maxs, num_ts):
    haar_sample(get_device, 2, T_max, num_t, default_IQ_max, default_detuning, J_max,
                os.path.join(data_dir, f"{file_name_parameters(J_max, default_IQ_max, default_detuning)}.pkl"))

Tmaxs = [1400, 400, 200, 100, 100, 100]
numt_ts = [50]*6
for IQ_max, T_max, num_t in zip(IQ_maxs, T_maxs, num_ts):
    haar_sample(get_device, 2, T_max, num_t, IQ_max, default_detuning, default_J_max,
                os.path.join(data_dir, f"{file_name_parameters(default_J_max, IQ_max, default_detuning)}.pkl"))

Tmaxs = [800, 400, 200, 290, 100, 100, 150]
numt_ts = [50]*6
for detuning, T_max, num_t in zip(detunings, T_maxs, num_ts):
    haar_sample(get_device, 2, T_max, num_t, default_IQ_max, detuning, default_J_max,
                os.path.join(data_dir, f"{file_name_parameters(default_J_max, default_IQ_max, detuning)}.pkl"))