import os

import numpy as np

from src.setup.file_names import file_name_parameters
from src.setup.get_dir import DATA_DIR
from tasks.molecule_scan import molecule_scan
from src.setup.molecules import h2, hehp

data_dir = os.path.join(DATA_DIR, "figure_6")
figure_1_data_dir = os.path.join(DATA_DIR, "figure_1")
figure_2_data_dir = os.path.join(DATA_DIR, "figure_2")

if not os.path.exists(os.path.join(figure_1_data_dir, "H2.pkl")):
    import figure_1
if not os.path.exists(os.path.join(figure_2_data_dir, "HeH.pkl")):
    import figure_2

# Default values
default_J_max = 2*np.pi
default_IQ_max = np.pi*0.02/(2*np.sqrt(2))
default_detuning=0.03

# Values to scan
multiples = [0.125, 0.25, 0.5, 2, 4, 8]
J_maxs = [2*np.pi*0.01] + [default_J_max*v for v in multiples]
IQ_maxs = [default_IQ_max*v for v in multiples]
detunings = [default_detuning*v for v in multiples]

T_maxs = [100]+[20]+[10]*5
num_ts = [256]*2+[128]*5
for J_max, T_max, num_t in zip(J_maxs, T_maxs, num_ts):
    molecule_scan(h2, T_max, num_t, default_IQ_max, default_detuning, J_max, 1, 0.3, 3.3, 80, True,
                  os.path.join(data_dir, f"H2{file_name_parameters(J_max, default_IQ_max, default_detuning)}.pkl"),
                  os.path.join(data_dir, f"H2{file_name_parameters(J_max, default_IQ_max, default_detuning)}_parameters.pkl"),
                  os.path.join(data_dir, f"H2{file_name_parameters(J_max, default_IQ_max, default_detuning)}_checkpoint.pkl"))
T_maxs = [20]+[10]*6
num_ts = [256]+[128]*6
for J_max, T_max, num_t in zip(J_maxs, T_maxs, num_ts):
    molecule_scan(hehp, T_max, num_t, default_IQ_max, default_detuning, J_max, 1, 0.65, 3.3, 80, False,
                  os.path.join(data_dir, f"HeH{file_name_parameters(J_max, default_IQ_max, default_detuning)}.pkl"),
                  os.path.join(data_dir, f"HeH{file_name_parameters(J_max, default_IQ_max, default_detuning)}_parameters.pkl"),
                  os.path.join(data_dir, f"HeH{file_name_parameters(J_max, default_IQ_max, default_detuning)}_checkpoint.pkl"))

T_maxs = [25, 20, 10, 20, 5, 5]
num_ts = [320, 256, 128, 256, 64, 64]
for IQ_max, T_max, num_t in zip(IQ_maxs, T_maxs, num_ts):
    molecule_scan(h2, T_max, num_t, IQ_max, default_detuning, default_J_max, 1, 0.3, 3.3, 80, True,
                  os.path.join(data_dir, f"H2{file_name_parameters(default_J_max, IQ_max, default_detuning)}.pkl"),
                  os.path.join(data_dir, f"H2{file_name_parameters(default_J_max, IQ_max, default_detuning)}_parameters.pkl"),
                  os.path.join(data_dir, f"H2{file_name_parameters(default_J_max, IQ_max, default_detuning)}_checkpoint.pkl"))
T_maxs = [40, 20, 10, 20, 5, 5]
num_ts = [512, 256, 128, 256, 64, 64]
for IQ_max, T_max, num_t in zip(IQ_maxs, T_maxs, num_ts):
    molecule_scan(hehp, T_max, num_t, IQ_max, default_detuning, default_J_max, 1, 0.65, 3.3, 80, False,
                  os.path.join(data_dir, f"HeH{file_name_parameters(default_J_max, IQ_max, default_detuning)}.pkl"),
                  os.path.join(data_dir, f"HeH{file_name_parameters(default_J_max, IQ_max, default_detuning)}_parameters.pkl"),
                  os.path.join(data_dir, f"HeH{file_name_parameters(default_J_max, IQ_max, default_detuning)}_checkpoint.pkl"))

T_maxs = [20, 10, 10, 10, 10, 10]
num_ts = [256, 128, 128, 128, 128, 128]
for detuning, T_max, num_t in zip(detunings, T_maxs, num_ts):
    molecule_scan(h2, T_max, num_t, default_IQ_max, detuning, default_J_max, 1, 0.3, 3.3, 80, True,
                  os.path.join(data_dir, f"H2{file_name_parameters(default_J_max, default_IQ_max, detuning)}.pkl"),
                  os.path.join(data_dir, f"H2{file_name_parameters(default_J_max, default_IQ_max, detuning)}_parameters.pkl"),
                  os.path.join(data_dir, f"H2{file_name_parameters(default_J_max, default_IQ_max, detuning)}_checkpoint.pkl"))
T_maxs = [20]*2+[10]*4
num_ts = [256]*2+[128]*4
for detuning, T_max, num_t in zip(detunings, T_maxs, num_ts):
    molecule_scan(hehp, T_max, num_t, default_IQ_max, detuning, default_J_max, 1, 0.65, 3.3, 80, False,
                  os.path.join(data_dir, f"HeH{file_name_parameters(default_J_max, default_IQ_max, detuning)}.pkl"),
                  os.path.join(data_dir, f"HeH{file_name_parameters(default_J_max, default_IQ_max, detuning)}_parameters.pkl"),
                  os.path.join(data_dir, f"HeH{file_name_parameters(default_J_max, default_IQ_max, detuning)}_checkpoint.pkl"))