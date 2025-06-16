import os

from src.setup.get_dir import DATA_DIR

data_dir = os.path.join(DATA_DIR, "figure_3_and_4")

if not all([os.path.exists(os.path.join(data_dir, "1_qubits.pkl")),
            os.path.exists(os.path.join(data_dir, "2_qubits.pkl")),
            os.path.exists(os.path.join(data_dir, "3_qubits.pkl")),
            os.path.exists(os.path.join(data_dir, "4_qubits.pkl"))]):
    import figure_3

