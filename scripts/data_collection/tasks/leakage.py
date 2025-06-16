import os
import pickle as pkl
import signal

import numpy as np

from qugradlab.pulses.sampling import SampleTimes
from qugradlab.systems.semiconducting.esr import ValleyChain
from saveable_objects import SaveableObject
from saveable_objects.checkpointing import failed
from saveable_objects.extensions import SaveableWrapper
from scipy.linalg import expm

from src import config
from src.setup.devices import get_device
from src.setup.get_dir import DATA_DIR
from src.setup.hamiltonian import Ham
from src.setup.molecules import lih as mol
from src.setup.pulse_form import get_pulse_form

figure_2_data_dir = os.path.join(DATA_DIR, "figure_2")

ValleyChain = SaveableWrapper[ValleyChain];
SaveableDict = SaveableWrapper[dict];

def calculate_leakage(valley_splitting, u, name, path):
    with open(os.path.join(figure_2_data_dir, "LiH_pulse_parameters.pkl"), "rb") as file:
        xs = pkl.load(file)
    with open(os.path.join(figure_2_data_dir, "LiH.pkl"), "rb") as file:
        Es = pkl.load(file)

    J_max = 2*np.pi
    drive = np.pi*0.02/(2*np.sqrt(2))
    detuning = 0.03
    T_max = 50
    qudits = 4
    T_min=0
    num_t = 128
    n=0
    tau=1
    length = 10
    initial_state_index = 3
    r_min = 1.2
    r_max = 3.3
    num_r = 80

    rs = np.linspace(r_min, r_max, num_r)
    ts = np.linspace(T_min, T_max, num_t)
    r_index = np.argmin(np.abs(rs-1.5949))
    print(rs[r_index])
    cost_hamiltonian_class = Ham(mol(rs[r_index]))
    E0 = cost_hamiltonian_class.min_eigenvalue
    T_index = np.argmax(Es[:,r_index]-E0<=1e-7)
    t = ts[T_index]
    print(t)

    x = xs[T_index, r_index]

    device = get_device(qudits, J_min=0, J_max=J_max, max_drive_strength=drive, detuning=detuning, use_graph=False)
    valley_device, loaded = ValleyChain.loadifparams(dots=qudits,
                                                     electrons=qudits,
                                                     max_drive_strength=np.pi*0.02/(2*np.sqrt(2)),
                                                     J_min=0,
                                                     J_max=2*np.pi,
                                                     zeeman_splittings=(zeeman_splittings := (np.array([2*np.pi*28]) if qudits == 1 else np.linspace(2*np.pi*(28+detuning), 2*np.pi*(28-detuning), qudits))),
                                                     valley_splitting=valley_splitting,
                                                     u = u,
                                                     u_valley_flip=u/60,
                                                     valley_spin_orbit_coupling=1E-4*valley_splitting,
                                                     path=(file_path:=os.path.join(DATA_DIR, f"{name, path}_valley_model_{qudits}_dots.pkl")))
    print(f"Loaded Hamiltonian from {file_path}" if loaded else f"Generated and saved Hamiltonain to {file_path}")

    initial_state = device.hilbert_space.basis_vector(initial_state_index)
    valley_initial_state = valley_device.hilbert_space.basis_vector(valley_device.hilbert_space[valley_device.hilbert_space.computational_projector(4)][device.hilbert_space.dim-1-initial_state_index])
    sample_times = SampleTimes(T=t, number_sample_points=int(np.ceil(t*u*10/length)*length)+1)
    generate_pulse_form = get_pulse_form(length, t, sample_times.dt, int(np.ceil(t*u*10/length)), n=n, tau=tau)
    driven_device = device.pulse_form(generate_pulse_form)
    drive_valley_device = valley_device.pulse_form(generate_pulse_form)
    U = expm(-1j*(sample_times.T)*driven_device.H0)
    U_valley = np.diag(np.exp(-1j*(sample_times.T)*np.diagonal(drive_valley_device.H0)))
    H = valley_device.hilbert_space.dialate_hamiltonian(U@cost_hamiltonian_class.H@U.T.conj(), device)
    X = np.array([[0, 1], [1, 0]])
    S = np.kron(np.kron(np.kron(X, X), X), X)
    valley_H = U_valley@valley_device.hilbert_space.dialate_hamiltonian(S@cost_hamiltonian_class.H@S, valley_device, levels=4)@U_valley.T.conj()

    if failed(states := SaveableObject.tryload(os.path.join(DATA_DIR, f"states.pkl"), strict_typing=False)):
        states = driven_device.propagate_all(x, initial_state, qudits, length)
        SaveableObject._save(states, f"states.pkl")

    if failed(valley_states := SaveableObject.tryload(os.path.join(DATA_DIR, f"{name, path}_states.pkl"), strict_typing=False)):
        valley_states = drive_valley_device.propagate_all(x, valley_initial_state, qudits, length)
        SaveableObject._save(valley_states, f"{name, path}_states.pkl")

    final_state = states[:, -1]
    valley_final_state = valley_states[:, -1]

    print(f"time step: {sample_times.dt} ns")

    data = SaveableDict({}, path=path)

    # Analytical model
    splittings = valley_splitting-(zeeman_splittings*np.array([1, 1, -1, -1])) # The array of 1s and -1s accounts for whether the input state is up or down.
    coupling = 1E-4*valley_splitting
    pre_factor = np.expand_dims(coupling**2/(np.square(0.5*splittings)+coupling**2), axis=-1)
    P_leakage = pre_factor*np.square(np.sin(np.multiply.outer(np.sqrt(np.square(0.5*splittings)+coupling**2), sample_times.t))) # the 0.5 in the sign is because we are working with splittings not couplings
    P_no_leagage = 1-P_leakage
    data["P_leakage"] = 1-np.prod(P_no_leagage, axis=0)

    P = np.square(np.abs(valley_states))

    computational_filter = valley_device.hilbert_space.computational_projector(4)
    valley_filter = np.logical_and(np.logical_not(computational_filter), valley_device.hilbert_space.single_occupation_states(4))
    double_occupation_filter = valley_device.hilbert_space.n_occupation_states(4, 2)
    triple_occupation_filter = valley_device.hilbert_space.n_occupation_states(4, 3)
    quad_occupation_filter = valley_device.hilbert_space.n_occupation_states(4, 4)

    data["computational_occupation"] = np.sum(P[computational_filter].T, axis=-1)
    data["valley_occupation"] = np.sum(P[valley_filter].T, axis=-1)
    data["double_occupation"] = np.sum(P[double_occupation_filter].T, axis=-1)
    data["triple_occupation"] = np.sum(P[triple_occupation_filter].T, axis=-1)
    data["quad_occupation"] = np.sum(P[quad_occupation_filter].T, axis=-1)
    data["sample_times"] = sample_times.t

    try:
        signal.pthread_sigmask(signal.SIG_BLOCK, [signal.SIGINT])
        print("Interupt signals have been blocked while the data is saved.")
        data.save()
    except BaseException as e:
        print("There was an error while saving the data.")
        raise e
    else:
        print("Data has saved successfully.")
    finally:
        print("Interupt signals have been unblocked.")
        signal.pthread_sigmask(signal.SIG_UNBLOCK, [signal.SIGINT])

    print(f"Heisenberg Model Energy: {(final_state.conj().T@H@final_state).real-cost_hamiltonian_class.min_eigenvalue}")
    print(f"Valley Model Energy: {(valley_final_state.conj().T@valley_H@valley_final_state).real-cost_hamiltonian_class.min_eigenvalue}")
    print(f"Valley Model Renormalised Energy: {(valley_final_state.conj().T@valley_H@valley_final_state/np.sum(np.sum(P[computational_filter].T[-1]))).real-cost_hamiltonian_class.min_eigenvalue}")