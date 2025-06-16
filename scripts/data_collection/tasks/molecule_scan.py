import os
import pickle as pkl

import numpy as np
import tensorflow as tf

from qugradlab.pulses.invertible_functions.packaging import pack
from qugradlab.pulses.sampling import SampleTimes
from scipy.linalg import expm
from scipy.optimize import minimize
from thread_chunks import chunk

import src.config as config

from src.setup.devices import get_device
from src.setup.hamiltonian import Ham
from src.setup.pulse_form import get_pulse_form, freq_scaling

def molecule_scan(mol, T_max, num_t, drive, detuning, J_max, initial_state_index, r_min, r_max, num_r, r_forward, energy_path, parameter_path, checkpoint_path):
    max_error = 1E-8
    method = "BFGS"
    qudits = int(np.log2(len(mol())))
    T_min=0
    n=0
    tau=1
    length = 10
    rs = np.linspace(r_min, r_max, num_r)
    ts = np.linspace(T_min, T_max, num_t)
    runs = 3

    device = get_device(qudits, J_min=0, J_max=J_max, max_drive_strength=drive, detuning=detuning, use_graph=False)
    def get_row(t):
        E = []
        xs = []
        J_signals = -np.ones((length, qudits-1))
        drive_signals = np.zeros((length, qudits, 2))
        
        x = pack([J_signals, freq_scaling.inverse(device.zeeman_splittings), drive_signals])
        initial_state = device.hilbert_space.basis_vector(initial_state_index)
        samples_per_point = (1+10000//length)
        sample_times = SampleTimes(T=t, number_sample_points=samples_per_point*(length)+1)
        generate_pulse_form = get_pulse_form(length, t, sample_times.dt, samples_per_point, n=n, tau=tau)
        driven_device = device.pulse_form(generate_pulse_form)
        del generate_pulse_form
        iterator = rs if r_forward else reversed(rs)
        for r in iterator:
            cost_hamiltonian_class = Ham(mol(r))
            H0 = device.hilbert_space.dialate_operator(cost_hamiltonian_class.H)
            del cost_hamiltonian_class
            U = expm(-1j*(sample_times.T)*driven_device.H0)
            H = U@H0@U.T.conj()
            del H0
            del U
            result = minimize(driven_device.gradient,
                            x,
                            args=(initial_state, qudits, length, H),
                            method=method,
                            jac=True,
                            options={"gtol": max_error})
            x = tf.clip_by_value(result.x, -1, 1)
            xs.append(x.numpy())
            E.append(result.fun)
            del result
            for _ in range(runs):
                J_signals = 2*np.random.rand(length, qudits-1)-1
                drive_signals = 2*np.random.rand(length, qudits, 2)-1
                x_rand = pack([J_signals, freq_scaling.inverse(device.zeeman_splittings), drive_signals])
                rand_result = minimize(driven_device.gradient,
                                x_rand,
                                args=(initial_state, qudits, length, H),
                                method=method,
                                jac=True,
                                options={"gtol": max_error})
                del x_rand
                if rand_result.fun < E[-1]:
                    E[-1] = rand_result.fun
                    x = tf.clip_by_value(rand_result.x, -1, 1)
                    xs[-1] = x
                del rand_result
            del H
        if not r_forward:
            xs = list(reversed(xs))
            E = list(reversed(E))
        del driven_device
        del initial_state
        del sample_times
        xs = np.array(xs)
        return E, xs


    results = chunk(get_row, np.expand_dims(ts, axis=-1), chunk_size=config.RAY_CONFIG["num_cpus"], path=checkpoint_path)
    results = list(zip(*results))

    E = np.array(results[0])
    xs = np.array(results[1])

    if not os.path.exists(energy_path):
        dirname = os.path.dirname(energy_path)
        if dirname != '':
            os.makedirs(dirname, exist_ok=True)

    if not os.path.exists(parameter_path):
        dirname = os.path.dirname(parameter_path)
        if dirname != '':
            os.makedirs(dirname, exist_ok=True)

    with open(energy_path, "wb") as file:
        pkl.dump(E, file, pkl.HIGHEST_PROTOCOL)
    with open(parameter_path, "wb") as file:
        pkl.dump(xs, file, pkl.HIGHEST_PROTOCOL)