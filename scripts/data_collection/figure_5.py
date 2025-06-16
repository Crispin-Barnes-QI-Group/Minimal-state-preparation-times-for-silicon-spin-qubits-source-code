import os

import numpy as np
import tensorflow as tf

from qugradlab.pulses.invertible_functions.packaging import pack
from qugradlab.pulses.sampling import SampleTimes
from scipy.linalg import expm
from scipy.optimize import minimize
from thread_chunks import chunk

from src import config
from src.setup.devices import get_device
from src.setup.get_dir import DATA_DIR
from src.setup.pulse_form import get_pulse_form, freq_scaling

data_dir = os.path.join(DATA_DIR, "figure_5")

max_error = 0
method = "BFGS"

qudits = 1
state_index = 0
levels = 2
hilbert_dim = levels**qudits
T_min=0
T_max=290
num_t = 290
n=0
tau=1
length = 40
runs=2

thetas = np.linspace(0, np.pi, 30)
phis = np.linspace(0, np.pi/4, 30)

def bloch_sphere_scan(length, theta, phi):
    E = []
    xs = []
    J_signals = np.zeros((length, qudits-1))
    strength = theta*(200*np.sqrt(2))/(T_max*np.pi)
    snorm = max(np.sin(phi), np.cos(phi))*theta*(200*np.sqrt(2))/(T_max*np.pi)
    if snorm > 1:
        strength /= snorm
    drive_signals = strength*np.stack([np.sin(phi)*np.ones((length, qudits)), np.cos(phi)*np.ones((length, qudits))], axis=-1)
    device = get_device(qudits, J_min=0, use_graph=False)
    x = pack([J_signals, freq_scaling.inverse(device.zeeman_splittings), drive_signals])
    del J_signals
    del drive_signals
    initial_state = device.hilbert_space.basis_vector(0)
    final_state = np.cos(theta/2)*device.hilbert_space.basis_vector(0)+np.sin(theta/2)*np.exp(1j*phi)*device.hilbert_space.basis_vector(1)
    H0 = np.identity(hilbert_dim)-np.outer(final_state.conj(), final_state) # infidelity
    for T in np.flip(np.linspace(T_min, T_max, num_t)):
        samples_per_point = (1+10000//length)
        sample_times = SampleTimes(T=T, number_sample_points=samples_per_point*(length))
        generate_pulse_form = get_pulse_form(length, T, sample_times.dt, samples_per_point, n=n, tau=tau)
        driven_device = device.pulse_form(generate_pulse_form)
        del generate_pulse_form

        U = expm(-1j*(sample_times.T)*driven_device.H0)
        del sample_times
        H = U@H0@U.T.conj()
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

        J_signals = np.zeros((length, qudits-1))
        strength = theta*(200*np.sqrt(2))/(T*np.pi)
        snorm = max(np.sin(phi), np.cos(phi))*theta*(200*np.sqrt(2))/(T*np.pi)
        if snorm > 1:
            strength /= snorm
        drive_signals = strength*np.stack([np.sin(phi)*np.ones((length, qudits)), np.cos(phi)*np.ones((length, qudits))], axis=-1)
        x_rand = pack([J_signals, freq_scaling.inverse(device.zeeman_splittings), drive_signals])
        rand_result = minimize(driven_device.gradient,
                               x_rand,
                               args=(initial_state, qudits, length, H),
                               method=method,
                               jac=True,
                               options={"gtol": max_error})
        if rand_result.fun < E[-1]:
            E[-1] = rand_result.fun
            x = tf.clip_by_value(rand_result.x, -1, 1)
            xs[-1] = x
        del rand_result
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
            if rand_result.fun < E[-1]:
                E[-1] = rand_result.fun
                x = tf.clip_by_value(rand_result.x, -1, 1)
                xs[-1] = x
            del rand_result
        del driven_device
        del H
    del device
    del H0
    return E, xs

output = chunk(bloch_sphere_scan,
               [[length, theta, phi] for theta in thetas for phi in phis],
               chunk_size=config.RAY_CONFIG["num_cpus"],
               path=os.path.join(data_dir, "Bloch_sphere_scan_checkpoint.pkl"))