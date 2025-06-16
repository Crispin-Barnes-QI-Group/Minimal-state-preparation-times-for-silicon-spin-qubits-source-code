import signal

import numpy as np
import tensorflow as tf

from qugradlab.pulses.invertible_functions.packaging import pack
from qugradlab.pulses.sampling import SampleTimes
from saveable_objects.extensions import SaveableWrapper
from scipy.linalg import expm
from scipy.optimize import minimize
from scipy.stats import unitary_group
from thread_chunks import chunk

from src import config
from src.setup.pulse_form import get_pulse_form, freq_scaling

def haar_sample(get_device, qudits, T_max, num_t, IQ_max, detuning, J_max, path):
    max_error = 0
    method = "BFGS"

    state_index = 0
    levels = 2
    hilbert_dim = levels**qudits
    T_min=0
    n=0
    tau=1
    length = 40
    runs=2

    def haar_random_hamiltonian_opt(length):
        E = []
        J_signals = 2*np.random.rand(length, qudits-1)-1
        IQ_signals = 2*np.random.rand(length, qudits, 2)-1
        device = get_device(qudits, J_max=J_max, J_min=0, max_drive_strength=IQ_max, detuning=detuning, use_graph=False)
        x = pack([J_signals, freq_scaling.inverse(device.zeeman_splittings), IQ_signals])
        del J_signals
        del IQ_signals
        state = device.hilbert_space.basis_vector(state_index)
        initial_state = unitary_group(hilbert_dim).rvs()@state
        final_state = unitary_group(hilbert_dim).rvs()@state
        del state
        # Infidelity
        H0 = np.identity(hilbert_dim)-np.outer(final_state.conj(), final_state)
        for T in np.flip(np.linspace(T_min, T_max, num_t)):
            samples_per_point = (1+10000//length)
            sample_times = SampleTimes(T=T, number_sample_points=samples_per_point*(length)+1)
            generate_pulse_form = get_pulse_form(length, T, sample_times.dt, samples_per_point, n=n, tau=tau)
            IQ_maxn_device = device.pulse_form(generate_pulse_form)
            del generate_pulse_form

            U = expm(-1j*(sample_times.T)*IQ_maxn_device.H0)
            del sample_times
            H = U@H0@U.T.conj()
            del U
            
            result = minimize(IQ_maxn_device.gradient,
                              x,
                              args=(initial_state, qudits, length, H),
                              method=method,
                              jac=True,
                              options={"gtol": max_error})
            x = tf.clip_by_value(result.x, -1, 1)
            E.append(result.fun)
            del result
            for _ in range(runs):
                J_signals = 2*np.random.rand(length, qudits-1)-1
                IQ_signals = 2*np.random.rand(length, qudits, 2)-1
                x_rand = pack([J_signals, freq_scaling.inverse(device.zeeman_splittings), IQ_signals])
                rand_result = minimize(IQ_maxn_device.gradient,
                                x_rand,
                                args=(initial_state, qudits, length, H),
                                method=method,
                                jac=True,
                                options={"gtol": max_error})
                if rand_result.fun < E[-1]:
                    E[-1] = rand_result.fun
                    x = tf.clip_by_value(rand_result.x, -1, 1)
                del rand_result
            del IQ_maxn_device
            del H
        del device
        del H0
        return E

    SaveableList = SaveableWrapper[list];

    E, _ = SaveableList.loadif([], path=path)

    num = 1024
    output = chunk(haar_random_hamiltonian_opt, [[length]]*num, chunk_size=config.RAY_CONFIG["num_cpus"])
    E += output
    try:
        signal.pthread_sigmask(signal.SIG_BLOCK, [signal.SIGINT])
        print("Interupt signals have been blocked while the data is saved.")
        E.save()
    except BaseException as e:
        print("There was an error while saving the data.")
        raise e
    else:
        print("Data has saved successfully.")
    finally:
        print("Interupt signals have been unblocked.")
        signal.pthread_sigmask(signal.SIG_UNBLOCK, [signal.SIGINT])