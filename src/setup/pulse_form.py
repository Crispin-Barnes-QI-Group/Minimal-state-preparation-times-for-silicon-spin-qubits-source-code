import numpy as np
import tensorflow as tf

from qugradlab.pulses.filtering import get_fixed_filter, \
                                       const_spline_time_spline_matrix
from qugradlab.pulses.invertible_functions import InvertibleFunction
from qugradlab.pulses.invertible_functions.packaging import unpack, \
                                                            package_complex
from qugradlab.pulses.invertible_functions.scaling import linear_rescaling


frequency_bounds = (2*np.pi*27, 2*np.pi*29)
freq_scaling = linear_rescaling.specify_parameters(min=frequency_bounds[0], max=frequency_bounds[1])

def get_pulse_form(length, T, dt, samples_per_point, tau=2*np.pi, n=0):
    @InvertibleFunction
    def scaling(x):
        J_signals, drive_frequencies, drive_signals = x
        drive_frequencies = freq_scaling(drive_frequencies)
        return drive_signals, drive_frequencies, J_signals
    @scaling.set_inverse
    def scaling_inverse(drive_signals, drive_frequencies, J_signals):
        drive_frequencies = freq_scaling.inverse(drive_frequencies)
        return drive_signals, drive_frequencies, J_signals
    unpack_and_scale = scaling.compose(unpack)
    def get_shapes(qudits, T):
        return [(T, qudits-1), (qudits,), (T, qudits, 2)]
    filter_func = get_fixed_filter(length, T/length, samples_per_point, tau, 1, n, np.array([0]), 0)
    def pulse_form(x, initial_state, qudits, length):
            x = tf.clip_by_value(x, -1, 1)
            drive_signals, drive_frequencies, J_signals = unpack_and_scale(x, get_shapes(qudits, length))
            drive_signals = tf.cast(drive_signals, dtype=tf.complex128)
            drive_signals = filter_func(const_spline_time_spline_matrix(drive_signals))
            drive_signals = package_complex(drive_signals)
            J_signals = tf.cast(J_signals, dtype=tf.complex128)
            J_signals = filter_func(const_spline_time_spline_matrix(J_signals+1))-1
            J_signals = tf.cast(tf.clip_by_value(tf.cast(J_signals, dtype=tf.float64), -1, 1), dtype=tf.complex128)
            return drive_signals, drive_frequencies, J_signals, initial_state, dt
    return pulse_form

def get_unpack(length, qudits):
    def get_shapes(qudits, T):
        return [(T, qudits-1), (qudits,), (T, qudits, 2)]
    return lambda x: unpack(x, get_shapes(qudits, length))
def get_unpack_and_scale(length, qudits):
    @InvertibleFunction
    def scaling(x):
        J_signals, drive_frequencies, drive_signals = x
        drive_frequencies = freq_scaling(drive_frequencies)
        return drive_signals, drive_frequencies, J_signals
    @scaling.set_inverse
    def scaling_inverse(drive_signals, drive_frequencies, J_signals):
        drive_frequencies = freq_scaling.inverse(drive_frequencies)
        return drive_signals, drive_frequencies, J_signals
    def get_shapes(qudits, T):
        return [(T, qudits-1), (qudits,), (T, qudits, 2)]
    unpack_and_scale = scaling.compose(unpack)
    return lambda x: unpack_and_scale(x, get_shapes(qudits, length))