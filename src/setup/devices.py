import numpy as np
from qugradlab.systems.semiconducting.esr import SpinChain, SpinChainAngledDrive

def get_device(qudits, J_min=0.01*np.pi, J_max=2*np.pi, max_drive_strength = np.pi*0.02/(2*np.sqrt(2)), # if drive term is 0.5gX then Rabi frequency is 0.5g radians per unit time g/(4pi) periods per unit time. Divide by root(2) to account for max I and Q, divide by 4 as four signals on the line.
               detuning = 0.03,
               use_graph=True):
    return SpinChain(spins = qudits,
                     feromagnetic = False,
                     zeeman_splittings=np.array([2*np.pi*28]) if qudits == 1 else np.linspace(2*np.pi*(28+detuning), 2*np.pi*(28-detuning), qudits),
                     max_drive_strength=max_drive_strength,
                     J_min=J_min,
                     J_max=J_max,
                     use_graph=use_graph)


def get_non_linear_zeeman_device(qudits, zeeman_splittings = 0.03, J_min=0.01*np.pi, J_max=2*np.pi, max_drive_strength = np.pi*0.02/(2*np.sqrt(2)), # if drive term is 0.5gX then Rabi frequency is 0.5g radians per unit time g/(4pi) periods per unit time. Divide by root(2) to account for max I and Q, divide by 4 as four signals on the line.
               use_graph=True):
    return SpinChain(spins = qudits,
                     zeeman_splittings=zeeman_splittings,
                     max_drive_strength=max_drive_strength,
                     J_min=J_min,
                     J_max=J_max,
                     use_graph=use_graph)


def get_drive_angle_device(qudits, dynamic_Zeeman, J_min=0.01*np.pi, J_max=2*np.pi, max_drive_strength = np.pi*0.02/(2*np.sqrt(2)), # if drive term is 0.5gX then Rabi frequency is 0.5g radians per unit time g/(4pi) periods per unit time. Divide by root(2) to account for max I and Q, divide by 4 as four signals on the line.
               detuning = 0.03,
               use_graph=True):
    return SpinChainAngledDrive(spins = qudits,
                                zeeman_splittings=np.array([2*np.pi*28]) if qudits == 1 else np.linspace(2*np.pi*(28+detuning), 2*np.pi*(28-detuning), qudits),
                                dynamic_Zeeman=dynamic_Zeeman,
                                max_drive_strength=max_drive_strength,
                                J_min=J_min,
                                J_max=J_max,
                                use_graph=use_graph)
