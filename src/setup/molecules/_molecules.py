"""
This file has been modified from
https://github.com/oimeitei/ctrlq/blob/master/mole/molecules.py.
The original license is included in the LICENSE file of this directory.
The modifications fall under the repository's license which can be found in the
LICENSE file in the root directory of this repository.
"""
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import ParityMapper
from qiskit_nature.converters.second_quantization import QubitConverter
import numpy
from qiskit_nature.second_q.transformers import FreezeCoreTransformer


def h2(dist=0.75):
    mol = PySCFDriver(atom=
                      'H 0.0 0.0 0.0;'\
                      'H 0.0 0.0 {}'.format(dist), charge=0,
                      spin=0, basis='sto-3g')
    mol = mol.run()
    
    nuclear_repulsion_energy = mol.nuclear_repulsion_energy
    num_particles = mol.num_alpha + mol.num_beta + 0
    
    qubitOp = QubitConverter(ParityMapper(), two_qubit_reduction=True, z2symmetry_reduction="auto").convert(mol.hamiltonian.second_q_op(), num_particles)
    
    
    cHam = qubitOp.to_matrix() + nuclear_repulsion_energy*numpy.identity(4)

    return cHam

def hehp(dist=1.0):
    mol = PySCFDriver(atom=
                      'He 0.0 0.0 0.0;'\
                      'H 0.0 0.0 {}'.format(dist), charge=1,
                      spin=0, basis='sto-3g')
    mol = mol.run()
    
    nuclear_repulsion_energy = mol.nuclear_repulsion_energy
    num_particles = mol.num_alpha + mol.num_beta + 0

    qubitOp = QubitConverter(ParityMapper(), two_qubit_reduction=True, z2symmetry_reduction="auto").convert(mol.hamiltonian.second_q_op(), num_particles)
    
    cHam = qubitOp.to_matrix() + nuclear_repulsion_energy*numpy.identity(4)

    return cHam

def lih(dist=1.5):
    mol = PySCFDriver(atom=
                      'H 0.0 0.0 0.0;'\
                      'Li 0.0 0.0 {}'.format(dist), charge=0,
                      spin=0, basis='sto-3g')
    mol = mol.run()
    repulsion_energy = mol.nuclear_repulsion_energy

    fc_transformer = FreezeCoreTransformer(remove_orbitals=[3, 4])

    mol = fc_transformer.transform(mol)

    qubitOp = QubitConverter(ParityMapper(), two_qubit_reduction=True, z2symmetry_reduction="auto").convert(mol.hamiltonian.second_q_op(), mol.num_alpha + mol.num_beta)

    energy_shift = mol.hamiltonian.constants['FreezeCoreTransformer']
    
    shift = energy_shift + repulsion_energy
    cHam = qubitOp.to_matrix() + shift*numpy.identity(16)

    return cHam