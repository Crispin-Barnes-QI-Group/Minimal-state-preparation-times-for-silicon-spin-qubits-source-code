import sys

from pathlib import Path

sys.path.append(str(Path(__file__).absolute().parent.parent))

import numpy as np

from tqdm import tqdm

class Ham():
    def __init__(self, matrix):
        matrix *= 1
        self.eigenvalues, self.eigenvectors = np.linalg.eigh(matrix)
        self.min_eigenvector = self.eigenvectors[:, 0]
        self.min_eigenvalue = self.eigenvalues[0]
        # This next step is to remove any numerical errors in the diagonalisation.
        self.H = self.eigenvectors@np.diag(self.eigenvalues)@self.eigenvectors.conj().T

def get_FCI_energies(mol, rs):
    E0 = np.array([Ham(mol(r)).min_eigenvalue for r in tqdm(rs)])
    return E0