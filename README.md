# Minimal state-preparation times for silicon spin qubits source code

This repository contains the source code required to reproduce and plot the data presented in:

Long, C. K., Mayhall, N. J., Economou, S. E., Barnes, E., Barnes, C. H. W., Martins, F., … Mertig, N. (2024). Minimal evolution times for fast, pulse-based state preparation in silicon spin qubits. arXiv [Quant-Ph]. Retrieved from http://arxiv.org/abs/2406.10913

A C++ library and several Python packages were developed for this project and have been packaged. The decision to distribute these libraries and packages separately was taken to increase the reusability of the code and improve the code quality and robustness by decoupling the modules. These packages are briefly described [below](#supporting-libraries-and-packages).

## Installation

The required Python packages can be installed by executing
```bash
pip install -r requirements.txt
```
in the root directory of this repository. To ensure reproducibility, all packages in ``requirements.txt`` are version pinned, and Python ``3.10.16`` should be used.

## Downloading the data

The data for the article can be found at:

Long, C. K., Barnes, C. H. W., Arvidsson-Shukur, D. R. M., & Mertig, N. (2025). Minimal state-preparation times for silicon spin qubits data [Data set]. Zenodo. https://doi.org/10.5281/zenodo.15676408

The data can be downloaded to the correct directories for plotting by running
```bash
bash scripts/download_data_from_zenodo.sh
```

## Reproducing the data

> **Warning:** Some of the intermediate data files (not uploaded to Zenodo) produced have sizes on the order of 50 GB.

> **Warning:** We expect the data collection scripts to take several weeks to months to run on CPUs with approximately 64 cores.

> **Warning:** The data collection scripts likely need more than 50 GB of RAM (or SWAP) to store the intermediate data. The exact memory requirements have not been benchmarked. That said, the data was collected on a device with 2 TB of RAM.

Alternatively, all of the data can be collected in series by executing
```bash
python scripts/data_collection/collect_all_data.py
```
in the root directory of this repository. Alternatively, if you only wish to collect the data for Figure ``X`` in our article, then you can execute
```bash
# Replace X with the figure number
python scripts/data_collection/figure_X.py 
```
in the root directory of this repository.

### Plotting the data

All the figures can be plotted from the collected data by executing
```bash
python scripts/plotting/plot_all_data.py
```
in the root directory of this repository. Alternatively, if you only wish to reproduce Figure ``X`` in our article then you can execute
```bash
# Replace X with the figure number
python scripts/plotting/figure_X.py 
```
in the root directory of this repository.

## Supporting Libraries and Packages

For this project, the following libraries and packages were developed:

### [Suzuki-Trotter-Evolver](https://github.com/Christopher-K-Long/Suzuki-Trotter-Evolver)

A high-performance C++ header-only library for evolving states under the Schrödinger equation using first-order Suzuki-Trotter and computing switching functions. Performance benchmarks with a Python wrapper can be found here: https://PySTE.readthedocs.io/en/latest/benchmarks/index.html.

**Source code repository:** [https://github.com/Christopher-K-Long/Suzuki-Trotter-Evolver](https://github.com/Christopher-K-Long/Suzuki-Trotter-Evolver)

**Documentation:** [https://Suzuki-Trotter-Evolver.readthedocs.io](https://Suzuki-Trotter-Evolver.readthedocs.io)

### [PySTE](https://github.com/Christopher-K-Long/PySTE)

A Python wrapper around [Suzuki-Trotter-Evolver](https://github.com/Christopher-K-Long/Suzuki-Trotter-Evolver) allowing for fast simulation of time-dependent Hamiltonians in Python. Our performance benchmarks (https://PySTE.readthedocs.io/en/latest/benchmarks/index.html) demonstrate that [PySTE](https://github.com/Christopher-K-Long/PySTE) can be up to 100 times faster than [QuTiP](https://qutip.org) for the types of quantum systems we focused on in our article. It is this acceleration that allowed us to scan the extensive ranges of device parameters presented in our article.

**Source code repository:** [https://github.com/Christopher-K-Long/PySTE](https://github.com/Christopher-K-Long/PySTE)

**Documentation:** [https://PySTE.readthedocs.io](https://PySTE.readthedocs.io)

### [QuGrad](https://github.com/Christopher-K-Long/QuGrad)

A Python package for quantum optimal control using [PySTE](https://github.com/Christopher-K-Long/PySTE) as a backend.

**Source code repository:** [https://github.com/Christopher-K-Long/QuGrad](https://github.com/Christopher-K-Long/QuGrad)

**Documentation:** [https://QuGrad.readthedocs.io](https://QuGrad.readthedocs.io)

### [QuGradLab](https://github.com/Christopher-K-Long/QuGradLab)

An extension to the Python package [QuGrad](https://QuGrad.readthedocs.io) that implements common Hilbert space structures, Hamiltonians, and pulse shapes for quantum control.

**Source code repository:** [https://github.com/Christopher-K-Long/QuGradLab](https://github.com/Christopher-K-Long/QuGradLab)

**Documentation:** [https://QuGradLab.readthedocs.io](https://QuGradLab.readthedocs.io)

## Updates

This repository will only be updated to fix bugs that prevent the reproducibility of the data presented in the article. To ensure reproducibility of the article any bugs that are found that have impacted the data in the article will not be fixed.
