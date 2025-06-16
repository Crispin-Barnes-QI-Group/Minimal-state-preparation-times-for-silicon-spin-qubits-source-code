import os
import subprocess

from src.setup.get_dir import ROOT_DIR

script_dir = os.path.join(ROOT_DIR, "scripts", "download_data_from_zenodo.sh")

subprocess.call(script_dir, shell=True)

from plotting import plot_all_data