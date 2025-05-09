import os
import numpy as np
import json
from zipfile import ZipFile

# Create the folder structure
base_dir = "/mnt/data/TorsionalAnalysis"
folders = [
    "TorsionalAnalysis/simulations",
    "TorsionalAnalysis/results/Baseline",
    "TorsionalAnalysis/results/DMF",
    "TorsionalAnalysis/postprocessing"
]

for folder in folders:
    os.makedirs(os.path.join("/mnt/data", folder), exist_ok=True)

# 1. Save helper module
helper_code = '''\
import numpy as np
from scipy.linalg import eigh

def build_full_matrices(m, c_inter, k_inter):
    N = len(m)
    M = np.diag(m)
    C = np.zeros((N, N))
    K = np.zeros((N, N))
    for i in range(N - 1):
        C[i, i] += c_inter[i]
        C[i, i+1] -= c_inter[i]
        C[i+1, i] -= c_inter[i]
        C[i+1, i+1] += c_inter[i]

        K[i, i] += k_inter[i]
        K[i, i+1] -= k_inter[i]
        K[i+1, i] -= k_inter[i]
        K[i+1, i+1] += k_inter[i]
    return M, C, K

def run_modal_analysis(M, K, m_array, k_array, mode_index=3):
    eigvals, eigvecs = eigh(K, M)
    freqs_hz = np.sqrt(eigvals) / (2 * np.pi)

    phi = eigvecs[:, mode_index]
    omega = 2 * np.pi * freqs_hz[mode_index]
    phi = phi / np.max(np.abs(phi))

    KE_dof = 0.5 * m_array * (phi * omega)**2
    PE_spring = 0.5 * k_array * np.diff(phi)**2

    return freqs_hz[mode_index], KE_dof, PE_spring
'''
with open(f"{base_dir}/postprocessing/helper_plotting.py", "w") as f:
    f.write(helper_code)

# 2. Save post-processing script
compare_script = '''\
import numpy as np
import matplotlib.pyplot as plt
import os

designs = ['Baseline', 'DMF']
mode_index = 3
colors = ['blue', 'green']

fig, axs = plt.subplots(1, 2, figsize=(12, 4))

for i, design in enumerate(designs):
    path = f'../results/{design}'
    freqs = np.load(f'{path}/modal_freqs.npy')
    KE = np.load(f'{path}/KE_dof.npy')
    PE = np.load(f'{path}/PE_spring.npy')

    axs[0].bar(np.arange(len(KE[mode_index])), KE[mode_index], alpha=0.5, label=f'{design} ({freqs[mode_index]:.2f} Hz)', color=colors[i])
    axs[1].bar(np.arange(len(PE[mode_index])), PE[mode_index], alpha=0.5, label=f'{design}', color=colors[i])

axs[0].set_title('Kinetic Energy per DOF')
axs[0].set_xlabel('DOF Index')
axs[0].set_ylabel('Joules')
axs[0].legend()
axs[0].grid(True)

axs[1].set_title('Potential Energy per Spring')
axs[1].set_xlabel('Spring Index')
axs[1].set_ylabel('Joules')
axs[1].legend()
axs[1].grid(True)

plt.tight_layout()
plt.show()
'''
with open(f"{base_dir}/postprocessing/compare_modes.py", "w") as f:
    f.write(compare_script)

# Zip the folder for download
zip_path = "/mnt/data/TorsionalAnalysis.zip"
with ZipFile(zip_path, "w") as zipf:
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            full_path = os.path.join(root, file)
            rel_path = os.path.relpath(full_path, "/mnt/data")
            zipf.write(full_path, arcname=rel_path)

zip_path
