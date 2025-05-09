import numpy as np
from scipy.linalg import eigh
import os, json

from your_module import build_full_matrices, run_modal_analysis  # you move your functions here

# === System Definition ===
m_base = np.array([...])  # your original inertias
c_base = np.array([...])
k_base = np.array([...])

# === DMF insertion ===
m_dmf = 5e-3
k_dmf = 1.0e4
c_dmf = 0.2

m = np.insert(m_base, 1, m_dmf)
c = np.insert(c_base, 0, c_dmf)
k = np.insert(k_base, 0, k_dmf)

# === Modal Analysis ===
M, _, K = build_full_matrices(m, c, k)

eigvals, eigvecs = eigh(K, M)
freqs_hz = np.sqrt(eigvals) / (2 * np.pi)

KE_all = []
PE_all = []

for i in range(len(freqs_hz)):
    _, KE, PE = run_modal_analysis(M, K, m, k, mode_index=i)
    KE_all.append(KE)
    PE_all.append(PE)

# === Save Results ===
result_dir = 'results/DMF'
os.makedirs(result_dir, exist_ok=True)

np.save(f'{result_dir}/modal_freqs.npy', freqs_hz)
np.save(f'{result_dir}/KE_dof.npy', np.array(KE_all))
np.save(f'{result_dir}/PE_spring.npy', np.array(PE_all))

# Save metadata
meta = {'m_dmf': m_dmf, 'k_dmf': k_dmf, 'c_dmf': c_dmf}
with open(f'{result_dir}/meta.json', 'w') as f:
    json.dump(meta, f, indent=2)
