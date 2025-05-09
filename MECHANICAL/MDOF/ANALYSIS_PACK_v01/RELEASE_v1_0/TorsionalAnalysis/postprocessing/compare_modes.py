import numpy as np
import matplotlib.pyplot as plt
import json
import os

designs = ['Baseline', 'DMF']  # can add more
mode_index = 3  # mode to compare

colors = ['blue', 'green', 'orange', 'purple']
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
