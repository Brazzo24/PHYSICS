import numpy as np

"""
Make a function out of this.
Differentiate Functions for: Speeds are given, thus ratios need to be calculated;
Or: Ratios are given, thus speeds need to be calculated.

"""
omega_ref = 100 #rad/s
I_ref = 4.0 # kgm^2
I = np.array([2.0, 1.0, 0.85, 3.5]) # kgm^2


t_gear = np.array([30, 30, 42, 40])
ratio = np.zeros(len(I))

for i in range(len(I)):
    ratio[i] = t_gear[0]/t_gear[i]





print(ratio)
print(I)

T = 0.5 * I * omega**2

for i in range(len(I)):
    # backcheck calculating kinetic energy
    T = 0.5 * I[i] * omega[i]**2
        
    if omega_ref > omega[i]:
        I[i] = I[i] / (ratio[i]**2)
    elif omega_ref < omega:
        I[i] = I[i] / (ratio[i]**2)
    else: I[i] = I[i]
    
print(I)

T_target = 0.5 * I * omega_ref**2

print("Original kinetic energy is:", T)
print("Target kinetic Energy is: ", T_target)

print("I-values reduced to reference speed become: ", I)

