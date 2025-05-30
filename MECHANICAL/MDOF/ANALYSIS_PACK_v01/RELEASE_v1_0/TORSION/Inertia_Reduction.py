import numpy as np

I1 = 2.0

omega_is = 100 #rad/s
omega_target = 10 #rad/s

ratio = omega_target/omega_is

if omega_target > omega_is:
    I1_target = I1/(ratio**2)
elif omega_target < omega_is:
    I1_target = I1 / (ratio**2)
else: I1_target = I1

# backcheck calculating kinetic energy
T = 0.5 * I1 * omega_is**2
T_target = 0.5 * I1_target * omega_target**2

print("Original kinetic energy is:", T)
print("Target kinetic Energy is: ", T_target)

if np.round(T) == np.round(T_target):
    print("Inertia Reduction ha been conducted correctly! :)")
else: 
    print("Oops! Something went wrong! :( \n Consider checking the ratio!")


print("I1 become I1_Target: ", I1_target)

