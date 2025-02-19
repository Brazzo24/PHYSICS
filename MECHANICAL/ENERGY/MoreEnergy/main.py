import numpy as np
import matplotlib.pyplot as plt

"""
SDOF Damped and Driven

"""

k = 10001       # Nm
c = 0.1    # 
m = 1       # kg

omega = 101  # rad/s

F_0 = 10 # N

om_n = np.sqrt(k/m)

x_0 = (F_0)/(np.sqrt((c*omega)**2 + (k - m*omega**2)**2))

phi = np.arctan((c*omega)/(k-m*omega**2))
phi = np.rad2deg(phi)

print("Amplitude is in:", x_0)
print("Phi is in deg:", phi)
print("Natrual Frequency is in rad/s:", om_n)

omegas = np.linspace(0.01, 1000)
x_0s = []
phis = []

for om in omegas:
    x_0 = (F_0)/(np.sqrt((c*om)**2 + (k - m*om**2)**2))

    phi = np.rad2deg(np.arctan((c*om)/(k-m*om**2)))
  

    x_0s.append(x_0)
    phis.append(phi)

fig, axs = plt.subplots(2)
fig.suptitle('suptitle')
axs[0].plot(omegas, x_0s)
# plt.loglog(True)
plt.xlabel('omegas')
plt.ylabel('x_0s')
plt.grid()

axs[1].plot(omegas, phis)
# plt.loglog(True)
plt.xlabel('omegas')
plt.ylabel('phis')
plt.grid(True)

plt.show()
