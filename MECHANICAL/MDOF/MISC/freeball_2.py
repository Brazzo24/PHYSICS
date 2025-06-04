import numpy as np
from math import sin
from scipy.linalg import eigh
from numpy.linalg import inv
from matplotlib import pyplot as plt

# set up the parameters
F0 = 5.0
omega = 10.0

k1 = 1000.0
k2 = 2000.0 
k3 = 3000.0
k4 = 4000.0

m1 = 10.0
m2 = 2.0
m3 = 3.0
m4 = 4.0

c1 = 10.1
c2 = 20.2
c3 = 30.3
c4 = 40.4

dof = 4

time_step = 1.0e-4
end_time = 10.0

#set up the matrices
"""
           GROUND
            |k1
[         mass1         ]
   |k2           |k3
[mass2]       [mass3]
   |            |k4
Force(t)      [mass4]
"""

K = np.array([[k1 + k2 + k3, -k2, -k2, 0], [-k2, k2, 0, 0], [-k3, 0, k3 + k3, -k4], [0, 0, -k4, k4]])
C = np.array([[c1 + c2 + c3, -c2, -c2, 0], [-c2, c2, 0, 0], [-c3, 0, c3 + c3, -c4], [0, 0, -c4, c4]])
M = np.array([[m1, 0, 0, 0], [0, m2, 0, 0], [0, 0, m3, 0], [0, 0, 0, m4]])
I = np.identity(dof)

A = np.zeros((2*dof, 2*dof))
B = np.zeros((2*dof, 2*dof))
Y = np.zeros((2*dof, 1))
F = np.zeros((2*dof, 1))

A[0:dof, 0:dof] = M
A[dof:2*dof, dof:2*dof] = I

B[0:dof, dof:2*dof] = K
B[0:dof, 0:dof] = C
B[dof:2*dof, 0:dof] = -I

print("Matrix A: \n", A)
print("Matrix B: \n", B)

# find natural frequencies and mode shapes
evals, evecs = eigh(K,M)
frequencies = np.sqrt(evals)
print(frequencies)
print()
print(evecs)

A_inv = inv(A)
force = []
X1 = []
X2 = []
X3 = []
X4 = []

# numerically integrate the Equations of Motion
for t in np.arange(0, end_time, time_step):
    F[1] = F0 * sin(omega*t)
    Y_new = Y + time_step * A_inv.dot(F - B.dot(Y))
    Y = Y_new
    force.extend(F[1])
    X1.extend(Y[0])
    X2.extend(Y[1])
    X3.extend(Y[2])
    X4.extend(Y[3])

# plot results
time = [round(t, 5) for t in np.arange(0, end_time, time_step)]

plt.plot(time, X1)
plt.plot(time, X2)
plt.plot(time, X3)
plt.plot(time, X4)
plt.ylabel("Displacement")
plt.xlabel("time (s)")
plt.legend(["X1", "X2", "X3", "X4"], loc='lower right')
plt.grid(True)
plt.show()

