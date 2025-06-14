import numpy as np
from math import sin
from scipy.linalg import eigh
from numpy.linalg import inv
from matplotlib import pyplot as plt

# set up the parameters
F0 = 5.0
omega = 10.0
k = 10000.0
m = 10.0
dof = 3

time_step = 1.0e-4
end_time = 10.0

#set up the matrices
K = np.array([[3*k, -k, -k], [-k, k, 0], [-k, 0, k]])
M = np.array([[2*m, 0, 0], [0, m, 0], [0, 0, m]])
I = np.identity(dof)

A = np.zeros((2*dof, 2*dof))
B = np.zeros((2*dof, 2*dof))
Y = np.zeros((2*dof, 1))
F = np.zeros((2*dof, 1))

A[0:3, 0:3] = M
A[3:6, 3:6] = I

B[0:3, 3:6] = K
B[3:6, 0:3] = -I

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

# numerically integrate the Equations of Motion
for t in np.arange(0, end_time, time_step):
    F[1] = F0 * sin(omega*t)
    Y_new = Y + time_step * A_inv.dot(F - B.dot(Y))
    Y = Y_new
    force.extend(F[1])
    X1.extend(Y[0])
    X2.extend(Y[1])
    X3.extend(Y[2])

# plot results
time = [round(t, 5) for t in np.arange(0, end_time, time_step)]

plt.plot(time, X1)
plt.plot(time, X2)
plt.plot(time, X3)
plt.ylabel("Displacement")
plt.xlabel("time (s)")
plt.show()

