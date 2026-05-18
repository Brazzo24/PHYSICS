import numpy as np
from control.matlab import *
import matplotlib.pyplot as plt

a = 0.8
b = 1.5
V = 3
h = 0.5
g = 10.0

Gbike = 3.2*tf([1, 3.75], [1,0,-20])
print(Gbike)

# natural response with initial condition phi(0) = 0.1rad
# y, t = impulse(0.1*Gbike/3.2) # divide by system gain

# plt.plot(t,y)
# plt.ylabel("Lean Angle (rad)")
# plt.xlabel("time (s)")
# plt.show()

# feedback control delta(t) = -K_p*phi(t); with K_p = 4
Gcl = feedback(4*Gbike,1)
print(Gcl)

y, t = impulse(0.1*Gcl/12.8) #divide by closed loop system gain

plt.plot(t,y)
plt.show()

