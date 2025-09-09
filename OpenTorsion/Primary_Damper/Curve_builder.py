import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact, FloatSlider

def torque_curve(x, a, b, c, d, f):
    return (a - b) * np.exp(-((c * (x - d)) / f) ** 2) + b

def plot_torque_curve(a=700, b=100, c=1.2, d=7000, f=4000):
    x = np.linspace(0, 15000, 1000)
    y = torque_curve(x, a, b, c, d, f)

    plt.figure(figsize=(10, 5))
    plt.plot(x, y, color='purple')
    plt.xlabel('RPM')
    plt.ylabel('Torque')
    plt.title('Interactive Torque Curve')
    plt.grid(True)
    plt.ylim(0, 1100)
    plt.show()

interact(plot_torque_curve,
         a=FloatSlider(min=100, max=1000, step=10, value=700),
         b=FloatSlider(min=0, max=500, step=10, value=100),
         c=FloatSlider(min=0.1, max=3, step=0.1, value=1.2),
         d=FloatSlider(min=0, max=15000, step=100, value=7000),
         f=FloatSlider(min=100, max=8000, step=100, value=4000));
