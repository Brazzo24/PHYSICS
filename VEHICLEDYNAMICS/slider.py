
import numpy as np
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

x = np.linspace(0, 10, 20)
y = np.sin(0.5 * x) * np.sin(x * np.random.randn(20))

spline = UnivariateSpline(x, y, s = 5)
x_spline = np.linspace(0, 10, 100)
y_spline = spline(x_spline)

fig = plt.figure()
plt.subplots_adjust(bottom=.25)
ax = fig.subplots()
p = ax.plot(x, y)
p, = ax.plot(x_spline, y_spline, 'y')
ax_slide = plt.axes([.25, .1, .65, .03])
s_factor = Slider(ax_slide, "Changing factor", 1, 5, valinit = 2.5, valstep = 0.05)

def update(val):
    current_v = s_factor.val
    spline = UnivariateSpline(x, y, s=current_v)
    p.set_ydata(spline(x_spline))
    fig.canvas.draw()

s_factor.on_changed(update)
plt.show()