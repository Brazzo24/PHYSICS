import numpy as np
from matplotlib import pyplot as plt

def generate_figure(figsize=(2, 2), xlim=[0, 1], ylim=[0, 1]):
    """Generate figure for plotting complex numbers

    Notebook: C2/C2_ComplexNumbers.ipynb

    Args:
        figsize: Figure size (Default value = (2, 2))
        xlim: Limits of x-axis (Default value = [0, 1])
        ylim: Limits of y-axis (Default value = [0, 1])
    """
    plt.figure(figsize=figsize)
    plt.grid()
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.xlabel(r'$\mathrm{Re}$')
    plt.ylabel(r'$\mathrm{Im}$')

def plot_vector(c, color='k', start=0, linestyle='-'):
    """Plot arrow corresponding to difference of two complex numbers

    Notebook: C2/C2_ComplexNumbers.ipynb

    Args:
        c: Complex number
        color: Color of arrow (Default value = 'k')
        start: Complex number encoding the start position (Default value = 0)
        linestyle: Linestyle of arrow (Default value = '-')

    Returns:
        arrow (matplotlib.patches.FancyArrow): Arrow
    """
    return plt.arrow(np.real(start), np.imag(start), np.real(c), np.imag(c),
                     linestyle=linestyle, head_width=0.05, fc=color, ec=color, overhang=0.3,
                     length_includes_head=True)

c = 1.5 + 0.8j

generate_figure(figsize=(7.5, 3), xlim=[0, 2.5], ylim=[0, 1])
v = plot_vector(c, color='k')

plt.text(1.5, 0.8, '$c$', size='16')
plt.text(0.8, 0.55, '$|c|$', size='16')
plt.text(0.25, 0.05, '$\gamma$', size='16')
plt.show()