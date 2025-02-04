import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def plot_vectors(vectors, labels, colors, title="3D Vector Visualization"):
    """
    Plots vectors in 3D space.

    Parameters:
    - vectors: List of vectors to plot, each vector as a 3-element array.
    - labels: List of labels corresponding to the vectors.
    - colors: List of colors for each vector.
    - title: Title of the plot.
    """
    if len(vectors) != len(labels) or len(vectors) != len(colors):
        raise ValueError("Vectors, labels, and colors must have the same length.")

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for vector, label, color in zip(vectors, labels, colors):
        ax.quiver(0, 0, 0, vector[0], vector[1], vector[2], color=color, label=label)

    # Set limits to ensure all vectors fit nicely
    max_val = max(np.linalg.norm(vec) for vec in vectors) + 1
    ax.set_xlim([-max_val, max_val])
    ax.set_ylim([-max_val, max_val])
    ax.set_zlim([-max_val, max_val])

    # Axis labels and plot title
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')
    plt.title(title)
    ax.legend()
    plt.show()