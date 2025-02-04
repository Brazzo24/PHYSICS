import numpy as np
import pandas as pd

print("load_file.py loaded")

def get_manual_input():
    """Defines a small number of inertias and stiffnesses for debugging."""
    inertias = np.array([0.15, 0.25, 0.35])  # Example values
    stiffness_values = np.array([5000, 7000])  # Example values

    # Ensure correct format: List of tuples (nodeA, nodeB, stiffness)
    stiffnesses = [(i, i+1, stiffness_values[i]) for i in range(len(stiffness_values))]

    return inertias, stiffnesses

def load_input(file_path="input_data.csv"):
    """
    Reads input data from a CSV file and returns inertia and stiffness lists.
    """
    df = pd.read_csv(file_path)

    # Extract inertias
    inertias = df.iloc[:, 0].tolist()

    # Extract stiffness values
    stiffness_values = df.iloc[:, 1].tolist()

    # Create list of tuples (nodeA, nodeB, stiffness)
    stiffnesses = [(i, i+1, stiffness_values[i]) for i in range(len(stiffness_values))]

    return inertias, stiffnesses