import pandas as pd

def load_input(file_path="input_data.csv"):
    """
    Reads input data from a CSV file and returns inertia and stiffness lists.

    Parameters
    ----------
    file_path : str
        The path to the input CSV file.

    Returns
    -------
    tuple of (list, list)
        - inertias: List of inertia values (kg·m^2)
        - stiffnesses: List of tuples (nodeA, nodeB, stiffness in Nm/rad)
    """
    df = pd.read_csv(file_path)

    # Extract inertias (First column)
    inertias = df.iloc[:, 0].tolist()

    # Extract stiffness values (Second column)
    stiffness_values = df.iloc[:, 1].tolist()

    # Automatically connect inertias with springs (Assuming sequential connection)
    stiffnesses = []
    for i in range(len(stiffness_values)):
        if i == 0:
            stiffnesses.append((0, i+1, stiffness_values[i]))  # First inertia connected to ground
        else:
            stiffnesses.append((i, i+1, stiffness_values[i]))  # Connect inertia pairs

    return inertias, stiffnesses