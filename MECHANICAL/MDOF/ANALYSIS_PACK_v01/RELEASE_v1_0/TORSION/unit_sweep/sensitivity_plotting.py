import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def plot_sensitivity_line(df, param_name):
    """Plot 1D sensitivity as line plots for each DOF column."""
    dof_columns = [col for col in df.columns if col.startswith("Energy_DOF_")]
    
    plt.figure(figsize=(10, 6))
    for col in dof_columns:
        plt.plot(df[param_name], df[col], marker='o', label=col)

    plt.xlabel(param_name)
    plt.ylabel("Kinetic Energy [J]")
    plt.title(f"Sensitivity of Modal Energy vs {param_name}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_sensitivity_heatmap(df, param1_prefix="m", param2_prefix="k"):
    """
    Plot 2D heatmaps for each DOF. Automatically detects param columns from prefixes.
    """
    dof_columns = [col for col in df.columns if col.startswith("Energy_DOF_")]
    
    # Auto-detect column names
    param1 = next((col for col in df.columns if col.startswith(param1_prefix)), None)
    param2 = next((col for col in df.columns if col.startswith(param2_prefix)), None)

    if param1 is None or param2 is None:
        raise ValueError(f"Could not find columns starting with '{param1_prefix}' and '{param2_prefix}'.")

    for col in dof_columns:
        pivot = df.pivot(index=param1, columns=param2, values=col)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(pivot, annot=True, fmt=".2e", cmap="viridis")
        plt.title(f"{col} sensitivity heatmap")
        plt.xlabel(param2)
        plt.ylabel(param1)
        plt.tight_layout()
        plt.show()
