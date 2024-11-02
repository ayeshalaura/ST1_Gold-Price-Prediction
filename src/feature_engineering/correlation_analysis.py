# -----------------------------------------------------------------------------
# Submitted by:     u3258928
# Date:             2024 November 02
# Course:           Software Technology 1
# Activity:         Assessment 3: Programming project
# -----------------------------------------------------------------------------


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os


def conduct_correlation_analysis(df: pd.DataFrame, output_dir: str = "reports/visualizations") -> pd.DataFrame:
    """
    Conducts correlation analysis on the DataFrame and visualizes the correlation matrix.

    Args:
        df (pd.DataFrame): The preprocessed dataset.
        output_dir (str): Directory to save the correlation heatmap.

    Returns:
        pd.DataFrame: The correlation matrix.
    """
    correlation_matrix = df.corr()

    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(20, 16))
    sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', linewidths=0.5)
    plt.title('Correlation Matrix Heatmap')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'correlation_matrix_heatmap.png'))
    plt.close()

    return correlation_matrix


def get_highly_correlated_features(correlation_matrix: pd.DataFrame, threshold: float = 0.8) -> list:
    """
    Identifies pairs of features with correlation above the specified threshold.

    Args:
        correlation_matrix (pd.DataFrame): The correlation matrix.
        threshold (float): The correlation threshold.

    Returns:
        list: List of tuples containing pairs of highly correlated features.
    """
    correlated_pairs = []
    cols = correlation_matrix.columns
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            if abs(correlation_matrix.iloc[i, j]) > threshold:
                correlated_pairs.append((cols[i], cols[j], correlation_matrix.iloc[i, j]))
    return correlated_pairs
