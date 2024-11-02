# -----------------------------------------------------------------------------
# Submitted by:     u3258928
# Date:             2024 November 02
# Course:           Software Technology 1
# Activity:         Assessment 3: Programming project
# -----------------------------------------------------------------------------


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from typing import Dict


def plot_model_comparison(metrics: dict, output_dir: str = "reports/visualizations") -> None:
    """
    Plots a comparison of different models based on evaluation metrics.

    Args:
        metrics (dict): Dictionary containing evaluation metrics for each model.
        output_dir (str): Directory to save the comparison plot.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert the metrics dictionary to a DataFrame
    metrics_df = pd.DataFrame.from_dict(metrics, orient='index')
    metrics_df.reset_index(inplace=True)
    metrics_df.rename(columns={'index': 'Model'}, inplace=True)
    
    # Melt the DataFrame for seaborn compatibility
    metrics_melted = metrics_df.melt(id_vars='Model', var_name='Metric', value_name='Value')
    
    plt.figure(figsize=(14, 8))
    sns.barplot(data=metrics_melted, x='Metric', y='Value', hue='Model')
    plt.title('Model Comparison based on Evaluation Metrics')
    plt.xlabel('Evaluation Metric')
    plt.ylabel('Score')
    plt.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'model_comparison.png'))
    plt.close()

def plot_actual_vs_predicted(y_true: pd.Series, y_preds: Dict[str, pd.Series], output_dir: str = "reports/visualizations") -> None:
    """
    Plots actual vs predicted values for each model.

    Args:
        y_true (pd.Series): The true target values.
        y_preds (Dict[str, pd.Series]): Dictionary of predicted values from each model.
        output_dir (str): Directory to save the plots.
    """
    os.makedirs(output_dir, exist_ok=True)
    for model_name, y_pred in y_preds.items():
        plt.figure(figsize=(14, 7))
        plt.plot(y_true.index, y_true.values, label='Actual', color='blue')
        plt.plot(y_true.index, y_pred, label='Predicted', color='red')
        plt.title(f'Actual vs Predicted Values - {model_name}')
        plt.xlabel('Date')
        plt.ylabel('Adjusted Close')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'actual_vs_predicted_{model_name}.png'))
        plt.close()
