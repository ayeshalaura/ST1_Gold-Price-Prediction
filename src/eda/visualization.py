# -----------------------------------------------------------------------------
# Submitted by:     u3258928
# Date:             2024 November 02
# Course:           Software Technology 1
# Activity:         Assessment 3: Programming project
# -----------------------------------------------------------------------------


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
import os
from typing import List

def decompose_adj_close(df: pd.DataFrame, model: str = 'additive', period: int = 30, output_dir: str = "reports/visualizations") -> None:
    """
    Decomposes the Adjusted Close price into trend, seasonality, and residual components.
    
    Args:
        df (pd.DataFrame): The preprocessed dataset with 'Adj Close' as a column.
        model (str): Type of seasonal component. 'additive' or 'multiplicative'.
        period (int): The periodicity of the data.
        output_dir (str): Directory to save the decomposition plots.
    """
    os.makedirs(output_dir, exist_ok=True)
    decomposition = seasonal_decompose(df['Adj Close'], model=model, period=period)
    
    fig = decomposition.plot()
    fig.set_size_inches(14, 7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'adj_close_decomposition.png'))
    plt.close()

def plot_adj_close_distribution(df: pd.DataFrame, output_dir: str = "reports/visualizations") -> None:
    """
    Plots the distribution of the Adjusted Close prices.

    Args:
        df (pd.DataFrame): The preprocessed dataset.
        output_dir (str): Directory to save the plot.
    """
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(10, 6))
    sns.histplot(df['Adj Close'], kde=True)
    plt.title('Distribution of Adjusted Close Prices')
    plt.xlabel('Adjusted Close')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'adj_close_distribution.png'))
    plt.close()


def plot_adj_close_over_time(df: pd.DataFrame, output_dir: str = "reports/visualizations") -> None:
    """
    Plots the Adjusted Close prices over time.

    Args:
        df (pd.DataFrame): The preprocessed dataset.
        output_dir (str): Directory to save the plot.
    """
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(14, 7))
    sns.lineplot(data=df, x=df.index, y='Adj Close')
    plt.title('Adjusted Close Price Over Time')
    plt.xlabel('Date')
    plt.ylabel('Adjusted Close')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'adj_close_over_time.png'))
    plt.close()


def plot_column_distributions(df: pd.DataFrame, output_dir: str = "reports/visualizations") -> None:
    """
    Plots histograms for columns with more than 12 unique values and bar charts for the rest.

    Args:
        df (pd.DataFrame): The preprocessed dataset.
        output_dir (str): Directory to save the plots.
    """
    os.makedirs(output_dir, exist_ok=True)
    for col in df.columns:
        unique_values = df[col].nunique()
        if unique_values > 12:
            plt.figure(figsize=(10, 6))
            sns.histplot(df[col].dropna(), kde=True)
            plt.title(f'Distribution of {col}')
            plt.xlabel(col)
            plt.ylabel('Frequency')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'{col}_histogram.png'))
            plt.close()
        else:
            plt.figure(figsize=(10, 6))
            sns.countplot(data=df, x=col)
            plt.title(f'Distribution of {col}')
            plt.xlabel(col)
            plt.ylabel('Count')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'{col}_bar_chart.png'))
            plt.close()


def get_column_info(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generates a DataFrame containing column information and their classifications.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: DataFrame with columns: 'Data Type', 'Non-Null Count', 'Classification'.
    """
    # Gather basic column information
    column_info = pd.DataFrame({
        'Data Type': df.dtypes.astype(str),
        'Non-Null Count': df.notnull().sum()
    }, index=df.columns)

    # Classify columns
    quantitative_cols: List[str] = df.select_dtypes(include=['float64', 'int32', 'int64']).columns.tolist()
    categorical_cols: List[str] = df.select_dtypes(include=['category', 'object']).columns.tolist()

    # Adjust classifications for specific columns
    for col in ['month', 'weekday']:
        if col in quantitative_cols:
            quantitative_cols.remove(col)
            categorical_cols.append(col)

    def classify_column(col: str) -> str:
        if col in quantitative_cols:
            return 'Quantitative'
        elif col in categorical_cols:
            return 'Categorical'
        else:
            return 'Qualitative'

    column_info['Classification'] = column_info.index.map(classify_column)

    return column_info


def plot_adj_close_decomposition(df: pd.DataFrame, output_dir: str = "reports/visualizations") -> None:
    """
    Plots the decomposition of Adjusted Close prices into trend, seasonality, and residuals.

    Args:
        df (pd.DataFrame): The preprocessed dataset.
        output_dir (str): Directory to save the decomposition plot.
    """
    from time_series_analysis import decompose_adj_close

    decompose_adj_close(df, model='additive', period=30, output_dir=output_dir)
    plt.close()
