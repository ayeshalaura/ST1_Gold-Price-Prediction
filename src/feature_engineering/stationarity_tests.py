# -----------------------------------------------------------------------------
# Submitted by:     u3258928
# Date:             2024 November 02
# Course:           Software Technology 1
# Activity:         Assessment 3: Programming project
# -----------------------------------------------------------------------------


import pandas as pd
from statsmodels.tsa.stattools import adfuller
import numpy as np
import logging


def adf_test(series: pd.Series, signif: float = 0.05, name: str = "") -> bool:
    """
    Performs the Augmented Dickey-Fuller test to check stationarity.

    Args:
        series (pd.Series): The time series data.
        signif (float): Significance level.
        name (str): Name of the series.

    Returns:
        bool: True if stationary, False otherwise.
    """
    result = adfuller(series.dropna(), autolag='AIC')
    p_value = result[1]
    is_stationary = p_value < signif
    if not is_stationary:
        logging.info(f"Series '{name}' is non-stationary (p-value={p_value:.4f}).")
    else:
        logging.info(f"Series '{name}' is stationary (p-value={p_value:.4f}).")
    return is_stationary


def apply_transformation(series: pd.Series) -> pd.Series:
    """
    Applies log transformation to stabilize variance.

    Args:
        series (pd.Series): The time series data.

    Returns:
        pd.Series: Transformed series.
    """
    return np.log(series)


def difference_series(series: pd.Series, lag: int = 1) -> pd.Series:
    """
    Applies differencing to remove trends.

    Args:
        series (pd.Series): The time series data.
        lag (int): The lag order.

    Returns:
        pd.Series: Differenced series.
    """
    return series.diff(lag)


def ensure_stationarity(df: pd.DataFrame, threshold: float = 0.05) -> pd.DataFrame:
    """
    Ensures all key variables are stationary by applying transformations.

    Args:
        df (pd.DataFrame): The input DataFrame.
        threshold (float): Significance level for ADF test.

    Returns:
        pd.DataFrame: DataFrame with stationary variables.
    """
    transformed_columns = []
    exclude_columns = ['year', 'month', 'weekday']
    # Identify all trend-related columns to exclude
    trend_columns = [col for col in df.columns if col.endswith('_Trend')]
    exclude_columns.extend(trend_columns)
    
    for column in df.select_dtypes(include=[np.number]).columns:
        if column == 'Adj Close' or column in exclude_columns:
            continue
        if not adf_test(df[column], signif=threshold, name=column):
            # Apply log transformation if all values are positive
            if (df[column] > 0).all():
                df[column] = apply_transformation(df[column])
                logging.info(f"Applied log transformation to '{column}'.")
            else:
                logging.info(f"Skipping log transformation for '{column}' due to non-positive values.")
            
            # Re-test stationarity after transformation
            if not adf_test(df[column], signif=threshold, name=column):
                # Apply differencing
                df[column] = difference_series(df[column])
                logging.info(f"Applied differencing to '{column}'.")
                
                # Drop NaN values introduced by differencing
                df.dropna(subset=[column], inplace=True)
    return df
