# -----------------------------------------------------------------------------
# Submitted by:     u3258928
# Date:             2024 November 02
# Course:           Software Technology 1
# Activity:         Assessment 3: Programming project
# -----------------------------------------------------------------------------


import pandas as pd
from typing import List, Tuple, Optional


def create_lagged_features(
    df: pd.DataFrame,
    lags: List[int],
    drop_original: bool = True,
    exclude_columns: Optional[List[str]] = None,
    target_column: str = 'Adj Close'
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Creates lagged features for specified numeric columns in the DataFrame and retains the target column.

    Args:
        df (pd.DataFrame): The input DataFrame with a DateTime index.
        lags (List[int]): List of lag periods (e.g., [1, 5, 22]).
        drop_original (bool): Whether to drop the original columns after creating lags.
        exclude_columns (List[str], optional): List of columns to exclude from lagging.
        target_column (str): Name of the target column to retain.

    Returns:
        Tuple[pd.DataFrame, List[str]]: The DataFrame with new lagged features and a list of new feature names.
    """
    if exclude_columns is None:
        exclude_columns = []
    
    # Define columns to lag
    numeric_columns = df.select_dtypes(include=['float64', 'int32', 'int64']).columns
    numeric_columns = [col for col in numeric_columns if col not in exclude_columns and col != target_column]
    
    lagged_data = {}
    for col in numeric_columns:
        for lag in lags:
            lagged_data[f"{col}_lag_{lag}"] = df[col].shift(lag)
    
    # Create lagged target column separately if needed
    for lag in lags:
        lagged_data[f"{target_column}_lag_{lag}"] = df[target_column].shift(lag)

    # Create DataFrame for lagged data and concatenate with the original DataFrame
    df_lagged = pd.concat([df, pd.DataFrame(lagged_data, index=df.index)], axis=1)
    new_features = list(lagged_data.keys())

    # Drop original non-target columns if specified
    if drop_original:
        df_lagged.drop(columns=numeric_columns, inplace=True)

    # Drop rows with NaN values introduced by lagging
    df_lagged.dropna(inplace=True)
    return df_lagged, new_features
