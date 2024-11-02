# -----------------------------------------------------------------------------
# Submitted by:     u3258928
# Date:             2024 November 02
# Course:           Software Technology 1
# Activity:         Assessment 3: Programming project
# -----------------------------------------------------------------------------


import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from typing import Tuple


def split_data(
    df: pd.DataFrame,
    target_column: str = 'Adj Close',
    n_splits: int = 5,
    test_size: float = 0.2
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Splits the dataset into training and testing sets while maintaining temporal order.

    Args:
        df (pd.DataFrame): The input DataFrame.
        target_column (str): The name of the target variable.
        n_splits (int): Number of splits for TimeSeriesSplit.
        test_size (float): Proportion of the dataset to include in the test split.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]: 
            X_train, X_test, y_train, y_test
    """
    X = df.drop(columns=[target_column])
    y = df[target_column]

    tscv = TimeSeriesSplit(n_splits=n_splits)
    split = list(tscv.split(X))

    train_indices, test_indices = split[-1]  # Last split for final train-test

    X_train, X_test = X.iloc[train_indices], X.iloc[test_indices]
    y_train, y_test = y.iloc[train_indices], y.iloc[test_indices]

    print(f"Training set size: {X_train.shape[0]} samples")
    print(f"Testing set size: {X_test.shape[0]} samples")
    print(f"Training period: {X_train.index.min()} to {X_train.index.max()}")
    print(f"Testing period: {X_test.index.min()} to {X_test.index.max()}")

    return X_train, X_test, y_train, y_test
