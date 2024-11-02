# -----------------------------------------------------------------------------
# Submitted by:     u3258928
# Date:             2024 November 02
# Course:           Software Technology 1
# Activity:         Assessment 3: Programming project
# -----------------------------------------------------------------------------


import pandas as pd


def add_date_features(df: pd.DataFrame) -> (pd.DataFrame, list):
    """
    Extracts date-based features from the DataFrame's Date index.

    Args:
        df (pd.DataFrame): The input DataFrame with Date as index.

    Returns:
        Tuple[pd.DataFrame, list]: The DataFrame with new date-based features and a list of feature names.
    """
    df['year'] = df.index.year
    df['month'] = df.index.month
    df['weekday'] = df.index.weekday
    new_features = ['year', 'month', 'weekday']
    return df, new_features
