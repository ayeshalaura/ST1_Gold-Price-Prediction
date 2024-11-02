# -----------------------------------------------------------------------------
# Submitted by:     u3258928
# Date:             2024 November 02
# Course:           Software Technology 1
# Activity:         Assessment 3: Programming project
# -----------------------------------------------------------------------------


import pandas as pd

def finalize_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Finalizes the feature set by converting categorical variables to numeric.

    Args:
        df (pd.DataFrame): The input DataFrame with categorical features.

    Returns:
        pd.DataFrame: The DataFrame with categorical variables encoded as numeric.
    """
    categorical_features = ['month', 'weekday']
    df_finalized = pd.get_dummies(df, columns=categorical_features, drop_first=True)
    return df_finalized
