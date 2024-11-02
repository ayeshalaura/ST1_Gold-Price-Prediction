# -----------------------------------------------------------------------------
# Submitted by:     u3258928
# Date:             2024 November 02
# Course:           Software Technology 1
# Activity:         Assessment 3: Programming project
# -----------------------------------------------------------------------------


import pandas as pd


def remove_duplicates(df: pd.DataFrame) -> (pd.DataFrame, int):
    """
    Removes duplicate records from the DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        Tuple[pd.DataFrame, int]: The DataFrame without duplicates and the number of duplicates removed.
    """
    initial_count = df.shape[0]
    df_cleaned = df.drop_duplicates()
    duplicates_removed = initial_count - df_cleaned.shape[0]
    return df_cleaned, duplicates_removed
