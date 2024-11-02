# -----------------------------------------------------------------------------
# Submitted by:     u3258928
# Date:             2024 November 02
# Course:           Software Technology 1
# Activity:         Assessment 3: Programming project
# -----------------------------------------------------------------------------


import pandas as pd


def handle_missing_values(df: pd.DataFrame) -> (pd.DataFrame, int):
    """
    Detects and handles missing values in the DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        Tuple[pd.DataFrame, int]: The DataFrame with missing values handled and the number of values imputed.
    """
    initial_missing = df.isnull().sum().sum()

    # Remove rows with minimal missing data
    df_cleaned = df.dropna()

    # Check for any remaining missing values
    remaining_missing = df_cleaned.isnull().sum().sum()
    if remaining_missing > 0:
        # Impute continuous variables with median
        continuous_vars = df_cleaned.select_dtypes(include=['float64', 'int64']).columns
        df_cleaned[continuous_vars] = df_cleaned[continuous_vars].fillna(df_cleaned[continuous_vars].median())

        # Impute categorical variables with mode
        categorical_vars = df_cleaned.select_dtypes(include=['object', 'category']).columns
        for col in categorical_vars:
            if df_cleaned[col].isnull().any():
                df_cleaned[col].fillna(df_cleaned[col].mode()[0], inplace=True)

    final_missing = df_cleaned.isnull().sum().sum()
    missing_imputed = initial_missing - final_missing
    return df_cleaned, missing_imputed
