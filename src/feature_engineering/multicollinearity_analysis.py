# -----------------------------------------------------------------------------
# Submitted by:     u3258928
# Date:             2024 November 02
# Course:           Software Technology 1
# Activity:         Assessment 3: Programming project
# -----------------------------------------------------------------------------


import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
from typing import List, Tuple
import numpy as np
import logging


def calculate_vif(df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    """
    Calculates the Variance Inflation Factor (VIF) for each feature in the DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame.
        features (List[str]): List of feature names to calculate VIF for.

    Returns:
        pd.DataFrame: DataFrame containing features and their corresponding VIF values.
    """
    X = df[features].values
    vif_data = pd.DataFrame()
    vif_data['Feature'] = features
    vif_values = []
    for i in range(len(features)):
        try:
            vif = variance_inflation_factor(X, i)
            vif_values.append(vif)
        except np.linalg.LinAlgError:
            vif_values.append(np.inf)
            logging.warning(f"VIF calculation for feature {features[i]} resulted in infinite value.")
    vif_data['VIF'] = vif_values
    return vif_data


def remove_perfect_multicollinearity(df: pd.DataFrame, features: List[str], exclude_features: List[str] = []) -> Tuple[pd.DataFrame, List[str]]:
    """
    Iteratively removes features that are perfectly correlated with others, excluding specified features.

    Args:
        df (pd.DataFrame): The input DataFrame.
        features (List[str]): List of feature names to evaluate.
        exclude_features (List[str], optional): List of feature names to exclude from removal.

    Returns:
        Tuple[pd.DataFrame, List[str]]: DataFrame with perfectly collinear features removed and list of removed features.
    """
    removed_features = []
    correlation_matrix = df[features].corr().abs()
    upper_triangle = correlation_matrix.where(
        np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
    )
    to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] == 1) and column not in exclude_features]

    while to_drop:
        # Remove the first feature with perfect correlation, recalculate correlations
        feature = to_drop.pop(0)
        df = df.drop(columns=[feature])
        features.remove(feature)
        removed_features.append(feature)

        # Recalculate the upper triangle for updated correlation matrix
        correlation_matrix = df[features].corr().abs()
        upper_triangle = correlation_matrix.where(
            np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
        )
        to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] == 1) and column not in exclude_features]

    return df, removed_features


def remove_multicollinear_features(df: pd.DataFrame, features: List[str], threshold: float = 5.0, exclude_features: List[str] = []) -> Tuple[pd.DataFrame, List[str]]:
    """
    Iteratively removes features with the highest VIF until all VIFs are below the threshold,
    excluding specified features from removal.

    Args:
        df (pd.DataFrame): The input DataFrame.
        features (List[str]): List of feature names to evaluate.
        threshold (float): VIF threshold to remove multicollinear features.
        exclude_features (List[str], optional): List of feature names to exclude from removal.

    Returns:
        Tuple[pd.DataFrame, List[str]]: DataFrame with multicollinear features removed and list of removed features.
    """
    features = features.copy()
    removed_features = []

    # Remove perfectly multicollinear features first, excluding specified features
    df, removed = remove_perfect_multicollinearity(df, features, exclude_features)
    removed_features.extend(removed)

    while True:
        vif_df = calculate_vif(df, features)
        max_vif = vif_df['VIF'].max()
        if max_vif > threshold:
            feature_to_remove = vif_df.loc[vif_df['VIF'].idxmax(), 'Feature']
            if feature_to_remove in exclude_features:
                logging.info(f"Feature '{feature_to_remove}' is excluded from removal despite high VIF.")
                break
            features.remove(feature_to_remove)
            df = df.drop(columns=[feature_to_remove])
            removed_features.append(feature_to_remove)
            logging.info(f"Removed feature '{feature_to_remove}' with VIF={max_vif:.2f}.")
        else:
            break

    return df, removed_features
