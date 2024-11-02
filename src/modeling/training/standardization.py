# -----------------------------------------------------------------------------
# Submitted by:     u3258928
# Date:             2024 November 02
# Course:           Software Technology 1
# Activity:         Assessment 3: Programming project
# -----------------------------------------------------------------------------


import pandas as pd
from sklearn.preprocessing import StandardScaler
from typing import Tuple
import joblib
import os


def standardize_data(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    scaler_path: str = "models/scaler.joblib"
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Standardizes the features using StandardScaler.

    Args:
        X_train (pd.DataFrame): Training feature set.
        X_test (pd.DataFrame): Testing feature set.
        scaler_path (str): Path to save the fitted scaler.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Standardized training and testing feature sets.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Convert back to DataFrame for consistency
    X_train_scaled = pd.DataFrame(X_train_scaled, index=X_train.index, columns=X_train.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, index=X_test.index, columns=X_test.columns)

    # Save the scaler for future use
    os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
    joblib.dump(scaler, scaler_path)
    print(f"Scaler saved to {scaler_path}")

    print("Feature standardization completed successfully.")
    return X_train_scaled, X_test_scaled
