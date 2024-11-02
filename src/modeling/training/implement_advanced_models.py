# -----------------------------------------------------------------------------
# Submitted by:     u3258928
# Date:             2024 November 02
# Course:           Software Technology 1
# Activity:         Assessment 3: Programming project
# -----------------------------------------------------------------------------


import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
import xgboost as xgb
from typing import Dict, Any
import joblib
import os


class AdvancedModelsTrainer:
    def __init__(self):
        """
        Initializes the Advanced Models Trainer with predefined models.
        """
        self.models: Dict[str, Any] = {
            'linear_regression': LinearRegression(),
            'decision_tree': DecisionTreeRegressor(random_state=42),
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'ada_boost': AdaBoostRegressor(n_estimators=100, random_state=42),
            'xgboost': xgb.XGBRegressor(n_estimators=100, random_state=42, objective='reg:squarederror'),
            'k_nearest_neighbors': KNeighborsRegressor(n_neighbors=5),
            'svm': SVR(kernel='rbf')
        }

    def train_models(self, X_train: pd.DataFrame, y_train: pd.Series):
        """
        Trains all predefined models.

        Args:
            X_train (pd.DataFrame): Training feature set.
            y_train (pd.Series): Training target set.
        """
        for name, model in self.models.items():
            model.fit(X_train, y_train)
            print(f"Trained {name} model.")

    def save_models(self, directory: str = "models/advanced"):
        """
        Saves all trained models to the specified directory.

        Args:
            directory (str): The directory to save the models.
        """
        os.makedirs(directory, exist_ok=True)
        for name, model in self.models.items():
            path = os.path.join(directory, f"{name}.joblib")
            joblib.dump(model, path)
            print(f"Saved {name} model to {path}.")

    def load_models(self, directory: str = "models/advanced"):
        """
        Loads all trained models from the specified directory.

        Args:
            directory (str): The directory from which to load the models.
        """
        for name in self.models.keys():
            path = os.path.join(directory, f"{name}.joblib")
            if os.path.exists(path):
                self.models[name] = joblib.load(path)
                print(f"Loaded {name} model from {path}.")
            else:
                print(f"Model file {path} does not exist.")

    def get_models(self) -> Dict[str, Any]:
        """
        Returns the dictionary of models.

        Returns:
            Dict[str, Any]: The models dictionary.
        """
        return self.models
