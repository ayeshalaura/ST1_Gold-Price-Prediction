# -----------------------------------------------------------------------------
# Submitted by:     u3258928
# Date:             2024 November 02
# Course:           Software Technology 1
# Activity:         Assessment 3: Programming project
# -----------------------------------------------------------------------------


import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
import xgboost as xgb
from typing import Dict, Any
import joblib
import os
from itertools import product


class HyperparameterTuner:
    def __init__(self):
        """
        Initializes the Hyperparameter Tuner with models and their respective hyperparameter grids.
        """
        self.param_grids: Dict[str, Dict[str, Any]] = {
            'linear_regression': {
                'fit_intercept': [True, False],
            },
            'decision_tree': {
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'random_forest': {
                'n_estimators': [100, 200],
                'max_depth': [None, 10],
                'min_samples_split': [2, 5]
            },
            'ada_boost': {
                'n_estimators': [50, 100],
                'learning_rate': [0.01, 0.1, 1.0]
            },
            'xgboost': {
                'n_estimators': [100, 200],
                'learning_rate': [0.01, 0.1],
                'max_depth': [3, 5]
            },
            'k_nearest_neighbors': {
                'n_neighbors': [3, 5, 7, 9],
                'weights': ['uniform', 'distance'],
                'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
            },
            'svm': {
                'C': [1, 10],
                'gamma': ['scale'],
                'kernel': ['rbf']
            }
        }
        self.best_estimators: Dict[str, Any] = {}

    def tune_model(self, model_name: str, model, param_grid: Dict[str, Any], X_train: pd.DataFrame, y_train: pd.Series):
        """
        Performs hyperparameter tuning using GridSearchCV.

        Args:
            model_name (str): The name of the model.
            model: The model instance.
            param_grid (Dict[str, Any]): The hyperparameter grid.
            X_train (pd.DataFrame): Training features.
            y_train (pd.Series): Training target.
        """
        num_combinations = len(list(product(*param_grid.values())))
        print(f"Starting hyperparameter tuning for {model_name} with {num_combinations} combinations...")

        search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=3,
            scoring='neg_mean_squared_error',
            verbose=1,
            n_jobs=-1
        )
        print(f"Using GridSearchCV for {model_name}.")
        search.fit(X_train, y_train)
        self.best_estimators[model_name] = search.best_estimator_
        print(f"Best parameters for {model_name}: {search.best_params_}")
        print(f"Best score for {model_name}: {search.best_score_}")

    def tune_all_models(self, X_train: pd.DataFrame, y_train: pd.Series):
        """
        Tunes hyperparameters for all models in the parameter grid.

        Args:
            X_train (pd.DataFrame): Training features.
            y_train (pd.Series): Training target.
        """
        for model_name, param_grid in self.param_grids.items():
            if model_name == 'linear_regression':
                model = LinearRegression()
            elif model_name == 'decision_tree':
                model = DecisionTreeRegressor(random_state=42)
            elif model_name == 'random_forest':
                model = RandomForestRegressor(random_state=42)
            elif model_name == 'ada_boost':
                model = AdaBoostRegressor(random_state=42)
            elif model_name == 'xgboost':
                model = xgb.XGBRegressor(random_state=42, objective='reg:squarederror')
            elif model_name == 'k_nearest_neighbors':
                model = KNeighborsRegressor()
            elif model_name == 'svm':
                model = SVR()
            else:
                continue
            self.tune_model(model_name, model, param_grid, X_train, y_train)

    def save_best_estimators(self, directory: str = "models/tuned"):
        """
        Saves the best estimators after tuning.

        Args:
            directory (str): The directory to save the models.
        """
        os.makedirs(directory, exist_ok=True)
        for model_name, estimator in self.best_estimators.items():
            path = os.path.join(directory, f"{model_name}_tuned.joblib")
            joblib.dump(estimator, path)
            print(f"Saved tuned {model_name} model to {path}.")

    def load_best_estimators(self, directory: str = "models/tuned"):
        """
        Loads the tuned estimators from the specified directory.

        Args:
            directory (str): The directory from which to load the models.
        """
        for model_name in self.param_grids.keys():
            path = os.path.join(directory, f"{model_name}_tuned.joblib")
            if os.path.exists(path):
                if model_name == 'linear_regression':
                    estimator = LinearRegression()
                elif model_name == 'decision_tree':
                    estimator = DecisionTreeRegressor()
                elif model_name == 'random_forest':
                    estimator = RandomForestRegressor()
                elif model_name == 'ada_boost':
                    estimator = AdaBoostRegressor()
                elif model_name == 'xgboost':
                    estimator = xgb.XGBRegressor(objective='reg:squarederror')
                elif model_name == 'k_nearest_neighbors':
                    estimator = KNeighborsRegressor()
                elif model_name == 'svm':
                    estimator = SVR()
                else:
                    continue
                estimator = joblib.load(path)
                self.best_estimators[model_name] = estimator
                print(f"Loaded tuned {model_name} model from {path}.")
            else:
                print(f"Tuned model file {path} does not exist.")
