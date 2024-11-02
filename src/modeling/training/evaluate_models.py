# -----------------------------------------------------------------------------
# Submitted by:     u3258928
# Date:             2024 November 02
# Course:           Software Technology 1
# Activity:         Assessment 3: Programming project
# -----------------------------------------------------------------------------


import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from typing import Dict, Any
import math
import joblib
import os


class ModelEvaluator:
    def __init__(self):
        """
        Initializes the Model Evaluator.
        """
        self.metrics = {}
        self.predictions = {}

    def evaluate(self, y_true: pd.Series, y_pred: pd.Series) -> Dict[str, float]:
        """
        Calculates evaluation metrics.

        Args:
            y_true (pd.Series): True target values.
            y_pred (pd.Series): Predicted target values.

        Returns:
            Dict[str, float]: Dictionary of evaluation metrics.
        """
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = math.sqrt(mse)
        r2 = r2_score(y_true, y_pred)

        return {
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'R2_Score': r2
        }

    def evaluate_model(self, model_name: str, model, X_test: pd.DataFrame, y_test: pd.Series):
        """
        Evaluates a single model and stores its metrics and predictions.

        Args:
            model_name (str): The name of the model.
            model: The trained model or baseline.
            X_test (pd.DataFrame): Testing features (unused for baseline).
            y_test (pd.Series): Testing target.
        """
        if model_name != 'baseline':
            y_pred = model.predict(X_test)
            metrics = self.evaluate(y_test, y_pred)
            self.metrics[model_name] = metrics
            self.predictions[model_name] = y_pred  # Store predictions
            print(f"Evaluation metrics for {model_name}: {metrics}")

    def evaluate_all_models(self, models: Dict[str, Any], X_test: pd.DataFrame, y_test: pd.Series):
        """
        Evaluates all provided models.

        Args:
            models (Dict[str, Any]): Dictionary of models to evaluate.
            X_test (pd.DataFrame): Testing features.
            y_test (pd.Series): Testing target.
        """
        for model_name, model in models.items():
            self.evaluate_model(model_name, model, X_test, y_test)

    def save_metrics(self, filepath: str = "reports/evaluation_metrics.csv"):
        """
        Saves the evaluation metrics to a CSV file.

        Args:
            filepath (str): The file path to save the metrics.
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        metrics_df = pd.DataFrame.from_dict(self.metrics, orient='index').reset_index()
        metrics_df.rename(columns={'index': 'Model'}, inplace=True)
        metrics_df.to_csv(filepath, index=False)
        print(f"Saved evaluation metrics to {filepath}.")

    def load_metrics(self, filepath: str = "reports/evaluation_metrics.csv"):
        """
        Loads evaluation metrics from a CSV file.

        Args:
            filepath (str): The file path from which to load the metrics.
        """
        if os.path.exists(filepath):
            metrics_df = pd.read_csv(filepath)
            self.metrics = metrics_df.set_index('Model').to_dict(orient='index')
            print(f"Loaded evaluation metrics from {filepath}.")
        else:
            print(f"Metrics file {filepath} does not exist.")
