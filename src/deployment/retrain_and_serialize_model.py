# -----------------------------------------------------------------------------
# Submitted by:     u3258928
# Date:             2024 November 02
# Course:           Software Technology 1
# Activity:         Assessment 3: Programming project
# -----------------------------------------------------------------------------


import logging
import pandas as pd
import os
import joblib

from utils.logging_config import setup_logging
from modeling.training.evaluate_models import ModelEvaluator


def select_best_model(metrics_filepath: str) -> str:
    """
    Selects the best model based on evaluation metrics.

    Args:
        metrics_filepath (str): Path to the evaluation metrics CSV file.

    Returns:
        str: The name of the best model.
    """
    metrics_df = pd.read_csv(metrics_filepath)

    best_models = {}
    # Define metric priorities: lower is better for MAE, MSE, RMSE; higher is better for R2_Score
    metric_priorities = {
        'MAE': 'min',
        'MSE': 'min',
        'RMSE': 'min',
        'R2_Score': 'max'
    }

    for metric, preference in metric_priorities.items():
        if preference == 'min':
            best_model = metrics_df.loc[metrics_df[metric].idxmin(), 'Model']
        else:
            best_model = metrics_df.loc[metrics_df[metric].idxmax(), 'Model']
        best_models[best_model] = best_models.get(best_model, 0) + 1

    # Tally the best counts
    best_model_tally = pd.Series(best_models)
    max_tally = best_model_tally.max()
    top_models = best_model_tally[best_model_tally == max_tally].index.tolist()

    if len(top_models) == 1:
        return top_models[0]
    else:
        # Resolve ties by comparing metrics in order of priority
        for metric, preference in metric_priorities.items():
            if preference == 'min':
                best_value = metrics_df[metrics_df['Model'].isin(top_models)][metric].min()
                candidates = metrics_df[
                    (metrics_df['Model'].isin(top_models)) &
                    (metrics_df[metric] == best_value)
                ]['Model'].tolist()
            else:
                best_value = metrics_df[metrics_df['Model'].isin(top_models)][metric].max()
                candidates = metrics_df[
                    (metrics_df['Model'].isin(top_models)) &
                    (metrics_df[metric] == best_value)
                ]['Model'].tolist()

            if len(candidates) == 1:
                return candidates[0]
            else:
                top_models = candidates  # Continue to next metric

        # If still tied after all metrics, return the first model
        return top_models[0]


def retrain_and_serialize_model(df_finalized: pd.DataFrame):
    """
    Retrains the best model on the entire dataset with scaled features and saves it.

    Args:
        df_finalized (pd.DataFrame): The preprocessed and finalized DataFrame.
    """
    setup_logging()
    logging.info("=== Retraining the Best Model on Entire Dataset ===")

    # Load evaluation metrics
    metrics_filepath = os.path.join("reports", "evaluation_metrics.csv")
    best_model_name = select_best_model(metrics_filepath)
    logging.info(f"Best model selected: {best_model_name}")
    print(f"Best model selected: {best_model_name}")

    # Load the best model
    tuned_model_path = os.path.join("models", "tuned", f"{best_model_name}_tuned.joblib")
    if not os.path.exists(tuned_model_path):
        logging.error(f"Tuned model file {tuned_model_path} does not exist.")
        raise FileNotFoundError(f"Tuned model file {tuned_model_path} does not exist.")
    best_model = joblib.load(tuned_model_path)
    logging.info(f"Loaded tuned model from {tuned_model_path}")

    # Load and apply the scaler
    scaler_path = os.path.join("models", "scaler.joblib")
    if not os.path.exists(scaler_path):
        logging.error(f"Scaler file {scaler_path} does not exist.")
        raise FileNotFoundError(f"Scaler file {scaler_path} does not exist.")
    scaler = joblib.load(scaler_path)
    logging.info(f"Loaded scaler from {scaler_path}")

    # Prepare features and target
    X = df_finalized.drop(columns=['Adj Close'])
    y = df_finalized['Adj Close']

    # Scale the features
    X_scaled = scaler.transform(X)
    X_scaled = pd.DataFrame(X_scaled, index=X.index, columns=X.columns)
    logging.info("Features scaled using the loaded scaler.")

    # Retrain the model on the entire scaled dataset
    best_model.fit(X_scaled, y)
    logging.info(f"Best model '{best_model_name}' retrained on entire dataset.")

    # Serialize and save the retrained model
    deployed_model_path = os.path.join("models", "deployed", f"{best_model_name}_deployed.joblib")
    os.makedirs(os.path.dirname(deployed_model_path), exist_ok=True)
    joblib.dump(best_model, deployed_model_path)
    logging.info(f"Retrained model saved to {deployed_model_path}")
    print(f"Retrained model saved to {deployed_model_path}")

    logging.info("=== Retraining and Serialization Completed ===")
