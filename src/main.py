# -----------------------------------------------------------------------------
# Submitted by:     u3258928
# Date:             2024 November 02
# Course:           Software Technology 1
# Activity:         Assessment 3: Programming project
# -----------------------------------------------------------------------------


import logging

from preprocessing.load_data import load_dataset
from preprocessing.remove_duplicates import remove_duplicates
from preprocessing.handle_missing_values import handle_missing_values
from preprocessing.date_features import add_date_features
from eda.visualization import (
    plot_adj_close_distribution,
    plot_adj_close_over_time,
    plot_column_distributions,
    plot_adj_close_decomposition
)
from eda.visualization import decompose_adj_close
from feature_engineering.correlation_analysis import conduct_correlation_analysis, get_highly_correlated_features
from feature_engineering.lagged_features import create_lagged_features
from feature_engineering.multicollinearity_analysis import remove_multicollinear_features, calculate_vif
from feature_engineering.stationarity_tests import ensure_stationarity
from feature_engineering.finalize_features import finalize_features
from modeling.training.split_data import split_data
from modeling.training.standardization import standardize_data
from modeling.training.implement_advanced_models import AdvancedModelsTrainer
from modeling.training.hyperparameter_tuning import HyperparameterTuner
from modeling.training.evaluate_models import ModelEvaluator
from modeling.visualization import plot_model_comparison, plot_actual_vs_predicted
from utils.logging_config import setup_logging
from deployment.retrain_and_serialize_model import retrain_and_serialize_model
from utils.pipeline import Pipeline
from utils.save_demo_row import save_last_rows


def load_and_display_dataset(context):
    print("\n\n--- Loading Dataset ---")
    df = load_dataset()
    logging.info("Dataset loaded successfully.")
    print("Dataset loaded successfully.")
    print("\n\n--- Displaying Initial Data ---")
    print(df.head())
    context['df'] = df
    return context


def remove_duplicates_step(context):
    df = context['df']
    print("\n\n--- Removing Duplicates ---")
    df, duplicates_removed = remove_duplicates(df)
    duplicates_remaining = df.duplicated().sum()
    logging.info(f"Duplicate rows removed: {duplicates_removed}; Duplicate rows remaining: {duplicates_remaining}")
    print(f"Duplicate rows removed: {duplicates_removed}")
    print(f"Duplicate rows remaining: {duplicates_remaining}")
    context['df'] = df
    return context


def handle_missing_values_step(context):
    df = context['df']
    print("\n\n--- Handling Missing Values ---")
    df, missing_imputed = handle_missing_values(df)
    missing_values = df.isna().sum().sum()
    logging.info(f"Missing values imputed: {missing_imputed}; Missing values remaining: {missing_values}")
    print(f"Missing values imputed: {missing_imputed}")
    print(f"Missing values remaining: {missing_values}")
    context['df'] = df
    return context


def add_date_features_step(context):
    df = context['df']
    print("\n\n--- Adding Date Features ---")
    df, new_features = add_date_features(df)
    logging.info(f"New date-based features added: {', '.join(new_features)}")
    print(f"New date-based features added: {', '.join(new_features)}")
    context['df'] = df
    return context


def plot_adj_close_distribution_step(context):
    df = context['df']
    print("\n\n--- Plotting Adjusted Close Distribution ---")
    plot_adj_close_distribution(df)
    logging.info("Adjusted Close distribution plot created successfully.")
    print("Adjusted Close distribution plot created successfully.")
    return context


def plot_adj_close_over_time_step(context):
    df = context['df']
    print("\n\n--- Plotting Adjusted Close Over Time ---")
    plot_adj_close_over_time(df)
    logging.info("Adjusted Close over time plot created successfully.")
    print("Adjusted Close over time plot created successfully.")
    return context


def plot_column_distributions_step(context):
    df = context['df']
    print("\n\n--- Plotting Column Distributions ---")
    plot_column_distributions(df)
    logging.info("Column distributions plots created successfully.")
    print("Column distributions plots created successfully.")
    return context


def plot_correlation_matrix_step(context):
    df = context['df']
    print("\n\n--- Plotting Correlation Matrix ---")
    conduct_correlation_analysis(df)
    logging.info("Correlation matrix heatmap created successfully.")
    print("Correlation matrix heatmap created successfully.")
    return context


def plot_adj_close_decomposition_step(context):
    df = context['df']
    print("\n\n--- Plotting Adjusted Close Decomposition ---")
    decompose_adj_close(df)
    logging.info("Adjusted Close decomposition plot created successfully.")
    print("Adjusted Close decomposition plot created successfully.")
    return context


def ensure_stationarity_step(context):
    df = context['df']
    print("\n\n--- Ensuring Stationarity ---")
    df_stationary = ensure_stationarity(df)
    logging.info("Stationarity ensured.")
    print("Stationarity ensured.")
    context['df'] = df_stationary
    return context


def multicollinearity_analysis_step(context):
    df = context['df']
    print("\n\n--- Conducting Multicollinearity Analysis ---")
    features = df.select_dtypes(include=['float64', 'int32', 'int64']).columns.tolist()
    exclude_features = ['Adj Close']
    df_reduced, removed_features = remove_multicollinear_features(df, features, threshold=5.0, exclude_features=exclude_features)
    removed_count = len(removed_features)
    if removed_count > 0:
        logging.info(f"Number of multicollinear features removed: {removed_count}")
        print(f"Number of multicollinear features removed: {removed_count}")
    else:
        logging.info("No multicollinear features found above the threshold.")
        print("No multicollinear features found above the threshold.")
    vif_df = calculate_vif(df_reduced, df_reduced.columns.tolist())
    print("VIF values for remaining features:")
    print(vif_df)
    context['df'] = df_reduced
    return context


def create_lagged_features_step(context):
    df = context['df']
    print("\n\n--- Creating Lagged Features ---")
    lags = [1, 5, 22]
    exclude_columns = ['year', 'month', 'weekday']
    df_with_lags, new_lagged_features = create_lagged_features(
        df, lags, drop_original=True, exclude_columns=exclude_columns, target_column='Adj Close'
    )
    logging.info(f"Lagged features created: {len(new_lagged_features)}")
    print(f"Lagged features created: {len(new_lagged_features)}")
    context['df'] = df_with_lags
    return context


def save_last_rows_step(context):
    df = context['df']
    print("\n\n--- Saving Last 100 Rows for Demo ---")
    save_last_rows(df, num_rows=100)
    logging.info("Last 100 rows of the dataset saved for demo purposes.")
    print("Last 100 rows of the dataset saved for demo purposes.")
    
    # Drop the last 100 rows from the DataFrame
    context['df'] = df.iloc[:-100]
    logging.info("Last 100 rows removed from the dataset after saving.")
    print("Last 100 rows removed from the dataset after saving.")
    return context


def finalize_features_step(context):
    df = context['df']
    print("\n\n--- Finalizing Features ---")
    df_finalized = finalize_features(df)
    logging.info("Categorical variables encoded successfully.")
    print("Categorical variables encoded successfully.")
    context['df'] = df_finalized
    return context


def split_data_step(context):
    df = context['df']
    print("\n\n--- Splitting Data into Training and Testing Sets ---")
    X_train, X_test, y_train, y_test = split_data(df)
    logging.info("Data splitting completed successfully.")
    context['X_train'] = X_train
    context['X_test'] = X_test
    context['y_train'] = y_train
    context['y_test'] = y_test
    return context


def standardize_data_step(context):
    X_train = context['X_train']
    X_test = context['X_test']
    print("\n\n--- Standardizing Features ---")
    X_train_scaled, X_test_scaled = standardize_data(X_train, X_test)
    logging.info("Data standardization completed successfully.")
    context['X_train_scaled'] = X_train_scaled
    context['X_test_scaled'] = X_test_scaled
    return context


def implement_advanced_models_step(context):
    X_train_scaled = context['X_train_scaled']
    y_train = context['y_train']
    print("\n\n--- Implementing Advanced Models ---")
    advanced_trainer = AdvancedModelsTrainer()
    advanced_trainer.train_models(X_train_scaled, y_train)
    advanced_trainer.save_models()
    logging.info("Advanced models trained and saved successfully.")
    print("Advanced models trained and saved successfully.")
    context['advanced_models'] = advanced_trainer.get_models()
    return context


def hyperparameter_tuning_step(context):
    advanced_models = context['advanced_models']
    X_train_scaled = context['X_train_scaled']
    y_train = context['y_train']
    print("\n\n--- Hyperparameter Tuning ---")
    tuner = HyperparameterTuner()
    tuner.tune_all_models(X_train_scaled, y_train)
    tuner.save_best_estimators()
    logging.info("Hyperparameter tuning completed and models saved successfully.")
    print("Hyperparameter tuning completed and models saved successfully.")
    context['tuned_models'] = tuner.best_estimators
    return context


def evaluate_models_step(context):
    tuned_models = context['tuned_models']
    advanced_models = context['advanced_models']
    X_test_scaled = context['X_test_scaled']
    y_test = context['y_test']
    print("\n\n--- Evaluating Models ---")
    evaluator = ModelEvaluator()
    all_models = tuned_models.copy()
    all_models.update({name: model for name, model in advanced_models.items() if name not in tuned_models})
    evaluator.evaluate_all_models(all_models, X_test_scaled, y_test)
    evaluator.save_metrics()
    logging.info("Model evaluation completed and metrics saved successfully.")
    print("Model evaluation completed and metrics saved successfully.")
    context['metrics'] = evaluator.metrics
    context['evaluator'] = evaluator
    return context


def visualize_model_comparison_step(context):
    metrics = context['metrics']
    print("\n\n--- Visualizing Model Comparison ---")
    plot_model_comparison(metrics)
    logging.info("Model comparison visualization created successfully.")
    print("Model comparison visualization created successfully.")
    return context


def visualize_predictions_step(context):
    evaluator = context['evaluator']
    y_test = context['y_test']
    print("\n\n--- Visualizing Actual vs Predicted Values ---")
    plot_actual_vs_predicted(y_test, evaluator.predictions)
    logging.info("Actual vs Predicted visualizations created successfully.")
    print("Actual vs Predicted visualizations created successfully.")
    return context


def retrain_and_serialize_model_step(context):
    df_finalized = context['df']
    print("\n\n--- Retraining and Serializing the Best Model ---")
    try:
        retrain_and_serialize_model(df_finalized)
        logging.info("Retraining and serialization of the best model completed successfully.")
        print("Retraining and serialization of the best model completed successfully.")
    except FileNotFoundError as e:
        logging.error(str(e))
        print(str(e))
    return context


def main():
    setup_logging()
    pipeline = Pipeline()

    # Add steps to the pipeline
    pipeline.add_step(load_and_display_dataset)
    pipeline.add_step(add_date_features_step)
    pipeline.add_step(plot_adj_close_distribution_step)
    pipeline.add_step(plot_adj_close_over_time_step)
    pipeline.add_step(plot_column_distributions_step)
    pipeline.add_step(plot_correlation_matrix_step)
    pipeline.add_step(plot_adj_close_decomposition_step)
    pipeline.add_step(remove_duplicates_step)
    pipeline.add_step(handle_missing_values_step)
    pipeline.add_step(ensure_stationarity_step)
    pipeline.add_step(multicollinearity_analysis_step)
    pipeline.add_step(create_lagged_features_step)
    pipeline.add_step(finalize_features_step)
    pipeline.add_step(save_last_rows_step)
    pipeline.add_step(split_data_step)
    pipeline.add_step(standardize_data_step)
    pipeline.add_step(implement_advanced_models_step)
    pipeline.add_step(hyperparameter_tuning_step)
    pipeline.add_step(evaluate_models_step)
    pipeline.add_step(visualize_model_comparison_step)
    pipeline.add_step(visualize_predictions_step)
    pipeline.add_step(retrain_and_serialize_model_step)

    logging.info("=== Starting Gold Price Prediction Data Pipeline ===")
    print("\n\n=== Starting Gold Price Prediction Data Pipeline ===")

    # Execute the pipeline
    context = pipeline.execute()

    logging.info("=== Data Pipeline Completed Successfully ===")
    print("\n\n=== Data Pipeline Completed Successfully ===\n")


if __name__ == "__main__":
    main()
