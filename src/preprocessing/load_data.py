# -----------------------------------------------------------------------------
# Submitted by:     u3258928
# Date:             2024 November 02
# Course:           Software Technology 1
# Activity:         Assessment 3: Programming project
# -----------------------------------------------------------------------------


import os
import pandas as pd
import kagglehub
import shutil

DATASET_NAME = "sid321axn/gold-price-prediction-dataset"
DATASET_FILENAME = "FINAL_USO.csv"
DATASET_PATH = os.path.join("data", "input", "raw", DATASET_FILENAME)


def download_dataset():
    """
    Downloads the dataset from Kaggle if it does not already exist in the data directory.
    """
    if not os.path.exists(DATASET_PATH):
        path = kagglehub.dataset_download(DATASET_NAME)
        extracted_file_path = find_file_in_directory(path, DATASET_FILENAME)
        if extracted_file_path:
            os.makedirs(os.path.dirname(DATASET_PATH), exist_ok=True)
            shutil.copy(extracted_file_path, DATASET_PATH)
        else:
            raise FileNotFoundError(f"{DATASET_FILENAME} not found in the downloaded dataset.")


def find_file_in_directory(directory, filename):
    """
    Recursively searches for a file within a directory.

    Args:
        directory (str): The root directory to search.
        filename (str): The name of the file to find.

    Returns:
        str or None: The full path to the file if found, else None.
    """
    for root, _, files in os.walk(directory):
        if filename in files:
            return os.path.join(root, filename)
    return None


def load_dataset():
    """
    Loads the dataset, parses the Date column, sets it as index, and sorts by date.

    Returns:
        pd.DataFrame: The preprocessed dataset.
    """
    download_dataset()
    df = pd.read_csv(DATASET_PATH)
    df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')
    df.set_index('Date', inplace=True)
    df.sort_index(inplace=True)
    return df
