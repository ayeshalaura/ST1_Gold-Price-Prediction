# -----------------------------------------------------------------------------
# Submitted by:     u3258928
# Date:             2024 November 02
# Course:           Software Technology 1
# Activity:         Assessment 3: Programming project
# -----------------------------------------------------------------------------


import pandas as pd
import os
import json


def save_last_rows(df: pd.DataFrame, num_rows: int = 100, output_dir: str = "data/input/demo", filename: str = "last_rows.json") -> None:
    """
    Extracts the last `num_rows` of the DataFrame and saves them as a JSON file for future use in a web application.

    Args:
        df (pd.DataFrame): The input DataFrame.
        num_rows (int): Number of rows to extract.
        output_dir (str): Directory to save the JSON file.
        filename (str): Name of the JSON file.
    """
    os.makedirs(output_dir, exist_ok=True)
    last_rows = df.tail(num_rows).to_dict(orient='records')
    with open(os.path.join(output_dir, filename), 'w') as f:
        json.dump(last_rows, f, indent=4)
