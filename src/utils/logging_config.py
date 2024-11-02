# -----------------------------------------------------------------------------
# Submitted by:     u3258928
# Date:             2024 November 02
# Course:           Software Technology 1
# Activity:         Assessment 3: Programming project
# -----------------------------------------------------------------------------


import os
import logging

def setup_logging(log_dir='logs', log_file='app.log'):
    """
    Configures logging settings and ensures the log directory exists.

    Args:
        log_dir (str): The directory for log files.
        log_file (str): The log file name.
    """
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(log_dir, log_file),
        filemode='a',
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )
