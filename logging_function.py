# logging_setup.py
"""
Module: logging_setup.py
Description:
    This module provides a function to set up a logger for a specific stage.
    The logger writes log messages to both a file (stored in a designated log folder)
    and the console. The log file is named based on the stage name and the current date.
"""

import os
import logging
from datetime import datetime
from config_tool_V1 import folder_log  # Import folder_log from your configuration settings

def setup_stage_logger(stage_name):
    """
    Sets up a logger for a specific stage, storing logs in a designated log folder.

    The logger writes output to a log file named with the stage name and the current date,
    and it also outputs log messages to the console. This function ensures that the log
    directory exists before creating the file handler.

    Parameters:
        stage_name (str): Name of the stage for log identification.

    Returns:
        logging.Logger: Configured logger for the specified stage.
    """
    # Ensure the log directory exists; create it if necessary.
    os.makedirs(folder_log, exist_ok=True)
    
    # Define the log file path using the stage name and today's date.
    date_str = datetime.today().strftime('%Y%m%d')
    log_file_path = os.path.join(folder_log, f"{stage_name}_execution_log_{date_str}.txt")

    # Create or retrieve a logger for the given stage.
    logger = logging.getLogger(stage_name)
    logger.setLevel(logging.INFO)
    
    # Prevent duplicate handlers if the logger is already set up.
    if not logger.hasHandlers():
        # Create a file handler to write log messages to the log file.
        file_handler = logging.FileHandler(log_file_path)
        file_handler.setLevel(logging.INFO)

        # Create a stream handler to output log messages to the console.
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.INFO)

        # Define a simple formatter for both handlers.
        simple_formatter = logging.Formatter('%(message)s')
        file_handler.setFormatter(simple_formatter)
        stream_handler.setFormatter(simple_formatter)

        # Add both handlers to the logger.
        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)
    
    return logger
