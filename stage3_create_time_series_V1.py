import os
import sys
import fnmatch  # Used for pattern matching in filenames
from datetime import datetime
import time
from create_time_series_V1 import select_synthload_profile, create_hourly_profiles, process_federal_state
from config_tool_V1 import *
import subprocess
from logging_function import setup_stage_logger  # Note: using logging_functions.py

# =============================================================================
# SETUP AND INITIALIZATION
# =============================================================================

# Record the start time of the script
start_time = time.time()

# Get today's date formatted for filenames
today = datetime.today().strftime("%Y_%m_%d")

# Set up the logger for Stage 3
logger = setup_stage_logger("Stage3")

# Log header to indicate the beginning of Stage 3 processing
logger.info("--------------------------------------------------------------------------------")
logger.info("--------------------------------------------------------------------------------")
logger.info("Start with Stage 3: Creation of time series based on NUTS3 level data")
logger.info("")
logger.info("--------------------------------------------------------------------------------")

# =============================================================================
# SELECT TIME SERIES PROFILE FILE
# =============================================================================

# Select the correct synthetic load profile file based on the year_of_interest and configuration settings.
file_path_profiles = select_synthload_profile(folder_path_profiles, year_of_interest, use_excel, excel_profiles_path)

logger.info("Start with Stage 3: Creating time series based on the distributed demand")

# =============================================================================
# FUNCTION DEFINITIONS
# =============================================================================

def generate_file_paths(folder_path):
    """
    Generates file paths dynamically for each federal state by scanning the folder for relevant Excel files.
    
    This function assumes that the filenames contain the federal state names.
    
    Parameters:
    - folder_path (str): Path to the folder containing Excel files.
    
    Returns:
    - dict: Dictionary where keys are federal state names and values are the corresponding file paths.
    """
    # Define the list of federal states
    federal_states = ['Burgenland', 'Kärnten', 'Niederösterreich', 'Oberösterreich',
                      'Salzburg', 'Steiermark', 'Tirol', 'Vorarlberg', 'Wien']
    
    # Dictionary to hold file paths for each state
    file_paths = {}

    # List all files in the given folder
    all_files = os.listdir(folder_path)
    
    # Loop through the states and find matching Excel files
    for state in federal_states:
        # Look for a file containing the federal state name with an '.xlsx' extension
        matching_files = fnmatch.filter(all_files, f"*{state}*.xlsx")
        
        # If a matching file is found, add the first match to the dictionary
        if matching_files:
            full_path = os.path.join(folder_path, matching_files[0])
            file_paths[state] = full_path
        else:
            logger.info(f"No matching file found for {state}")

    return file_paths

# Create hourly profiles and time series mapping (run once)
# The function returns a profile mapping and a time series dictionary
profile_mapping, time_series_dict = create_hourly_profiles(file_path_profiles, year_of_interest, use_excel)

def process_all_federal_states(folder_path):
    """
    Processes time series creation for all federal states by dynamically generating file paths,
    and then processing each state's data.
    
    Parameters:
    - folder_path (str): Path to the folder containing the federal state Excel files.
    
    Returns:
    - None
    """
    # Generate file paths for each federal state
    file_paths = generate_file_paths(folder_path)
    
    # Loop through each file and process the corresponding federal state data
    for federal_state, file_path_load in file_paths.items():
        logger.info(f"Processing federal state data for {federal_state}: {file_path_load}")
        # Process the federal state data using precomputed profiles and time series mapping.
        process_federal_state(
            file_path_load,
            file_path_mapping,
            output_directory_time_series,
            output_directory_aggregated_data,
            profile_mapping,
            time_series_dict,
            year_of_interest,
            use_weights,
            weight_file_path,
            use_excel,
            excel_profiles_path
        )

# =============================================================================
# EXECUTION OF TIME SERIES PROCESSING
# =============================================================================

# Process all federal states based on the input folder path
process_all_federal_states(input_folder_path)

# =============================================================================
# FINALIZATION AND OPTIONAL POST-PROCESSING
# =============================================================================

# Calculate elapsed runtime and convert it to hours, minutes, and seconds
end_time = time.time()
elapsed_time = end_time - start_time
hours, rem = divmod(elapsed_time, 3600)
minutes, seconds = divmod(rem, 60)

# Optionally run the LEGO2NUTS assignment if configured
if use_nuts_assignment:
    logger.info("Start: running Lego to NUTS assignment script")
    subprocess.run([sys.executable, "lego2nuts_assignment_V1.py"])
    logger.info("Finished with LEGO2NUTS assignment")
else:
    logger.info("LEGO2NUTS assignment is not executed")
logger.info("")

# Log final status and runtime
logger.info("Finished with creation of time series")
logger.info("")
logger.info(f"Finished with Stage 3 - Total runtime script: {int(hours)}h {int(minutes)}m {int(seconds)}s")
logger.info("")
