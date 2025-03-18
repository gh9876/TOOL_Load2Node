import os
import numpy as np
import pandas as pd
import re
import unicodedata
import fnmatch  # Used for pattern matching in filenames
import time
from datetime import date
from datetime import datetime
from config_tool_V1 import *
from distribution_functions import *
from logging_function import setup_stage_logger  # using logging_functions.py

# =============================================================================
# SETUP AND INITIALIZATION
# =============================================================================

# Record the start time of Stage 4
start_time = time.time()

# Get today's date formatted for filenames
today = datetime.today().strftime("%Y_%m_%d")  # For filenames

# Set up the logger for Stage 4
logger = setup_stage_logger("Stage4")

# Log header for Stage 4
logger.info("--------------------------------------------------------------------------------")
logger.info("--------------------------------------------------------------------------------")
logger.info("Start with Stage 4: Distribute created time series to LEGO nodes")
logger.info("")
logger.info("--------------------------------------------------------------------------------")

# =============================================================================
# FILE PATH GENERATION
# =============================================================================

def generate_file_paths(folder_path):
    """
    Generates file paths dynamically for each federal state by scanning the folder for relevant files.
    
    This function assumes that the filenames contain the federal state names in some form and have an '.xlsx' extension.
    
    Parameters:
    - folder_path (str): Path to the folder containing the files.
    
    Returns:
    - dict: A dictionary where keys are federal state names and values are the corresponding file paths.
    """
    # Define the federal states
    federal_states = ['Burgenland', 'Kärnten', 'Niederösterreich', 'Oberösterreich',
                      'Salzburg', 'Steiermark', 'Tirol', 'Vorarlberg', 'Wien']
    
    # Create the dictionary to hold the file paths
    file_paths = {}

    # List all files in the given folder
    all_files = os.listdir(folder_path)
    
    # Loop through the states and find matching files for each state
    for state in federal_states:
        # Look for a file that contains the federal state name and has an '.xlsx' extension
        matching_files = fnmatch.filter(all_files, f"*{state}*.xlsx")
        
        # If there's a matching file for this state, add it to the dictionary
        if matching_files:
            # Assume the first matching file is the correct one (adjust logic if needed)
            full_path = os.path.join(folder_path, matching_files[0])
            file_paths[state] = full_path
        else:
            logger.info(f"No matching file found for {state}")

    return file_paths

# Generate file paths from the designated folder
file_paths = generate_file_paths(folder_path)

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_federal_state_name(file_path):
    """
    Extracts the federal state name from the file path by parsing the filename.
    
    Parameters:
    - file_path (str): The full path to the file.
    
    Returns:
    - str: The extracted federal state name.
    """
    filename = os.path.basename(file_path)
    state_name = filename.split('_')[0]
    return state_name

def load_input_data(nuts_data_path, mapping_data_path):
    """
    Loads the NUTS data and LEGO mapping data from the provided Excel files,
    and computes the total sum of generated time series data.
    
    Parameters:
    - nuts_data_path (str): Path to the Excel file containing NUTS data.
    - mapping_data_path (str): Path to the Excel file containing LEGO mapping data.
    
    Returns:
    - tuple: (nuts_data (DataFrame), mapping_data (DataFrame), total_sum_generated_time_series (float))
    """
    # Load the data from Excel files
    nuts_data = pd.read_excel(nuts_data_path, index_col=None)
    mapping_data = pd.read_excel(mapping_data_path, index_col=None)

    # For this instance, assume numeric data starts from the second column
    numeric_columns = nuts_data.columns[1:]
    numeric_data = nuts_data[numeric_columns].apply(pd.to_numeric, errors='coerce')  # Convert to numeric, ignoring errors
    
    # Sum over the entire numeric DataFrame to get total generated time series
    total_sum_generated_time_series = numeric_data.sum().sum()

    # Return both datasets and the total sum
    return nuts_data, mapping_data, total_sum_generated_time_series

# =============================================================================
# MAIN PROCESSING FUNCTION FOR DISTRIBUTION
# =============================================================================

def process_all_federal_states(file_paths, output_lego2nuts, stage4_output_folder):
    """
    Processes all federal states by loading input data, executing state-specific distribution functions,
    and saving the results.
    
    Parameters:
    - file_paths (dict): Dictionary of file paths keyed by federal state.
    - output_lego2nuts (str): Path to the LEGO mapping data file.
    - stage4_output_folder (str): Output directory to save distribution results.
    
    Returns:
    - None
    """
    for state, file_path in file_paths.items():
        logger.info("--------------------------------------------------------------------------------")
        logger.info("--------------------------------------------------------------------------------")
        logger.info(f"Processing {state} from {file_path}")
        
        # Load the necessary data for the state
        nuts_data, mapping_data, total_demand_input = load_input_data(file_path, output_lego2nuts)
        nodes_with_multiple_nuts3 = find_and_split_nodes_with_multiple_nuts_levels(mapping_data)
        
        # Prepare filenames based on the federal state name
        federal_state_name = get_federal_state_name(file_path)
        filename_result_distribution = f'{federal_state_name}_distribution_result_nuts2lego_test_{today}.xlsx'
        
        # Call the appropriate distribution function based on the state
        if state == 'Niederösterreich':
             result_distribution = distribute_nuts_level_data_niederoesterreich(nuts_data, mapping_data, total_demand_input, federal_state_name, stage4_output_folder, filename_result_distribution)
        elif state == 'Burgenland':
             result_distribution = distribute_nuts_level_data_burgenland(nuts_data, mapping_data, total_demand_input, federal_state_name, stage4_output_folder, filename_result_distribution)
        elif state == 'Oberösterreich':
            result_distribution = distribute_nuts_level_data_oberoesterreich(nuts_data, mapping_data, total_demand_input, federal_state_name, stage4_output_folder, filename_result_distribution)
        elif state == 'Tirol':
            result_distribution = distribute_nuts_level_data_tirol(nuts_data, mapping_data, total_demand_input, federal_state_name, stage4_output_folder, filename_result_distribution)
        elif state == 'Kärnten':
            result_distribution = distribute_nuts_level_data_kaernten(nuts_data, mapping_data, total_demand_input, federal_state_name, stage4_output_folder, filename_result_distribution)
        elif state == 'Salzburg':
            result_distribution = distribute_nuts_level_data_salzburg(nuts_data, mapping_data, total_demand_input, federal_state_name, stage4_output_folder, filename_result_distribution)
        elif state == 'Steiermark':
            result_distribution = distribute_nuts_level_data_steiermark(nuts_data, mapping_data, total_demand_input, federal_state_name, stage4_output_folder, filename_result_distribution)
        elif state == 'Vorarlberg':
            result_distribution = distribute_nuts_level_data_vorarlberg(nuts_data, mapping_data, total_demand_input, federal_state_name, stage4_output_folder, filename_result_distribution)
        elif state == 'Wien':
            result_distribution = distribute_nuts_level_data_wien(nuts_data, mapping_data, total_demand_input, federal_state_name, stage4_output_folder, filename_result_distribution)

# Execute processing for all federal states
process_all_federal_states(file_paths, output_lego2nuts, stage4_output_folder)

# =============================================================================
# DATA COMBINATION AND FINAL FILE SAVING
# =============================================================================

def combine_data_and_save_files(folder_path, date_format):
    """
    Combines the distribution result files for all federal states and saves the combined DataFrame to an Excel file.
    
    Parameters:
    - folder_path (str): Path to the folder containing the distribution result files.
    - date_format (str): Date format string for validating the file dates.
    
    Returns:
    - None
    """
    logger.info("Start with combining latest federal state output")

    latest_files = {}
    today_str = datetime.today().strftime(date_format)  # Today's date in the required format, e.g., "2024_11_11"
    found_states = set()

    # List and normalize files in the folder
    files_in_folder = [unicodedata.normalize('NFC', file_name) for file_name in os.listdir(folder_path)]
    logger.info(f"Files in folder: {files_in_folder}")

    # Loop through files and select those matching the distribution result naming convention and today's date
    for file_name in files_in_folder:
        if file_name.endswith('.xlsx') and "distribution_result_nuts2lego" in file_name:
            # Extract state name and date from the filename
            match = re.match(r"([A-Za-zöäüÖÄÜß]+)_distribution_result_nuts2lego_test_(\d{4}_\d{2}_\d{2})", file_name)
            if match:
                state_name, file_date = match.groups()
                # Check if the file date matches today's date
                if file_date == today_str:
                    file_path = os.path.join(folder_path, file_name)
                    latest_files[state_name] = file_path
                    found_states.add(state_name)
                    logger.info(f"File selected for {state_name}: {file_name}")
                else:
                    logger.info(f"Skipped file due to date mismatch: {file_name}")

    # Combine data from the selected files
    combined_data = []
    for state, file_path in latest_files.items():
        df = pd.read_excel(file_path)
        time_series_columns = [f'k{str(i).zfill(4)}' for i in range(1, 8761)]
        required_columns = ['LEGO ID'] + time_series_columns

        if all(col in df.columns for col in required_columns):
            df_filtered = df[required_columns]
            combined_data.append(df_filtered)
        else:
            logger.warning(f"Warning: File {file_path} does not have the expected structure.")

    # Check for missing states
    all_states = {'Tirol', 'Vorarlberg', 'Wien', 'Burgenland', 'Salzburg', 'Steiermark',
                  'Niederösterreich', 'Oberösterreich', 'Kärnten'}
    missing_states = all_states - found_states
    if missing_states:
        logger.warning(f"Warning: Missing files for states: {', '.join(missing_states)}")

    # Save combined data if available
    if combined_data:
        combined_df = pd.concat(combined_data, ignore_index=True).drop_duplicates(subset=['LEGO ID'], keep='last')
        output_path = os.path.join(folder_path, f'AT_LEGO_nodes_time_series_{today_str}.xlsx')
        combined_df.to_excel(output_path, index=False)
        logger.info(f"Combined data saved to: {output_path}")
    else:
        logger.info("No valid files found or no data to combine.")

# Execute data combination and saving process
result_nodes_df = combine_data_and_save_files(stage4_output_folder, date_format="%Y_%m_%d")

# =============================================================================
# FINALIZATION
# =============================================================================

# Calculate and log the total runtime of Stage 4
end_time = time.time()
elapsed_time = end_time - start_time
hours, rem = divmod(elapsed_time, 3600)
minutes, seconds = divmod(rem, 60)

logger.info(f"Finished - Total runtime script: {int(hours)}h {int(minutes)}m {int(seconds)}s")
logger.info("")
logger.info("--------------------------------------------------------------------------------")
