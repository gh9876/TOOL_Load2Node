"""
Module: process_input_data.py
Description:
    This module processes input data for LEGO demand and network information.
    It reads and processes data from various Excel files, assigns NUTS levels using
    the OpenCage API, validates and reassigns incorrect NUTS codes if needed, and 
    finally saves the processed output as an Excel file.
"""

import pandas as pd
from opencage.geocoder import OpenCageGeocode
import os
import time
from datetime import datetime
from config_tool_V1 import *

# =============================================================================
# INITIAL PROCESSING OF INPUT DATA
# =============================================================================

print("Start processing input data: LEGO demand and Network information")

def process_data(input_lego, input_demand_file, input_at_nuts):
    """
    Processes input data from multiple sources and returns processed DataFrames.

    This function processes:
    - LEGO network information (from an Excel file with bus info)
    - NUTS information (filtered to include only 5-digit NUTS codes)
    - Demand and distribution data (including population info)

    After processing, it merges the network and demand data to extract node information.

    Parameters:
        input_lego (str): File path for LEGO network information.
        input_demand_file (str): File path for demand and distribution data.
        input_at_nuts (str): File path for NUTS information.

    Returns:
        tuple: (df_lego, df_nuts3, df_population_info, df_node_population)
            df_lego: Processed LEGO network information.
            df_nuts3: Processed NUTS information (5-digit codes).
            df_population_info: Processed demand/distribution data.
            df_node_population: Subset of node information where population is flagged.
    """
    # -------------------------------------------------------------------------
    # Function to process LEGO network information
    def process_lego_network_info(file_path):
        df_lego_network_raw = pd.read_excel(file_path, sheet_name='BusInfo', header=4)
        df_lego_network_processed = df_lego_network_raw.copy()
        df_lego_network_processed.rename(columns={'Unnamed: 2': 'LEGO ID'}, inplace=True)
        df_lego_filtered = df_lego_network_processed[['LEGO ID', 'BaseVolt', 'Name', 'long', 'lat']].copy()
        # Convert lat and lon to numeric values (handle errors)
        df_lego_filtered.loc[:, 'lat'] = pd.to_numeric(df_lego_filtered['lat'], errors='coerce')
        df_lego_filtered.loc[:, 'lon'] = pd.to_numeric(df_lego_filtered['long'], errors='coerce')
        return df_lego_filtered

    # -------------------------------------------------------------------------
    # Function to process NUTS information
    def process_nuts_info(file_path):
        nuts3_df = pd.read_excel(file_path)
        # Filter to include only 5-digit NUTS codes
        nuts3_df = nuts3_df[nuts3_df['NUTS Code'].str.len() == 5]
        return nuts3_df

    # -------------------------------------------------------------------------
    # Function to process demand and distribution data
    def process_distribution_data(file_path):
        df = pd.read_excel(file_path, sheet_name='Distribution', skiprows=4)
        df.rename(columns={'Unnamed: 2': 'LEGO ID'}, inplace=True)
        df['Population'] = pd.to_numeric(df['Population'], errors='coerce')
        # Flag rows with valid population data
        df['Population_Flag'] = df['Population'].apply(lambda x: 1 if pd.notnull(x) and x > 0 else 0)
        return df

    # Process each input file
    df_lego = process_lego_network_info(input_lego)
    df_nuts3 = process_nuts_info(input_at_nuts)
    df_population_info = process_distribution_data(input_demand_file)

    # Merge network and demand data on LEGO ID
    df_merge_network_demand_info = df_lego.merge(df_population_info, right_on='LEGO ID', left_on='LEGO ID')

    # Extract node information and select nodes with valid population
    df_node_info = df_merge_network_demand_info[['LEGO ID', 'BaseVolt', 'Name', 'lon', 'lat', 'Population_Flag']]
    df_node_population = df_node_info[df_node_info['Population_Flag'] == 1]

    return df_lego, df_nuts3, df_population_info, df_node_population

# Process input data using provided file paths
df_lego, df_nuts3, df_population_info, df_node_population = process_data(input_lego, input_demand_file, input_at_nuts)
print("Finished with processing input data")
print()

# =============================================================================
# ASSIGNING NUTS LEVELS VIA OPENCAGE API
# =============================================================================

def assign_nuts_with_opencage(input_dataframe, api_key):
    """
    Assigns NUTS1, NUTS2, and NUTS3 levels to nodes using the OpenCage API.

    This function initializes an OpenCage geocoder and, for each node in the input
    DataFrame, performs a reverse geocoding query using its latitude and longitude.
    It then extracts the NUTS codes from the response and updates the DataFrame.

    Parameters:
        input_dataframe (DataFrame): DataFrame containing node information with 'lat' and 'lon' columns.
        api_key (str): OpenCage API key.

    Returns:
        DataFrame: A copy of the input DataFrame updated with columns for NUTS1, NUTS2, NUTS3, and a status flag.
    """
    df = input_dataframe.copy()  # Avoid view issues by making a full copy

    # Initialize the geocoder with the provided API key
    geocoder = OpenCageGeocode(api_key)

    print("Start fetching NUTS levels")
    
    def get_nuts_from_opencage(latitude, longitude):
        """
        Performs a reverse geocoding query to obtain NUTS level data.

        Parameters:
            latitude (float): Latitude of the location.
            longitude (float): Longitude of the location.

        Returns:
            tuple: (NUTS3, NUTS2, NUTS1) codes if available; otherwise, (None, None, None).
        """
        result = geocoder.reverse_geocode(latitude, longitude)
        if result:
            if 'annotations' in result[0] and 'NUTS' in result[0]['annotations']:
                nuts_data = result[0]['annotations']['NUTS']
                return nuts_data.get('NUTS3'), nuts_data.get('NUTS2'), nuts_data.get('NUTS1')
        return None, None, None

    # Initialize lists to store NUTS codes and status for each node
    nuts3_codes = []
    nuts2_codes = []
    nuts1_codes = []
    nuts_statuses = []
    
    for index, row in df.iterrows():
        try:
            latitude = row['lat']
            longitude = row['lon']
            nuts3_code, nuts2_code, nuts1_code = get_nuts_from_opencage(latitude, longitude)
            if nuts3_code and nuts2_code and nuts1_code:
                nuts3_codes.append(nuts3_code)
                nuts2_codes.append(nuts2_code)
                nuts1_codes.append(nuts1_code)
                nuts_statuses.append('Success')
            else:
                nuts3_codes.append(None)
                nuts2_codes.append(None)
                nuts1_codes.append(None)
                nuts_statuses.append('Missing NUTS Data')
        except Exception as e:
            nuts3_codes.append(None)
            nuts2_codes.append(None)
            nuts1_codes.append(None)
            nuts_statuses.append('Error')

    # Update the DataFrame with the retrieved NUTS codes and status
    df.loc[:, 'NUTS3'] = nuts3_codes
    df.loc[:, 'NUTS2'] = nuts2_codes
    df.loc[:, 'NUTS1'] = nuts1_codes
    df.loc[:, 'NUTS_Status'] = nuts_statuses

    # Extract the 'code' value from the returned dictionaries if applicable
    df['NUTS1'] = df['NUTS1'].apply(lambda x: x['code'] if isinstance(x, dict) else x)
    df['NUTS2'] = df['NUTS2'].apply(lambda x: x['code'] if isinstance(x, dict) else x)
    df['NUTS3'] = df['NUTS3'].apply(lambda x: x['code'] if isinstance(x, dict) else x)
    print("Fetching NUTS level assignment done!")
    print()
    return df

def validate_nuts_country(df_with_nuts3, expected_country_code):
    """
    Validates that all NUTS3 levels in the DataFrame belong to the expected country.

    The function checks whether the 'NUTS3' code for each node starts with the provided
    country code. It prints and returns rows where the NUTS3 code does not match.

    Parameters:
        df_with_nuts3 (DataFrame): DataFrame containing NUTS3 level assignments.
        expected_country_code (str): Expected two-character country code (e.g., 'AT').

    Returns:
        DataFrame: A DataFrame of rows with invalid NUTS3 codes, or an empty DataFrame if all are valid.
    """
    print(f"Validating NUTS levels belong to the country: {expected_country_code}")
    invalid_nuts3 = df_with_nuts3[~df_with_nuts3['NUTS3'].str.startswith(expected_country_code, na=False)]
    if not invalid_nuts3.empty:
        print(f"Found {len(invalid_nuts3)} NUTS3 levels that are assigned to the wrong country.")
        print(invalid_nuts3[['LEGO ID', 'NUTS3']])
        return invalid_nuts3
    else:
        print("All NUTS3 levels belong to the correct country.")
        return pd.DataFrame()

def assign_unassigned_nuts3(df_with_nuts3, unassigned_nuts3):
    """
    Prompts the user to manually assign valid counterparts for unassigned NUTS3 levels.

    For each unassigned NUTS3 level, the function identifies valid counterparts based on a
    common prefix and prompts the user to assign one. The assignments are then merged into the DataFrame.

    Parameters:
        df_with_nuts3 (DataFrame): DataFrame containing NUTS3 assignments.
        unassigned_nuts3 (set): Set of unassigned NUTS3 levels.

    Returns:
        tuple: (updated DataFrame, list of unassigned NUTS3 levels still unassigned)
    """
    df_with_nuts3_copy = df_with_nuts3.copy()
    manual_assignment = {}
    unassigned_nuts3_list = list(unassigned_nuts3)
    
    for unassigned in unassigned_nuts3_list.copy():
        print(f"Unassigned NUTS3 level: {unassigned}")
        unassigned_prefix = unassigned[:4]  # Assumes NUTS3 level format e.g., 'AT11X'
        valid_counterparts = df_with_nuts3_copy[df_with_nuts3_copy['NUTS3'].str.startswith(unassigned_prefix)]['NUTS3'].dropna().unique()
        if len(valid_counterparts) == 0:
            print(f"No valid counterparts found for {unassigned}. Skipping.")
            continue
        while True:
            counterpart = input(f"Please assign a valid counterpart NUTS3 level for {unassigned} (Valid options: {', '.join(valid_counterparts)}): ")
            if counterpart in valid_counterparts:
                manual_assignment[unassigned] = counterpart
                print(f"Unassigned NUTS3 level {unassigned} has been assigned to {counterpart}.")
                unassigned_nuts3_list.remove(unassigned)
                break
            else:
                print(f"Invalid input! Please choose a valid NUTS3 level from the list: {', '.join(valid_counterparts)}")
    for unassigned, counterpart in manual_assignment.items():
        mask = df_with_nuts3_copy['NUTS3'] == counterpart
        df_with_nuts3_copy.loc[mask, 'NUTS3'] = df_with_nuts3_copy.loc[mask, 'NUTS3'] + ', ' + unassigned
    return df_with_nuts3_copy, unassigned_nuts3_list

def reassign_invalid_nuts_levels(df_with_nuts3, invalid_nuts3):
    """
    Reassigns incorrect NUTS3 levels based on user input.

    For each row with an invalid NUTS3 code, the function prompts the user to enter a
    correct 3-digit NUTS code that starts with the expected country code. The DataFrame
    is then updated with the new assignment.

    Parameters:
        df_with_nuts3 (DataFrame): DataFrame containing the invalid NUTS3 entries.
        invalid_nuts3 (DataFrame): DataFrame of rows with invalid NUTS3 codes.

    Returns:
        DataFrame: Updated DataFrame with corrected NUTS3 levels.
    """
    df_with_nuts3_copy = df_with_nuts3.copy()
    manual_assignment = {}
    for index, row in invalid_nuts3.iterrows():
        unassigned = row['NUTS3']
        print(f"NUTS3 level {unassigned} is assigned to the wrong country.")
        while True:
            counterpart = input(f"Please assign a valid NUTS3 level for {unassigned} (Enter a 3-digit NUTS code starting with 'AT'): ")
            if counterpart.startswith('AT') and len(counterpart) == 5 and counterpart[2:].isdigit():
                manual_assignment[unassigned] = counterpart
                print(f"NUTS3 level {unassigned} has been reassigned to {counterpart}.")
                break
            else:
                print("Invalid input! Please enter a valid 3-digit NUTS level starting with 'AT'.")
    for unassigned, counterpart in manual_assignment.items():
        mask = df_with_nuts3_copy['NUTS3'] == unassigned
        df_with_nuts3_copy.loc[mask, 'NUTS3'] = counterpart
        print(f"Reassignment: {unassigned} -> {counterpart}")
    return df_with_nuts3_copy

def find_unassigned_nuts3_levels(lego_information, nuts_information):
    """
    Finds NUTS3 levels that are present in the nuts information but not assigned to any LEGO node.

    Parameters:
        lego_information (DataFrame): DataFrame containing LEGO node NUTS3 assignments.
        nuts_information (DataFrame): DataFrame containing all valid NUTS codes.

    Returns:
        set: A set of unassigned NUTS3 levels.
    """
    lego_information['NUTS3'] = lego_information['NUTS3'].apply(lambda x: x if isinstance(x, str) else str(x))
    all_nuts3_levels = set(nuts_information['NUTS Code'].unique())
    assigned_nuts3_levels = set(lego_information['NUTS3'].dropna().unique())
    unassigned_nuts3_levels = all_nuts3_levels - assigned_nuts3_levels
    filtered_unassigned_nuts3 = {nuts3 for nuts3 in unassigned_nuts3_levels if nuts3[2:].isdigit()}
    return filtered_unassigned_nuts3

# =============================================================================
# FULL PROCESS TO ASSIGN NUTS LEVELS
# =============================================================================

def process_and_assign_nuts_levels(df_node_population, df_nuts3, api_key, expected_country_code):
    """
    Executes the full process to assign NUTS3 levels to nodes and ensure that all nodes have valid assignments.

    The process includes:
      1. Assigning NUTS levels using the OpenCage API.
      2. Validating that the assigned NUTS3 levels belong to the expected country.
      3. Reassigning incorrect NUTS3 levels.
      4. Finding and manually assigning unassigned NUTS3 levels.

    Parameters:
        df_node_population (DataFrame): DataFrame containing node population info.
        df_nuts3 (DataFrame): DataFrame containing valid NUTS codes.
        api_key (str): OpenCage API key.
        expected_country_code (str): Expected country code prefix (e.g., 'AT').

    Returns:
        DataFrame: Updated DataFrame with valid NUTS3 level assignments.
    """
    df_with_nuts = assign_nuts_with_opencage(df_node_population, api_key)
    invalid_nuts3 = validate_nuts_country(df_with_nuts, expected_country_code)
    if not invalid_nuts3.empty:
        df_with_nuts = reassign_invalid_nuts_levels(df_with_nuts, invalid_nuts3)
    unassigned_nuts3 = find_unassigned_nuts3_levels(df_with_nuts, df_nuts3)
    if unassigned_nuts3:
        df_with_nuts, unassigned_nuts3_list = assign_unassigned_nuts3(df_with_nuts, unassigned_nuts3)
    print()
    print("NUTS3 level assignment done")
    print()
    return df_with_nuts

# Run full NUTS level assignment process
df_with_nuts_updated = process_and_assign_nuts_levels(df_node_population, df_nuts3, API_KEY, 'AT')

# =============================================================================
# SAVE OUTPUT DATAFRAME AS EXCEL FILE
# =============================================================================

def save_dataframe_as_excel(df, input_file_path, filename):
    """
    Saves a DataFrame as an Excel file to the specified directory.

    Parameters:
        df (DataFrame): The DataFrame to save.
        input_file_path (str): Directory in which to save the file.
        filename (str): Desired filename (without extension).

    Returns:
        None
    """
    file_path = os.path.join(input_file_path, filename)
    if not file_path.endswith(".xlsx"):
        file_path += ".xlsx"
    df.to_excel(file_path, index=False)
    print(f"DataFrame saved as Excel file at: {file_path}")
    print()

# Save the final processed DataFrame
save_dataframe_as_excel(df_with_nuts_updated, folder_path_output, output_lego2nuts)
