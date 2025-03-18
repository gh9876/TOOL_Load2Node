import csv
import os
from collections import defaultdict
import pandas as pd
import calendar
from config_tool_V1 import *
import numpy as np
from logging_function import setup_stage_logger  # Note the file name is logging_function.py

# =============================================================================
# OVERALL PURPOSE:
# This module processes synthetic load profiles by selecting the correct load 
# profile file based on a specified year, generating hourly profiles (from CSV 
# or Excel sources), loading sector weights, and finally processing each 
# federal state's data to produce hourly demand and aggregated time series.
# =============================================================================

# =============================================================================
# GLOBAL VARIABLES
# =============================================================================

# Set to keep track of processed states to avoid duplicate processing.
processed_states = set()

# Set up a logger for this module
logger = setup_stage_logger("TimeSeriesCreation")

# =============================================================================
# FUNCTION: select_synthload_profile
# =============================================================================
def select_synthload_profile(folder_path, year_of_interest, use_excel, excel_profiles_path=None):
    """
    Selects the appropriate synthetic load profile file based on the year of interest 
    and whether to use Excel or CSV files.

    Parameters:
      - folder_path (str): Directory containing CSV files.
      - year_of_interest (int): The target year for which the load profile is needed.
      - use_excel (bool): Flag indicating whether to use an Excel file.
      - excel_profiles_path (str, optional): Path to the Excel file (if use_excel is True).

    Returns:
      - str: The file path of the selected load profile.

    Raises:
      - FileNotFoundError: If the specified Excel file or a matching CSV file cannot be found.
    """
    # Ensure the year is within the available data range.
    if year_of_interest < 2018:
        logger.warning(f"Year of interest {year_of_interest} is earlier than available data (2018). Using 2018 instead.")
        year_of_interest = 2018
    elif year_of_interest > 2029:
        logger.warning(f"Year of interest {year_of_interest} is later than available data (2029). Using 2029 instead.")
        year_of_interest = 2029

    # Convert year_of_interest to string for filename matching.
    year_of_interest = str(year_of_interest)

    if use_excel:
        if excel_profiles_path and os.path.exists(excel_profiles_path):
            logger.info(f"Using Excel file: {excel_profiles_path}")
            return excel_profiles_path
        else:
            raise FileNotFoundError(f"Excel file not found at: {excel_profiles_path}")

    # Search for a CSV file that contains the year_of_interest in its filename.
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.csv') and year_of_interest in file_name:
            file_path = os.path.join(folder_path, file_name)
            logger.info(f"Selected CSV profile for year {year_of_interest}: {file_name}")
            return file_path
    raise FileNotFoundError(f"No load profile found for the year {year_of_interest}")

# =============================================================================
# FUNCTION: create_hourly_profiles
# =============================================================================
def create_hourly_profiles(file_path_profiles, year_of_interest, use_excel):
    """
    Creates hourly load profiles from the specified file by processing and resampling data.

    Depending on the use_excel flag, it will either process CSV data or use an alternative 
    Excel-based approach.

    Parameters:
      - file_path_profiles (str): Path to the load profile file.
      - year_of_interest (int or str): The target year for processing.
      - use_excel (bool): Flag indicating whether to use Excel processing.

    Returns:
      - tuple: (profile_mapping, time_series_dict) where:
          profile_mapping (dict): Mapping from profile ID to generated variable name.
          time_series_dict (dict): Dictionary of time series DataFrames keyed by variable names.
    """

    # ---------------------------
    # Inner function for CSV processing: read_and_organize_csv
    # ---------------------------
    def read_and_organize_csv(file_path):
        """
        Reads and organizes CSV data into a dictionary with the profile type as key.

        Parameters:
          - file_path (str): Path to the CSV file.

        Returns:
          - defaultdict: Dictionary with keys as profile type (Typnummer) and values as lists 
                        of (time, value) tuples.
        """
        data = defaultdict(list)
        with open(file_path, mode='r', encoding='utf-8-sig') as file:
            first_line = file.readline().strip()
            delimiter = ';' if ';' in first_line else ','
            file.seek(0)
            csv_reader = csv.reader(file, delimiter=delimiter)
            headers = next(csv_reader)
            logger.info(f"Headers: {headers}")
            for row in csv_reader:
                try:
                    typnummer = int(row[0])
                    zeit = row[1]
                    wert = float(row[2].replace(',', '.'))
                    data[typnummer].append((zeit, wert))
                except IndexError as e:
                    logger.warning(f"IndexError: {e}. Row data: {row}")
                    continue
            logger.info("Created synthetic load profiles")
        # Convert the organized data to a DataFrame structure.
        data_list = []
        for typnummer, values in data.items():
            for zeit, wert in values:
                data_list.append({'Typnummer': typnummer, 'Zeit': zeit, 'Wert': wert})
        df = pd.DataFrame(data_list)
        # Convert 'Zeit' column to datetime and set appropriate indices.
        df['Zeit'] = pd.to_datetime(df['Zeit'], dayfirst=True)
        df.set_index(['Typnummer', 'Zeit'], inplace=True)
        # Resample the data to hourly values.
        hourly_profiles = df.groupby('Typnummer').resample('h', level='Zeit').mean().reset_index()
        logger.info(f"Number of rows after resampling: {len(hourly_profiles)}")
        # Reorganize the resampled data into a dictionary.
        organized_data = defaultdict(list)
        for _, row in hourly_profiles.iterrows():
            typnummer = row['Typnummer']
            zeit = row['Zeit']
            wert = row['Wert']
            organized_data[typnummer].append((zeit, wert))
        return organized_data

    # ---------------------------
    # Inner function for CSV processing: process_and_resample_csv
    # ---------------------------
    def process_and_resample_csv(file_path_profiles, year_of_interest):
        """
        Processes and resamples CSV load profile data to create hourly profiles.

        Steps:
          1. Reads and organizes CSV data.
          2. Converts organized data to a DataFrame.
          3. Resamples the data to hourly values.
          4. Removes February 29th entries for leap years.
          5. Creates a time series dictionary and profile mapping.

        Parameters:
          - file_path_profiles (str): Path to the CSV file.
          - year_of_interest (int or str): The target year.

        Returns:
          - tuple: (profile_mapping, time_series_dict)
        """
        logger.info("Reading and organizing CSV data...")
        organized_data = read_and_organize_csv(file_path_profiles)

        logger.info("Converting organized data to DataFrame...")
        data_list = []
        for typnummer, values in organized_data.items():
            for zeit, wert in values:
                data_list.append({'Typnummer': typnummer, 'Zeit': zeit, 'Wert': wert})
        df = pd.DataFrame(data_list)
        logger.info(f"DataFrame created with shape: {df.shape}")

        df['Zeit'] = pd.to_datetime(df['Zeit'])
        logger.info("Converted 'Zeit' column to datetime.")
        df.set_index(['Typnummer', 'Zeit'], inplace=True)
        logger.info("Set 'Typnummer' and 'Zeit' as index.")

        hourly_profiles = df.groupby('Typnummer').resample('h', level='Zeit').mean().reset_index()
        logger.info(f"Resampled data to hourly values. Number of rows after resampling: {len(hourly_profiles)}")

        # Identify constant profiles
        constant_profiles = {}
        for typnummer, df_group in hourly_profiles.groupby('Typnummer'):
            if df_group['Wert'].nunique() == 1:
                constant_profiles[typnummer] = df_group['Wert'].iloc[0]
                logger.warning(f"Detected constant profile for Typnummer {typnummer} with value {constant_profiles[typnummer]}")

        # Remove February 29th for leap years
        is_leap_year = calendar.isleap(int(year_of_interest))
        if is_leap_year:
            logger.info(f"Leap year detected for {year_of_interest}. Removing February 29th data after resampling.")
            feb_29_mask = (hourly_profiles['Zeit'].dt.month == 2) & (hourly_profiles['Zeit'].dt.day == 29)
            feb_29_entries = hourly_profiles[feb_29_mask]
            logger.info(f"Number of February 29th entries to be removed after resampling: {len(feb_29_entries)}")
            hourly_profiles = hourly_profiles[~feb_29_mask]
            logger.info(f"Removed February 29th data after resampling. Remaining rows: {len(hourly_profiles)}")
        else:
            logger.info(f"No leap year detected for {year_of_interest}. No data removed.")

        # Create time series dictionary and profile mapping.
        time_series_dict = {}
        profile_mapping = {}
        logger.info("Creating time series dictionary and profile mapping...")
        for typnummer, df_group in hourly_profiles.groupby('Typnummer'):
            variable_name = f"hourly_profile_{typnummer}"
            if typnummer in constant_profiles:
                constant_value = constant_profiles[typnummer]
                df_group['Wert'] = constant_value  # Assign constant value to the time series
                logger.info(f"Constant profile for Typnummer {typnummer} using constant value {constant_value}")
            time_series_dict[variable_name] = df_group
            profile_mapping[str(typnummer)] = variable_name
            logger.info(f"Created Load Profile '{variable_name}' with {len(df_group)} rows.")
        logger.info("Finished processing CSV data.\n")
        return profile_mapping, time_series_dict

    # ---------------------------
    # Inner function for Excel processing: use_alternative_load_profiles
    # ---------------------------
    def use_alternative_load_profiles(file_path_profiles):
        """
        Processes load profiles from an Excel file by reading mapping and load profile sheets.

        Parameters:
          - file_path_profiles (str): Path to the Excel file.

        Returns:
          - tuple: (profile_mapping, time_series_dict)
        """
        excel_data = pd.ExcelFile(file_path_profiles)
        df_mapping = pd.read_excel(excel_data, sheet_name='Mapping')
        df_time_series = pd.read_excel(excel_data, sheet_name="LoadProfiles")

        df_time_series['time'] = pd.to_datetime(df_time_series['time'])
        profile_mapping = {}
        time_series_dict = {}

        for _, row in df_mapping.iterrows():
            sector = row['Sektor']
            lastprofil = row['Lastprofil']
            variable_name = f"hourly_profile_{lastprofil}"
            if lastprofil in df_time_series.columns:
                logger.info(f"Found last profile '{lastprofil}' in time series columns.")
                df_sector_series = df_time_series[['time', lastprofil]].copy()
                df_sector_series_clean = df_sector_series.dropna(subset=[lastprofil])
                logger.info(f"First 5 entries for '{variable_name}':\n{df_sector_series_clean.head()}")
                logger.info(f"Last 5 entries for '{variable_name}':\n{df_sector_series_clean.tail()}")
                series_length = len(df_sector_series_clean)
                if series_length == 35136:
                    logger.info(f"Quarter-hourly values detected for profile {lastprofil} (normal year).")
                    df_sector_series_clean = df_sector_series_clean.set_index('time').resample('h').mean().reset_index()
                    df_sector_series_clean.rename(columns={lastprofil: 'Wert'}, inplace=True)
                    logger.info(f"Resampled to hourly. New length: {len(df_sector_series_clean)}")
                    feb_29_mask = (df_sector_series_clean['time'].dt.month == 2) & (df_sector_series_clean['time'].dt.day == 29)
                    df_sector_series_clean = df_sector_series_clean[~feb_29_mask]
                    logger.info(f"Removed February 29th. New length: {len(df_sector_series_clean)}")
                elif series_length == 35184:
                    logger.info(f"Quarter-hourly values detected for profile {lastprofil} (leap year).")
                    df_sector_series_clean = df_sector_series_clean.set_index('time').resample('h').mean().reset_index()
                    df_sector_series_clean.rename(columns={lastprofil: 'Wert'}, inplace=True)
                    logger.info(f"Resampled to hourly. New length: {len(df_sector_series_clean)}")
                    feb_29_mask = (df_sector_series_clean['time'].dt.month == 2) & (df_sector_series_clean['time'].dt.day == 29)
                    df_sector_series_clean = df_sector_series_clean[~feb_29_mask]
                    logger.info(f"Removed February 29th. New length: {len(df_sector_series_clean)}")
                elif series_length == 8760:
                    logger.info(f"Hourly values detected for profile {lastprofil} (normal year). No resampling needed.")
                    df_sector_series_clean.rename(columns={lastprofil: 'Wert'}, inplace=True)
                elif series_length == 8784:
                    logger.info(f"Hourly values detected for profile {lastprofil} (leap year).")
                    df_sector_series_clean.rename(columns={lastprofil: 'Wert'}, inplace=True)
                    feb_29_mask = (df_sector_series_clean['time'].dt.month == 2) & (df_sector_series_clean['time'].dt.day == 29)
                    df_sector_series_clean = df_sector_series_clean[~feb_29_mask]
                    logger.info(f"Removed February 29th. New length: {len(df_sector_series_clean)}")
                else:
                    logger.info(f"Unexpected series length {series_length} for profile {lastprofil}. Skipping.")
                    continue
                if len(df_sector_series_clean) != 8760:
                    logger.warning(f"Warning: Length for '{variable_name}' is {len(df_sector_series_clean)}, expected 8760.")
                time_series_dict[variable_name] = df_sector_series_clean
                profile_mapping[str(lastprofil)] = variable_name
                logger.info(f"Created Load Profile '{variable_name}' for sector '{sector}' with {len(df_sector_series_clean)} rows.")
            else:
                logger.info(f"Warning: Lastprofil '{lastprofil}' not found in the time series data. Skipping sector '{sector}'.")
        logger.info(f"\nFinal profile mapping size: {len(profile_mapping)}, time series dict size: {len(time_series_dict)}")
        return profile_mapping, time_series_dict

    # ---------------------------
    # Main logic to choose processing path based on use_excel flag
    # ---------------------------
    if use_excel:
        return use_alternative_load_profiles(file_path_profiles)
    else:
        return process_and_resample_csv(file_path_profiles, year_of_interest)


# =============================================================================
# FUNCTION: load_all_sector_weights
# =============================================================================
def load_all_sector_weights(file_path):
    """
    Loads all sector weights from the specified Excel file.

    Parameters:
      - file_path (str): Path to the CONFIG_SECTOR_WEIGHTS.xlsx file.

    Returns:
      - dict: A dictionary of DataFrames for each sheet, with columns renamed to ['Sector', 'Weight'].
    """
    weight_sectors = pd.read_excel(file_path, sheet_name=None, header=None)
    for sheet in weight_sectors:
        weight_sectors[sheet].columns = ['Sector', 'Weight']
    return weight_sectors

# =============================================================================
# FUNCTION: get_sector_weights_for_state
# =============================================================================
def get_sector_weights_for_state(federal_state_name, weight_sectors):
    """
    Fetches sector weights for a given federal state.

    Parameters:
      - federal_state_name (str): The name of the federal state.
      - weight_sectors (dict): Dictionary of sector weight DataFrames.

    Returns:
      - DataFrame or None: The DataFrame containing weights for the federal state, or None if not found.
    """
    if federal_state_name in weight_sectors:
        logger.info(f"Sector weights found for {federal_state_name}.")
        sector_weights_df = weight_sectors[federal_state_name]
        logger.info(f"{sector_weights_df.head()}")
        return sector_weights_df
    else:
        logger.info(f"No sector weights found for {federal_state_name}.")
        return None

# =============================================================================
# FUNCTION: process_federal_state
# =============================================================================
def process_federal_state(file_path_load, file_path_mapping, output_directory_time_series, output_directory_aggregated_data, profile_mapping, time_series_dict, year_of_interest, use_weights, weight_file_path, use_excel, excel_profiles_path):
    """
    Processes the data for a given federal state by generating hourly demand data and 
    saving both combined and aggregated time series files.

    Parameters:
      - file_path_load (str): Path to the input load file for the federal state.
      - file_path_mapping (str): Path to the mapping file.
      - output_directory_time_series (str): Output directory for time series files.
      - output_directory_aggregated_data (str): Output directory for aggregated files.
      - profile_mapping (dict): Mapping of profile IDs to variable names.
      - time_series_dict (dict): Dictionary of time series DataFrames.
      - year_of_interest (int or str): The target year for processing.
      - use_weights (bool): Flag to indicate whether to apply sector weights.
      - weight_file_path (str): Path to the sector weights file.
      - use_excel (bool): Flag to indicate Excel-based processing.
      - excel_profiles_path (str): Path to the Excel profiles file.

    Returns:
      - None
    """
    # ---------------------------
    # Helper function to extract federal state name from file path.
    # ---------------------------
    def get_federal_state_from_path(file_path):
        filename = os.path.basename(file_path)
        if filename.startswith("Energiebilanz_Verteilung_") and filename.endswith(".xlsx"):
            state_name = filename.split('_')[2]
            return state_name.capitalize()
        else:
            logger.warning(f"Warning: File name does not match expected pattern: {filename}")
            return None

    federal_state_name = get_federal_state_from_path(file_path_load)
    if federal_state_name in processed_states:
        logger.info(f"Skipping already processed state: {federal_state_name}")
        return
    processed_states.add(federal_state_name)

    file_name_aggregated = f"{federal_state_name}_time_series_aggregated"
    
    # Load mapping data
    if use_excel and excel_profiles_path:
        mapping_df = pd.read_excel(excel_profiles_path, sheet_name='Mapping')
        logger.info(f"Loaded mapping from {excel_profiles_path}")
    else:
        mapping_df = pd.read_excel(file_path_mapping, sheet_name='Mapping')
        logger.info(f"Loaded mapping from {file_path_mapping}")

    # Load sector weights for the current state
    weight_sectors = load_all_sector_weights(path_sector_weights)
    sector_weights_df = get_sector_weights_for_state(federal_state_name, weight_sectors)
    if sector_weights_df is not None:
        logger.info(f"Sector weights for {federal_state_name}:")
        logger.info(f"{sector_weights_df}")
        sector_weights = pd.Series(sector_weights_df.Weight.values, index=sector_weights_df.Sector).to_dict()
    else:
        logger.info(f"No sector weights found for {federal_state_name}. Proceeding without sector-specific weights.")
        sector_weights = None

    # ---------------------------
    # Inner function: process_and_generate_hourly_demand_dataframe
    # ---------------------------
    def process_and_generate_hourly_demand_dataframe(file_path_load, time_series_dict, profile_mapping, year_of_interest, use_weights, sector_weights, output_directory_time_series, federal_state_name, use_excel, excel_profiles_path, file_path_mapping):
        """
        Processes the sector data for the federal state, applies weights if necessary, and 
        generates hourly demand time series for each sector.
        """
        logger.info(f"\n--- Starting data processing for file: {file_path_load} ---")
        logger.info(f"Year of Interest: {year_of_interest}, Use Weights: {use_weights}")

        # Load sector data from the file.
        sectors_df = pd.read_excel(file_path_load, sheet_name='Sheet1')
        logger.info(f"Loaded sector data from: {file_path_load}")

        # Apply weights if available.
        if use_weights and sector_weights is not None:
            sectors_df['Weighted Energy Consumption (MWh)'] = sectors_df['Allocated Energy Consumption (MWh)'] * sectors_df['Sektor'].str.strip().map(sector_weights).fillna(1)
        else:
            sectors_df['Weighted Energy Consumption (MWh)'] = sectors_df['Allocated Energy Consumption (MWh)']

        all_hourly_demands = []
        nuts_ids = []
        sector_ids = []
        is_leap_year = calendar.isleap(int(year_of_interest))
        expected_length = 8784 if is_leap_year else 8760

        # Process each sector in the DataFrame.
        for index, sector_row in sectors_df.iterrows():
            demand = sector_row['Weighted Energy Consumption (MWh)']
            sector_id = sector_row['Sektor']
            nuts_id = sector_row['NUTS-Region']
            logger.info(f"\nProcessing sector: {sector_id}, NUTS ID: {nuts_id}, Demand: {demand}")

            # Retrieve Lastprofil ID from the mapping.
            lastprofil_id = mapping_df.loc[mapping_df['Sektor'] == sector_id, 'Lastprofil'].values[0]
            lastprofil_id_str = str(lastprofil_id)
            logger.info(f"Looking for Lastprofil ID '{lastprofil_id_str}' in profile_mapping...")

            # Validate time series key existence.
            time_series_key = profile_mapping.get(lastprofil_id_str)
            if time_series_key is None:
                logger.info(f"Profile mapping content: {profile_mapping}")
                raise KeyError(f"Time series key for Lastprofil ID '{lastprofil_id_str}' not found in profile_mapping.")
            if time_series_key not in time_series_dict:
                logger.info(f"Time series dictionary content: {list(time_series_dict.keys())}")
                raise KeyError(f"Time series key '{time_series_key}' not found in time_series_dict.")

            time_series_df = time_series_dict[time_series_key]
            time_series = time_series_df['Wert'].values[1:]

            # Handle NaN values in the time series.
            if np.isnan(time_series).any():
                logger.info(f"Filling NaN values in time series for sector {sector_id}.")
                time_series = np.nan_to_num(time_series, nan=0.0)

            logger.info(f"Loaded time series for sector '{sector_id}' (Lastprofil ID: {lastprofil_id_str}). First 5 values: {time_series[:5]}")

            # Adjust time series length if necessary.
            if len(time_series) != expected_length:
                if len(time_series) == 35136:
                    time_series = time_series.reshape(-1, 4).sum(axis=1)
                    logger.info(f"Reshaped time series to {len(time_series)} after aggregation.")
                elif len(time_series) == 8760:
                    logger.info("Time series length is already 8760, no adjustment needed.")
                elif len(time_series) == 8761:
                    logger.info("Time series has 8761 values, adjusting to 8760 by removing the last value.")
                    time_series = time_series[:8760]
                elif len(time_series) == 8784:
                    logger.info("Time series length is 8784, no adjustment needed for leap year.")
                elif len(time_series) == 8759:
                    logger.info("Time series has 8759 values, adjusting to 8760 by duplicating the last value.")
                    time_series = np.append(time_series, time_series[-1])
                else:
                    raise ValueError(f"Unexpected time series length: {len(time_series)}")

            # Normalize constant time series.
            if np.all(time_series == time_series[0]):
                logger.info(f"Constant time series detected for sector {sector_id}. Normalizing to avoid skewed results.")
                normalized_time_series = time_series / time_series.sum()
            else:
                if time_series.sum() == 0:
                    logger.warning(f"Warning: Sum of time series for sector {sector_id} is zero, skipping normalization.")
                    normalized_time_series = time_series
                else:
                    normalized_time_series = time_series / time_series.sum()

            hourly_demand = normalized_time_series * demand
            demand_df = pd.DataFrame(hourly_demand)
            logger.info("-------------")
            logger.info(f"Hourly demand for sector {sector_id}: First 5 values: {hourly_demand[:5]}")
            logger.info(f"Hourly demand for sector {sector_id}: Last 5 values: {demand_df.tail()}")
            logger.info("----------------")

            all_hourly_demands.append(hourly_demand)
            sector_ids.append(sector_id)
            nuts_ids.append(nuts_id)

        actual_length = len(all_hourly_demands[0])
        logger.info(f"Actual time series length used for column names: {actual_length}")
        column_names = [f'k{i+1:04d}' for i in range(actual_length)]
        hourly_demand_df = pd.DataFrame(all_hourly_demands, columns=column_names)
        hourly_demand_df.insert(0, 'Sector ID', sector_ids)
        hourly_demand_df.insert(1, 'NUTS-ID', nuts_ids)
        logger.info(f"Created DataFrame with shape: {hourly_demand_df.shape}")
        return hourly_demand_df

    # Process and generate the hourly demand DataFrame.
    combined_df_8760 = process_and_generate_hourly_demand_dataframe(
        file_path_load, 
        time_series_dict, 
        profile_mapping, 
        year_of_interest, 
        use_weights=use_weights, 
        sector_weights=sector_weights,  
        output_directory_time_series=output_directory_time_series, 
        federal_state_name=federal_state_name, 
        use_excel=use_excel, 
        excel_profiles_path=excel_profiles_path, 
        file_path_mapping=file_path_mapping
    )

    if combined_df_8760 is None:
        logger.error(f"Error: Could not generate hourly demand data for {federal_state_name}. Skipping this state.")
        return

    # ---------------------------
    # Inner function: aggregate_and_save
    # ---------------------------
    def aggregate_and_save(combined_df, output_folder, filename, use_weights):
        """
        Aggregates hourly demand data by summing over NUTS-ID and saves the resulting DataFrame to Excel.

        Parameters:
          - combined_df (DataFrame): DataFrame containing hourly demand data.
          - output_folder (str): Directory to save the aggregated file.
          - filename (str): Base filename for the aggregated file.
          - use_weights (bool): Flag indicating if weights were applied (affects filename suffix).

        Returns:
          - DataFrame: Aggregated DataFrame.
        """
        aggregated_df_8760 = combined_df.groupby('NUTS-ID').sum().reset_index()
        aggregated_df_8760 = aggregated_df_8760.drop(columns=['Sector ID'])
        weight_suffix = '_weights' if use_weights else ''
        filename_with_weights = f"{filename}{weight_suffix}_{date}.xlsx"
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        output_path = os.path.join(output_folder, filename_with_weights)
        aggregated_df_8760.to_excel(output_path, index=False)
        logger.info(f"Aggregated DataFrame saved to: {output_path}")
        return aggregated_df_8760

    aggregated_df = aggregate_and_save(combined_df_8760, output_directory_aggregated_data, file_name_aggregated, use_weights)

    # ---------------------------
    # Inner function: save_created_time_series
    # ---------------------------
    def save_created_time_series(combined_df, aggregated_df, output_dir, use_weights, use_excel, federal_state_name, year_of_interest):
        """
        Saves both the combined and aggregated time series DataFrames to Excel files with descriptive filenames.

        Parameters:
          - combined_df (DataFrame): The combined hourly demand DataFrame.
          - aggregated_df (DataFrame): The aggregated DataFrame.
          - output_dir (str): Directory to save the files.
          - use_weights (bool): Flag indicating whether weights were used.
          - use_excel (bool): Flag indicating Excel-based profile generation.
          - federal_state_name (str): The federal state being processed.
          - year_of_interest (int or str): The target year.

        Returns:
          - None
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        profile_type = '_optional_profiles' if use_excel else '_synthetic_profiles'
        weight_suffix = '_weights' if use_weights else ''
        combined_df_path = os.path.join(output_dir, f'{federal_state_name}_sectors_mapping_{year_of_interest}{weight_suffix}{profile_type}_{date}.xlsx')
        aggregated_df_path = os.path.join(output_dir, f'{federal_state_name}_nuts_mapping_{year_of_interest}{weight_suffix}{profile_type}_{date}.xlsx')
        combined_df.to_excel(combined_df_path)
        aggregated_df.to_excel(aggregated_df_path)
        logger.info(f"Saved combined_df to {combined_df_path}")
        logger.info(f"Saved aggregated_df to {aggregated_df_path}")
        logger.info("\n--------------------------------------------------------------------------------")

    save_created_time_series(combined_df_8760, aggregated_df, output_directory_time_series, use_weights, use_excel, federal_state_name, year_of_interest)
