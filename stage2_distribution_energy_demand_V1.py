import pandas as pd
import os
from datetime import datetime
import time
from config_tool_V1 import *
from logging_function import setup_stage_logger

# =============================================================================
# SETUP AND INITIALIZATION
# =============================================================================

# Setup the logger for this stage based on the stage name.
logger = setup_stage_logger("Stage1_Data_Preprocessing")
logger.info("Starting Data Preprocessing Stage...")

# Example logging for loading and processing steps.
try:
    logger.info("Loading data...")
    # Code for loading data goes here
    # ...

    logger.info("Processing data...")
    # Code for data processing goes here
    # ...

    logger.info("Stage completed successfully.")
except Exception as e:
    logger.error(f"An error occurred: {e}")

# Record the start time of Stage 2
start_time = time.time()

# =============================================================================
# STAGE 2: Breakdown of Energy Demand by NUTS3 Level
# =============================================================================

print("--------------------------------------------------------------------------------")
print("--------------------------------------------------------------------------------")
print("Start with Stage 2: Breakdown of energy demand by NUTS3 level")
print()
print("--------------------------------------------------------------------------------")

# List of all federal states to be processed
federal_states = [
    'Burgenland', 'Niederösterreich', 'Wien', 'Kärnten', 'Steiermark', 
    'Oberösterreich', 'Salzburg', 'Tirol', 'Vorarlberg'
]

# Current date used for file naming
today = datetime.today().strftime('%Y%m%d')

# =============================================================================
# FUNCTION DEFINITIONS
# =============================================================================

def check_required_files(*file_paths):
    """
    Checks whether all required files exist.

    Parameters:
    - file_paths (str): Variable number of file path strings.

    Returns:
    - bool: True if all files exist; False otherwise.
    """
    all_files_exist = True
    for file_path in file_paths:
        if not os.path.exists(file_path):
            print(f"Required file not found: {file_path}")
            all_files_exist = False
    return all_files_exist

def get_nuts_codes(federal_state_name):
    """
    Retrieves NUTS2 and NUTS3 codes for a given federal state.

    Parameters:
    - federal_state_name (str): The name of the federal state.

    Returns:
    - dict: Dictionary containing 'nuts2_region' and 'nuts3_regions' for the state, or None if not found.
    """
    nuts_mapping = {
        'Burgenland': {'nuts2_region': 'AT11', 'nuts3_regions': ['AT111', 'AT112', 'AT113']},
        'Niederösterreich': {'nuts2_region': 'AT12', 'nuts3_regions': ['AT121', 'AT122', 'AT123', 'AT124', 'AT125', 'AT126', 'AT127']},
        'Wien': {'nuts2_region': 'AT13', 'nuts3_regions': ['AT130']},
        'Kärnten': {'nuts2_region': 'AT21', 'nuts3_regions': ['AT211', 'AT212', 'AT213']},
        'Steiermark': {'nuts2_region': 'AT22', 'nuts3_regions': ['AT221', 'AT222', 'AT223', 'AT224', 'AT225', 'AT226']},
        'Oberösterreich': {'nuts2_region': 'AT31', 'nuts3_regions': ['AT311', 'AT312', 'AT313', 'AT314', 'AT315']},
        'Salzburg': {'nuts2_region': 'AT32', 'nuts3_regions': ['AT321', 'AT322', 'AT323']},
        'Tirol': {'nuts2_region': 'AT33', 'nuts3_regions': ['AT331', 'AT332', 'AT333', 'AT334', 'AT335']},
        'Vorarlberg': {'nuts2_region': 'AT34', 'nuts3_regions': ['AT341', 'AT342']}
    }
    return nuts_mapping.get(federal_state_name)

def distribute_energy_consumption(file_path_population, file_path_energy, energy_sheet_name, nuts2_region):
    """
    Distributes energy consumption for private households based on population data.

    Note:
    This function uses the global variable `year_of_interest` to determine the energy consumption column.

    Parameters:
    - file_path_population (str): Path to the Excel file containing population data.
    - file_path_energy (str): Path to the Excel file containing energy data.
    - energy_sheet_name (str): Name of the sheet in the energy data file.
    - nuts2_region (str): NUTS2 region code for filtering population data.

    Returns:
    - pd.DataFrame: DataFrame with allocated energy consumption for private households.
    """
    population_data = pd.read_excel(file_path_population)
    population_data['NUTS-Region'] = population_data['NUTS-Region'].astype(str)
    subregion_population = population_data.copy()[
        (population_data['NUTS-Region'].str.startswith(nuts2_region)) & 
        (population_data['NUTS-Region'].str.len() == 5)
    ]
    energy_data = pd.read_excel(file_path_energy, sheet_name=energy_sheet_name)
    # Create a copy of energy data (if further modifications are needed)
    energy_data.copy()
    private_households_data = energy_data[energy_data['Sektor'] == 'Private Haushalte']
    total_energy_consumption = private_households_data[f"Sektoraler Energetischer Endverbrauch {year_of_interest}"].values[0]
    total_population = subregion_population[2022].sum()
    subregion_population['Allocated Energy Consumption (MWh)'] = (
        subregion_population[2022] / total_population * total_energy_consumption
    )
    result_df = subregion_population[['NUTS-Region', 'Allocated Energy Consumption (MWh)']].copy()
    result_df['Sektor'] = 'Private Haushalte'
    result_df['ÖNACE'] = 'T'
    return result_df

def allocate_energy_consumption(file_path_a, sheet_name_a, file_path_c, file_path_population, nuts2_region):
    """
    Allocates energy consumption across sectors and subregions based on employment data.

    Parameters:
    - file_path_a (str): Path to the Excel file containing energy data.
    - sheet_name_a (str): Sheet name in the energy data file.
    - file_path_c (str): Path to the Excel file containing workforce data.
    - file_path_population (str): Path to the Excel file containing population data.
    - nuts2_region (str): NUTS2 region code for filtering data.

    Returns:
    - tuple: Two DataFrames; first with sector-wise allocation, second with final combined allocation.
    """
    data_a = pd.read_excel(file_path_a, sheet_name=sheet_name_a)
    data_c = pd.read_excel(file_path_c)
    data_c = data_c[
        data_c['NUTS-ID'].astype(str).str.startswith(nuts2_region) & 
        (data_c['NUTS-ID'].astype(str).str.len() == 5)
    ]
    data_c['Beschäftigte'] = pd.to_numeric(data_c['Beschäftigte'], errors='coerce')
    sector_employment_sum = data_c.groupby('ÖNACE')['Beschäftigte'].sum().reset_index()
    sector_employment_sum.rename(columns={'Beschäftigte': 'Total Employment'}, inplace=True)
    data_c = pd.merge(data_c, sector_employment_sum, on='ÖNACE')
    data_c['Employment Proportion'] = data_c['Beschäftigte'] / data_c['Total Employment']
    merged_data = pd.merge(data_a, data_c, left_on='ÖNACE', right_on='ÖNACE')
    merged_data['Allocated Energy Consumption (MWh)'] = (
        merged_data[f"Sektoraler Energetischer Endverbrauch {year_of_interest}"] * merged_data['Employment Proportion']
    )
    # Get allocation for private households and concatenate with the sector allocation
    private_household_consumption = distribute_energy_consumption(file_path_population, file_path_a, sheet_name_a, nuts2_region)
    final_data = pd.concat([merged_data, private_household_consumption], ignore_index=True)
    final_data = final_data[
        final_data['NUTS-Region'].astype(str).str.startswith(nuts2_region) & 
        (final_data['NUTS-Region'].astype(str).str.len() == 5)
    ]
    return merged_data, final_data

def adjust_and_combine_dataframes(allocated_data_sectors, allocated_data_privat):
    """
    Adjusts and combines dataframes from sectoral and private household allocations.

    Parameters:
    - allocated_data_sectors (pd.DataFrame): DataFrame with sector-based energy allocation.
    - allocated_data_privat (pd.DataFrame): DataFrame with private household energy allocation.

    Returns:
    - pd.DataFrame: Combined DataFrame with standardized columns.
    """
    allocated_data_privat = allocated_data_privat[['NUTS-Region', 'Sektor', 'ÖNACE', 'Allocated Energy Consumption (MWh)']]
    if 'NUTS-ID' in allocated_data_sectors.columns:
        allocated_data_sectors = allocated_data_sectors.rename(columns={'NUTS-ID': 'NUTS-Region'})
    allocated_data_sectors = allocated_data_sectors[['NUTS-Region', 'Sektor', 'ÖNACE', 'Allocated Energy Consumption (MWh)']]
    combined_data = pd.concat([allocated_data_sectors, allocated_data_privat], ignore_index=True)
    return combined_data

def process_agricultural_data_for_state(file_path_area, file_path_demand, file_path_stats, flachennutzung_sheet, electrical_demand_sheet, nuts2_region, nuts3_regions):
    """
    Processes agricultural data for a federal state by combining land use, energy demand, and general statistics.

    Parameters:
    - file_path_area (str): Path to the Excel file containing land use data.
    - file_path_demand (str): Path to the Excel file containing electrical demand data.
    - file_path_stats (str): Path to the Excel file containing general statistics data.
    - flachennutzung_sheet (str): Sheet name in the land use data file.
    - electrical_demand_sheet (str): Sheet name in the electrical demand data file.
    - nuts2_region (str): NUTS2 region code for filtering.
    - nuts3_regions (list): List of NUTS3 region codes to filter general statistics.

    Returns:
    - pd.DataFrame: DataFrame with allocated energy consumption for agriculture,
      or an empty DataFrame if data is missing.
    """
    flachennutzung_df = pd.read_excel(file_path_area, sheet_name=flachennutzung_sheet)
    electrical_demand_df = pd.read_excel(file_path_demand, sheet_name=electrical_demand_sheet)
    gemeindestatistik_df = pd.read_excel(file_path_stats)
    
    # Standardize NUTS column by stripping whitespace if it exists
    if 'NUTS ' in gemeindestatistik_df.columns:
        gemeindestatistik_df['NUTS '] = gemeindestatistik_df['NUTS '].str.strip()
    else:
        print("The 'NUTS' column was not found. Please check the column names and update accordingly.")
        return pd.DataFrame()

    # Extract the total agricultural area from the land use data
    total_agricultural_area_row = flachennutzung_df.loc[flachennutzung_df.iloc[:, 0].str.contains("Gesamtfläche insgesamt", na=False)]
    if not total_agricultural_area_row.empty:
        total_agricultural_area = total_agricultural_area_row.iloc[0, 1]
    else:
        print("No data found for total agricultural area.")
        return pd.DataFrame()

    # Extract the total electrical demand for the agricultural sector
    if not electrical_demand_df.empty:
        total_electrical_demand = electrical_demand_df.loc[
            electrical_demand_df['Sektor'] == 'Landwirtschaft', 
            f"Sektoraler Energetischer Endverbrauch {year_of_interest}"
        ].values[0]
    else:
        print("No data found for electrical demand.")
        return pd.DataFrame()

    # Filter general statistics data for the specified NUTS3 regions
    gemeindestatistik_df_filtered = gemeindestatistik_df[gemeindestatistik_df['NUTS '].isin(nuts3_regions)].copy()
    if gemeindestatistik_df_filtered.empty:
        print("No data found after filtering for specified NUTS3 regions.")
        return pd.DataFrame()

    # Convert area data from km² to hectares (1 km² = 100 ha) after cleaning the string format
    gemeindestatistik_df_filtered['Fläche ha'] = gemeindestatistik_df_filtered['Fläche in km² 1.1.2021'] \
        .str.replace('.', '').str.replace(',', '.').astype(float) * 100

    total_general_area_ha = gemeindestatistik_df_filtered['Fläche ha'].sum()
    gemeindestatistik_df_filtered['Area Proportion'] = gemeindestatistik_df_filtered['Fläche ha'] / total_general_area_ha
    gemeindestatistik_df_filtered['Agricultural Area (ha)'] = gemeindestatistik_df_filtered['Area Proportion'] * total_agricultural_area
    gemeindestatistik_df_filtered['Electrical Demand (MWh)'] = (
        gemeindestatistik_df_filtered['Agricultural Area (ha)'] / 
        gemeindestatistik_df_filtered['Agricultural Area (ha)'].sum()
    ) * total_electrical_demand

    # Rename columns and add sector classification for agriculture
    gemeindestatistik_df_filtered = gemeindestatistik_df_filtered.rename(columns={
        'NUTS ': 'NUTS-Region',
        'Electrical Demand (MWh)': 'Allocated Energy Consumption (MWh)'
    })
    gemeindestatistik_df_filtered['Sektor'] = 'Landwirtschaft'
    gemeindestatistik_df_filtered['ÖNACE'] = 'A'
    
    return gemeindestatistik_df_filtered[['NUTS-Region', 'Sektor', 'ÖNACE', 'Allocated Energy Consumption (MWh)']]

def save_combined_dataframe(combined_df_final, output_folder, flachennutzung_sheet):
    """
    Saves the combined DataFrame to an Excel file with a standardized filename.

    Parameters:
    - combined_df_final (pd.DataFrame): The final combined DataFrame.
    - output_folder (str): Directory path to save the file.
    - flachennutzung_sheet (str): Identifier for the dataset (used in the filename).

    Returns:
    - None
    """
    print(f"Processing {flachennutzung_sheet} dataset")
    os.makedirs(output_folder, exist_ok=True)
    filename = f"Energiebilanz_Verteilung_{flachennutzung_sheet}_{today}.xlsx"
    file_path = os.path.join(output_folder, filename)
    combined_df_final.to_excel(file_path, index=False)
    print(f"DataFrame saved successfully to {file_path}")

def process_all_federal_states(federal_states, data_path_energy, data_path_workforce, data_path_population, data_path_farming, file_path_stats, stage2_output_folder):
    """
    Processes energy demand data for all federal states by allocating energy consumption across regions.

    Parameters:
    - federal_states (list): List of federal state names.
    - data_path_energy (str): Path to the energy data Excel file.
    - data_path_workforce (str): Path to the workforce data Excel file.
    - data_path_population (str): Path to the population data Excel file.
    - data_path_farming (str): Path to the land use data Excel file.
    - file_path_stats (str): Path to the general statistics data Excel file.
    - stage2_output_folder (str): Directory to save the output files.

    Returns:
    - None
    """
    # Check for required input files before processing
    required_files = [data_path_energy, data_path_workforce, data_path_population, data_path_farming]
    if not check_required_files(*required_files):
        print("Correct input data is not found, please run Stage 1 before running Stage 2.")
        return

    for flachennutzung_sheet in federal_states:
        nuts_codes = get_nuts_codes(flachennutzung_sheet)
        if nuts_codes:
            nuts2_region = nuts_codes['nuts2_region']
            nuts3_regions = nuts_codes['nuts3_regions']
        else:
            raise ValueError(f"Federal state name '{flachennutzung_sheet}' not found in NUTS mapping.")

        # ---------------------------------------------------------------------
        # Replace German umlauts in sheet names to create valid Excel sheet identifiers
        # ---------------------------------------------------------------------
        def replace_umlauts(text):
            """
            Replaces German umlauts in a given text with their equivalent representations.

            Parameters:
            - text (str): The input string.

            Returns:
            - str: The modified string with umlauts replaced.
            """
            umlaut_map = {
                "ä": "ae",
                "ö": "oe",
                "ü": "ue",
                "Ä": "Ae",
                "Ö": "Oe",
                "Ü": "Ue",
                "ß": "ss"
            }
            for umlaut, replacement in umlaut_map.items():
                text = text.replace(umlaut, replacement)
            return text

        electrical_demand_sheet = replace_umlauts(flachennutzung_sheet + 'Daten')
        
        # Allocate energy consumption based on sector and private households
        allocated_data_sectors, allocated_data_privat = allocate_energy_consumption(
            data_path_energy, electrical_demand_sheet, data_path_workforce, data_path_population, nuts2_region
        )
        
        # Combine the allocated data from sectors and private households
        combined_data = adjust_and_combine_dataframes(allocated_data_sectors, allocated_data_privat)
        
        # Process agricultural (land use) data for the state
        land_forest_df = process_agricultural_data_for_state(
            data_path_farming, data_path_energy, file_path_stats, flachennutzung_sheet, electrical_demand_sheet, nuts2_region, nuts3_regions
        )
        
        # Merge sector/private household data with agricultural data and fill missing values with 0
        combined_df_final = adjust_and_combine_dataframes(combined_data, land_forest_df)
        combined_df_final = combined_df_final.fillna(0)

        # Save the combined DataFrame to an Excel file
        save_combined_dataframe(combined_df_final, stage2_output_folder, flachennutzung_sheet)

        # Print total allocated energy consumption for the current state
        total_demand = combined_df_final['Allocated Energy Consumption (MWh)'].sum()
        print(f"Total Allocated Energy Consumption for {flachennutzung_sheet} (MWh): {total_demand}")
        print()

# =============================================================================
# EXECUTION OF STAGE 2 PROCESSING
# =============================================================================

process_all_federal_states(
    federal_states,
    data_path_energy,
    data_path_workforce,
    data_path_population,
    data_path_farming,
    file_path_stats,
    stage2_output_folder
)

# Calculate and print the total runtime of Stage 2
end_time = time.time()
elapsed_time = end_time - start_time
hours, rem = divmod(elapsed_time, 3600)
minutes, seconds = divmod(rem, 60)
print(f"Finished with Stage 2 - Total runtime script: {int(hours)}h {int(minutes)}m {int(seconds)}s")
print()
