import os
import pandas as pd
import time
from glob import glob
from datetime import datetime
from config_tool_V1 import *
import time

date = datetime.today().strftime('%Y%m%d')
start_time = time.time()

print("--------------------------------------------------------------------------------")
print("--------------------------------------------------------------------------------")
print("Start with Stage 1: Data Preprocessing")
print()
print("--------------------------------------------------------------------------------")

# Process input data
print("Processing energy balance sheets for federal states")

def process_energy_data(folder_path, sheet_name, sheet_index, year_of_interest):
    """
    Processes energy data from ODS files for federal states, extracting data for the specified year and sectors.

    Parameters:
    - folder_path: str, the path to the folder containing the ODS files.
    - sheet_name: str, the name of the sheet containing the data.
    - sheet_index: int, the index of the sheet (optional).
    - year_of_interest: int, the year for which the data is extracted.

    Returns:
    - Dictionary of DataFrames where keys are file names and values are DataFrames with energy consumption data 
    filtered by sectors (ÖNACE) and year of interest.
    """

    # Check if the year_of_interest is within valid range
    if year_of_interest < 1988:
        print(f"Warning: Year of interest {year_of_interest} is earlier than the available data (1988). Using 1988 instead.")
        year_of_interest = 1988
    elif year_of_interest > 2022:
        print(f"Warning: Year of interest {year_of_interest} is later than the latest available data (2022). Using 2022 instead.")
        year_of_interest = 2022

    print(f"Processing data for the year: {year_of_interest}")

    # Mapping of categories to ÖNACE codes
    category_to_onace = {
        "Eisen- und Stahlerzeugung": "C",
        "Chemie und  Petrochemie": "C",
        "Nicht Eisen Metalle": "C",
        "Steine und Erden, Glas": "C",
        "Fahrzeugbau": "C",
        "Maschinenbau": "C",
        "Bergbau": "B",
        "Nahrungs- und Genußmittel, Tabak": "C",
        "Papier und Druck": "C",
        "Holzverarbeitung": "C",
        "Bau": "F",
        "Textil und Leder": "C",
        "Sonst. Produzierender Bereich": "C",
        "Eisenbahn": "H",
        "Sonstiger Landverkehr": "H",
        "Transport in Rohrfernleitungen": "H",
        "Binnenschiffahrt": "H",
        "Flugverkehr": "H",
        "Öffentliche und Private Dienstleistungen": "S",
        "Private Haushalte": "T",
        "Landwirtschaft": "A"
    }

    def read_ods_to_dataframe(file_path, sheet_name=None, sheet_index=0):
        print(f"Reading data from file: {file_path}")
        
        if not os.path.exists(file_path):
            print(f"Error: The file {file_path} does not exist.")
            return None

        if not file_path.lower().endswith('.ods'):
            print(f"Error: The file {file_path} is not an ODS file.")
            return None

        df = pd.read_excel(file_path, sheet_name=sheet_name, engine='odf', skiprows=3)
        print(f"Successfully loaded data from {file_path}")
        return df

    def preprocess_dataframe(df, year_of_interest):
        print("Preprocessing data for the year of interest")
        year_row = df.columns

        if int(year_of_interest) in year_row:
            year_column = int(year_of_interest)  # Ensure we're working with integers
        else:
            print(f"Error: No column found for the year {year_of_interest}.")
            return pd.DataFrame()

        selected_rows = df.iloc[435:462, :].copy()
        filtered_selected_rows = selected_rows[selected_rows.iloc[:, 0].isin(category_to_onace.keys())].copy()
        if filtered_selected_rows.empty:
            print("Warning: No matching categories found in the selected rows.")
            return pd.DataFrame()

        filtered_data_year = filtered_selected_rows[[filtered_selected_rows.columns[0], year_column]].copy()
        filtered_data_year.columns = ["Sektor", f"Sektoraler Energetischer Endverbrauch {year_of_interest}"]
        filtered_data_year["ÖNACE 2008"] = filtered_data_year["Sektor"].map(category_to_onace)
        filtered_data_year["Einheit"] = "MWh"

        final_df = filtered_data_year[["Sektor", "ÖNACE 2008", f"Sektoraler Energetischer Endverbrauch {year_of_interest}", "Einheit"]]
        final_df = final_df.rename(columns={"ÖNACE 2008": "ÖNACE"})
        print(f"Data preprocessing completed for year {year_of_interest}")

        return final_df

    ods_files = glob(os.path.join(folder_path, '*.ods'))
    dataframes = {}
    
    for file_path in ods_files:
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        print(f"\nStarting processing for file: {file_name}")

        df = read_ods_to_dataframe(file_path, sheet_name, sheet_index)
        
        if df is not None:
            processed_df = preprocess_dataframe(df, year_of_interest)

            if not processed_df.empty:
                globals()[file_name] = processed_df
                dataframes[file_name] = processed_df
                print(f"Finished processing and storing data for file: {file_name}")
            else:
                print(f"Warning: No data for {year_of_interest} in file {file_name}")
        else:
            print(f"Skipping file {file_name} due to loading issues.")

    return dataframes

data_energy_states = process_energy_data(folder_path_states, sheet_name_sectors, sheet_index, year_of_interest)
print("\nProcessed DataFrames:")
for df_name in data_energy_states.keys():
    print(f"- {df_name}")
print("Finished processing energy balance sheets")    
print()


# Additional Processing Functions
print("Processing employment statistics data")
print()

def process_workforce_data(file_path, sheet_name):
    """
    Processes workforce data from the specified file and sheet, extracting relevant columns for analysis.

    Parameters:
    - file_path: str, the path to the workforce data file.
    - sheet_name: str, the name of the sheet containing the workforce data.

    Returns:
    - DataFrame containing ÖNACE sector classification, sector description, NUTS-ID region, and the number of employees 
    ('Beschäftigte') in each sector.
    """
    print(f"Loading workforce data from file: {file_path}, sheet: {sheet_name}")
    data_workforce_raw = pd.read_excel(file_path, sheet_name=sheet_name, engine='odf', header=1)
    data_workforce = data_workforce_raw.copy()
    
    # Rename columns for clarity
    rename_dict = {
        'ÖNACE 2008Abschnitt': 'ÖNACE',
        'Beschäftigteim Jahres-durchschnitt inges.': 'Beschäftigte'
    }
    data_workforce = data_workforce.rename(columns=rename_dict)
    columns_to_keep = ['ÖNACE', 'Kurzbezeichnung', 'NUTS-ID', 'Beschäftigte']

    data_workforce = data_workforce[columns_to_keep]

    return data_workforce

data_workforce = process_workforce_data(file_path_workforce, sheet_name_workforce)
print("Finished processing employment statistics data")
print()


print(f"Processing population statistics data for the year: {year_of_interest}")
print()

def process_population_data(file_path, sheet_name, year_of_interest):
    """
    Processes the population data from the specified file and sheet, filtering for the year of interest.
    
    Parameters:
    - file_path: str, the path to the population data file.
    - sheet_name: str, the name of the sheet in the file.
    - year_of_interest: int, the year for which the data is being processed.
    
    Returns:
    - DataFrame containing the NUTS-Region, Region Name, and population data for the year of interest.
    """
    print(f"Loading population data from file: {file_path}, sheet: {sheet_name}")
    data_population_raw = pd.read_excel(file_path, sheet_name=sheet_name, engine='odf', header=1)

    # Drop the last row (which might contain unwanted data like totals)
    data_population_cleaned = data_population_raw.drop(data_population_raw.index[-1])

    # Rename columns for clarity
    data_population_cleaned = data_population_cleaned.rename(columns={"Unnamed: 1": "Region Name"})

    year_of_interest = int(year_of_interest)
    if year_of_interest not in data_population_cleaned.columns:
        raise ValueError(f"Year {year_of_interest} not found in the population data.")

    # Select only relevant columns for further processing
    data_population_selected_year = data_population_cleaned[['NUTS-Region', 'Region Name', year_of_interest]]

    return data_population_selected_year

data_population = process_population_data(file_path_population, sheet_name_population, year_of_interest)
print("Finished processing population statistics data")
print()


print("Processing land use data of forestry and agriculture")
print()

def process_farming_data(file_path, sheet_name):
    print(f"Loading land use data from file: {file_path}, sheet: {sheet_name}")
    data_farming_raw = pd.read_excel(file_path, sheet_name=sheet_name, engine='odf', header=1)

    # Extract the row where 'Flächennutzung' equals 'Gesamtfläche insgesamt'
    row_of_interest = data_farming_raw[data_farming_raw['Flächennutzung'] == 'Gesamtfläche insgesamt']

    # Automatically get the list of region columns (excluding the first column 'Flächennutzung')
    regions = data_farming_raw.columns[1:]

    # Dictionary to hold the DataFrames for each region
    data_area_agriculture = {}

    # Loop through each region and create a DataFrame with only the row of interest
    for region in regions:
        data_area_agriculture[region] = row_of_interest[['Flächennutzung', region]].copy()

    print("Land use data successfully processed.")
    return data_area_agriculture

data_area_agriculture = process_farming_data(file_path_farming, sheet_name_farming)
print("Finished land use data")
print("--------------------------------------------------------------------------------")

# Save Data to Excel Files
def export_dataframe_to_excel(dataframe, file_name, folder_name):
    # Ensure the folder exists, create if it doesn't
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    
    # Create the full file path
    output_file_path = os.path.join(folder_name, file_name)
    
    # Export the DataFrame to an Excel file
    with pd.ExcelWriter(output_file_path, engine='xlsxwriter') as writer:
        dataframe.to_excel(writer, index=False)
    
    print(f"Data exported to {output_file_path}")

def export_land_use_data(region_dataframes, file_name, folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    output_file_path = os.path.join(folder_name, file_name)

    with pd.ExcelWriter(output_file_path, engine='xlsxwriter') as writer:
        for region, df in region_dataframes.items():
            df.to_excel(writer, sheet_name=region, index=False)

    print(f"Data exported to {output_file_path}")

def export_energy_data(dataframes, file_name, folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    output_file_path = os.path.join(folder_path, file_name)

    with pd.ExcelWriter(output_file_path, engine='xlsxwriter') as writer:
        for file_name, df in dataframes.items():
            sheet_name = file_name[:-11]
            df.to_excel(writer, sheet_name=sheet_name, index=False)

    print(f"Electrical demand data exported to {output_file_path}")

# Export Processed Data
export_energy_data(data_energy_states, filename_energy_balance_output, stage1_output_folder)
print()

export_dataframe_to_excel(data_workforce, filename_workforce_output, stage1_output_folder)
print()

export_dataframe_to_excel(data_population, filename_population_output, stage1_output_folder)
print()

export_land_use_data(data_area_agriculture, filename_farming_output, stage1_output_folder)


# Final Runtime Output
end_time = time.time()
elapsed_time = end_time - start_time
hours, rem = divmod(elapsed_time, 3600)
minutes, seconds = divmod(rem, 60)

print("--------------------------------------------------------------------------------")
print()
print(f"Finished with Stage 1")
print(f"Total runtime script: {int(hours)}h {int(minutes)}m {int(seconds)}s")
print("--------------------------------------------------------------------------------")
print("--------------------------------------------------------------------------------")