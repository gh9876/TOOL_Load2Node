import os
from datetime import datetime
from pathlib import Path

#####################################################################################################
#############  Basic Settings #############
year_of_interest = 2022

use_weights = False  # Set this to False for processing without any weights factoring

use_excel = False # Set this to True for processing with alternative load profiles

use_nuts_assignment = False  # Set this to True for running the LEGO2NUTS assignment process

# Automatically set the project root to the directory containing this script
project_root = Path(__file__).resolve().parent

# Define date format for filenames
date = datetime.today().strftime('%Y%m%d')

# Define common folders relative to the project root
folder_log = os.path.join(project_root, "LOG")

stage1_input_folder = os.path.join(project_root, "Data_Input")
raw_data_folder = os.path.join(stage1_input_folder, "RAW_DATA")

stage1_output_folder = os.path.join(project_root, "Data_Output", "Stage1")
stage2_output_folder = os.path.join(project_root, "Data_Output", "Stage2")
stage3_output_folder = os.path.join(project_root, "Data_Output", "Stage3")
stage4_output_folder = os.path.join(project_root, "Data_Output", "Stage4")


#####################################################################################################
#############  STAGE 1 #############
# File A
folder_path_states = os.path.join(raw_data_folder, "Energiebilanzen")
sheet_name_sectors = "Elektrische_Energie"  # Specify the sheet name, or use None to use sheet_index
sheet_index = 0  # Specify the sheet index (0-based) if sheet_name is None


# Filenames and sheet names for Stage 1
filename_workforce = "G_ARB_1_Arbeitsstaetten_Hauptergebnisse_nach_NUTS3_Regionen.ods"
sheet_name_workforce = "LSE2022_ARB"
filename_population = "Bev_NUTS-Regionen_Zeitreihe.ods"
sheet_name_population = "Zeitreihe"
filename_farming = "LuFBetriebeNachFlaechennutzung2020nBL.ods"
sheet_name_farming = "Flaechennutzung"

# Construct file paths for input data by combining folder and filenames
file_path_workforce = os.path.join(raw_data_folder, filename_workforce)
file_path_population = os.path.join(raw_data_folder, filename_population)
file_path_farming = os.path.join(raw_data_folder, filename_farming)

# Output filenames for Stage 1 with the dynamic date
filename_energy_balance_output = f'AT_Elektrische_Energie_aggregated_{date}.xlsx'
filename_workforce_output = f"AT_Beschäftigte_NUTS_{date}.xlsx"
filename_population_output = f"AT_Bevölkerung_NUTS_{date}.xlsx"
filename_farming_output = f"AT_Flächennutzung_Land_Forstwirtschaft_{date}.xlsx"

#####################################################################################################
#############  STAGE 2 #############
# I/O data for Stage 2, referencing Stage 1 output folder
data_path_energy = os.path.join(stage1_output_folder, filename_energy_balance_output)
data_path_workforce = os.path.join(stage1_output_folder, filename_workforce_output)
data_path_population = os.path.join(stage1_output_folder, filename_population_output)
data_path_farming = os.path.join(stage1_output_folder, filename_farming_output)
file_path_stats = os.path.join(stage1_input_folder, "Gemeindestatistik_NUTS_level.xlsx")

#####################################################################################################
#############  STAGE 3 #############
# Load profiles folder path
folder_path_profiles = os.path.join(project_root, "Data_Input", "LoadProfiles", "APCS_LoadProfiles")
input_folder_path = stage2_output_folder
output_directory_time_series = os.path.join(stage3_output_folder, "Time_Series")
output_directory_aggregated_data = os.path.join(stage3_output_folder, "Aggregated_Data")

# Mapping and configuration files
file_path_mapping = os.path.join(project_root, "Data_Input", "Mapping_sector_loadprofile.xlsx")
weight_file_path = os.path.join(project_root, "Data_Input", "CONFIG_TOOL.xlsx")
path_sector_weights = os.path.join(project_root, "Data_Input", "CONFIG_SECTOR_WEIGHTS.xlsx")

excel_profiles_path = os.path.join(project_root, "Data_Input", "alternative_load_profiles.xlsx")

#####################################################################################################
#############  LEGO2NUTS assignment #############
API_KEY = '34e634f6ab5347e3a13f7208e923709c'

# Path for LEGO input files within the Data_Input folder
lego_input_folder = os.path.join(project_root, "Data_Input", "LEGO")
input_lego = os.path.join(lego_input_folder, "09_Network_Businfo.xlsx")
input_demand_file = os.path.join(lego_input_folder, "03_Demand.xlsx")
input_at_nuts = os.path.join(project_root, "Data_Input", "AT_NUTS_levels.xlsx")

folder_path_output = os.path.join(project_root, "Data_Output", "LEGO2NUTS_assignment")

if use_nuts_assignment:
    output_lego2nuts = os.path.join(folder_path_output, f"LEGO_nodes_NUTS_assignment_{date}.xlsx")
else:
    output_lego2nuts = os.path.join(folder_path_output, f"LEGO_nodes_NUTS_assignment_base_V1.xlsx")


#####################################################################################################
#############  STAGE 4 #############
# folder_path = output_directory_time_series

folder_path = output_directory_aggregated_data

output_file_lego2nuts = os.path.join(folder_path_output, f"LEGO_nodes_NUTS_assignment_{date}.xlsx")

#####################################################################################################
# Function to ensure all directories exist
def ensure_directories_exist(*dirs):
    for directory in dirs:
        os.makedirs(directory, exist_ok=True)

# Call the function to create all necessary directories
ensure_directories_exist(
    folder_log,
    stage1_input_folder,
    raw_data_folder,
    stage1_output_folder,
    stage2_output_folder,
    stage3_output_folder,
    stage4_output_folder,
    output_directory_time_series,
    output_directory_aggregated_data,
    folder_path_output,
)
