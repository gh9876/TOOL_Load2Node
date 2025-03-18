import os
import sys
import time
import subprocess
from config_tool_V1 import *
from install_requirements import *

# Run the requirements check at the beginning of the script
check_and_install_requirements()
print("=====================================")
    
# Record the total start time
start_time = time.time()

# Introduction
print()
print("============================================================================================================")
print("                   Tool: Load2Node                                                                          ")
print("============================================================================================================")
print("This tool is designed to process and allocate energy demands, and develop individual consumption time series")
print("based on input data from Statistik Austria. The scope of action includes the region of Austria.\n")
print("Stages include data preprocessing, energy allocation, time series creation and distribution to LEGO nodes.\n")
print("Select from various stages or combinations to run:\n")
print("1. Data Preprocessing")
print("2. Energy Allocation")
print("3. Time Series Creation")
print("4. Distribution to LEGO Nodes\n")
print("You may also select preset combinations to streamline")
print("multiple stages in a single run.\n")
print("============================================================================================================")

# Get the current working directory dynamically
current_dir = os.getcwd()

# List of Python scripts representing each stage of the tool, located in the current working directory
scripts = [
    os.path.join(current_dir, "stage1_data_preprocessing_V1.py"),
    os.path.join(current_dir, "stage2_distribution_energy_demand_V1.py"),
    os.path.join(current_dir, "stage3_create_time_series_V1.py"),
    os.path.join(current_dir, "stage4_distribution2nodes_V1.py")
]

# Descriptive names for each stage
stage_names = [
    "Data Preprocessing",
    "Energy Allocation",
    "Time Series Creation",
    "Distribution to LEGO Nodes"
]

# Define preset combinations for ease of selection
combinations = {
    "1": [0],               # Run only Stage 1
    "2": [1],               # Run only Stage 2
    "3": [2],               # Run only Stage 3
    "4": [3],               # Run only Stage 4
    "5": [1, 2],            # Run Stages 2 + 3
    "6": [2, 3],            # Run Stages 3 + 4
    "7": [1, 2, 3],         # Run Stages 2 to 4
    "8": [0, 1, 2, 3]       # Run all stages
}

# Display options to the user
print("Select stages to run:")
print("1: Run only 'Data Preprocessing'")
print("2: Run only 'Energy Allocation'")
print("3: Run only 'Time Series Creation'")
print("4: Run only 'Distribution to LEGO Nodes'")
print("5: Run Stages 2 + 3")
print("6: Run Stages 3 + 4")
print("7: Run Stages 2 to 4")
print("8: Run all stages")
print()
selection = input("Enter the number of your choice: ")

# Validate selection
if selection not in combinations:
    print("Invalid selection. Please restart and choose a valid option.")
    sys.exit(1)

# Display script paths being executed for verification
print("\nScript paths being executed:")
for script in scripts:
    print(script)

# Run the selected stages
for index in combinations[selection]:
    stage = stage_names[index]
    script = scripts[index]
    print(f"\n=== Starting Stage: {stage} ===")
    print(f"Running script: {script}\n")
    
    # Record the start time of the current stage
    stage_start_time = time.time()
    
    try:
        # Run the Python script and display its output directly to the terminal
        result = subprocess.run(
            [sys.executable, script],
            check=True,
            stdout=sys.stdout,
            stderr=sys.stderr
        )
        # Calculate the runtime for the current stage
        stage_end_time = time.time()
        elapsed_time = stage_end_time - stage_start_time
        hours, rem = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(rem, 60)
        print(f"\n=== Finished Stage: {stage} successfully in:{int(hours)}h {int(minutes)}m {int(seconds)}s\n ===")
    except subprocess.CalledProcessError as e:
        print(f"\n*** Error during Stage: {stage} ***")
        print(f"Script {script} failed with error: {e}")
        break

# Calculate the total runtime
end_time = time.time()
elapsed_time = end_time - start_time
hours, rem = divmod(elapsed_time, 3600)
minutes, seconds = divmod(rem, 60)

final_message = f"\nSelected stages completed in: {int(hours)}h {int(minutes)}m {int(seconds)}s\n"
print(final_message)