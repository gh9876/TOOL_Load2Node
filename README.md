# Tool: Load2Node

**Tool: Load2Node** is a modular, open-source Python tool developed to process and distribute energy load data for use in simulation programs such as LEGO. The tool converts raw input datasets (e.g., energy balance sheets, employment statistics, population data, agricultural land use data) into structured output that can be directly used for simulation purposes. It also integrates geographical information through NUTS (Nomenclature of Territorial Units for Statistics) assignment via the OpenCage Geocoder API.

## Overview

The tool is designed with a clear four-stage architecture that enables step-by-step data processing and facilitates the validation of intermediate results. This modular design not only aids in troubleshooting but also ensures that each component (or "Stage") can be executed independently.

## Features

- **Modular Architecture:**  
  Each stage is implemented in its own Python script, allowing for independent execution and easier debugging.
  
- **Data Preprocessing:**  
  Cleans and standardizes raw data from multiple sources (LEGO network, energy balance sheets, employment, population, and agricultural data).  
  Exports aggregated datasets for subsequent processing.

- **Energy Allocation:**  
  Distributes energy consumption to NUTS3 regions using sectoral weighting and employment data.  
  Validates that the distributed values match the original energy consumption.

- **Time Series Generation:**  
  Creates synthetic load profiles from standard or custom load data, resampling to an hourly resolution.  
  Handles leap years by removing February 29th to maintain consistent 8760-hour time series.

- **LEGO Node Distribution:**  
  Maps aggregated time series data to LEGO nodes based on NUTS assignments, dividing data proportionally if a node spans multiple regions.  
  Applies adjustments for transport losses.

- **NUTS Code Assignment & Validation:**  
  Uses the OpenCage API to assign NUTS1, NUTS2, and NUTS3 codes to nodes, validates them against expected country codes, and allows manual reassignment for corrections.

- **Logging & Validation:**  
  Detailed logging is available for each stage via a custom logger (`logging_setup.py`), and multiple validation steps ensure data consistency throughout the process.

### Processing Stages

1. **Stage 1: Data Preprocessing**  
   - Loads and processes input files including energy balance sheets, employment statistics, population data, and land use information.
   - Cleans and standardizes data, ensuring that the output is consistent and ready for further processing.
   - Exports aggregated datasets for use in later stages.

2. **Stage 2: Energy Allocation**  
   - Distributes the energy consumption data on a federal-state basis.
   - Utilizes sectoral weights (defined in an external Excel file) and NUTS classifications to proportionally allocate energy demand.
   - Validates the results by comparing allocated energy with original consumption values.

3. **Stage 3: Time Series Creation**  
   - Generates synthetic load profiles based on standard or user-defined profiles.
   - Resamples data to hourly resolution, handling leap years (removing February 29th) to ensure each year has 8760 data points.
   - Aggregates and exports time series data for each federal state.

4. **Stage 4: Distribution to LEGO Nodes**  
   - Distributes the created hourly time series to LEGO nodes using mapping data.
   - Handles cases where a LEGO node is associated with multiple NUTS3 regions by proportionally dividing the time series.
   - Applies corrections (e.g., transport loss adjustments) and validates the final distribution.

Additionally, the tool includes a dedicated module for assigning and validating NUTS levels for each network node. Using the OpenCage API, the tool performs reverse geocoding to assign NUTS1, NUTS2, and NUTS3 codes. It then validates these codes to ensure they match the expected country (e.g., Austria, with the prefix "AT") and prompts for manual intervention if discrepancies occur.

## Development Environment

The tool is developed in **Python** using **Visual Studio Code (VS Code)** as the integrated development environment (IDE). VS Code was chosen for its cross-platform capabilities, extensive extension ecosystem, integrated Git support, and free availability.  
Primary dependencies include:  
- **Pandas:** For efficient data processing using DataFrames.  
- **NumPy:** For fast numerical computations and array manipulations.

The tool’s open-source nature ensures that it can be modified and extended without licensing costs, allowing for future enhancements.

## Repository

The complete source code is hosted on GitHub and is available at:  
[https://github.com/gh9876/Tool_Load2Node](https://github.com/gh9876/Tool_Load2Node)

### Prerequisites
- **Python 3.x**
- **Git** (install via Homebrew on macOS if necessary)
- Other dependencies listed in `requirements.txt`

## Installation

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/gh9876/Tool_Load2Node.git
