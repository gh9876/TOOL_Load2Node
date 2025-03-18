"""
Module: distribution_functions.py
Description: 
    This module provides functions for distributing time series data (load profiles) among LEGO nodes
    based on mapping data. The functions include utilities to handle nodes with multiple NUTS3 levels,
    validate the distribution results, and distribute the demand for each federal state of Austria.
    For each state (e.g., Burgenland, Niederösterreich, Oberösterreich, etc.), similar functions are implemented
    with state‐specific filtering based on NUTS2 codes. The detailed documentation is provided for the 
    Burgenland distribution function; the other functions follow a similar structure.
"""

import os
import numpy as np
import pandas as pd
from logging_function import setup_stage_logger

logger = setup_stage_logger("distribution_functions")

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def find_and_split_nodes_with_multiple_nuts_levels(mapping_data):
    """
    Finds and splits nodes with multiple associated NUTS3 levels.

    This function splits the 'NUTS3' column in the mapping data by commas, handles NaN values,
    and then groups the data by 'LEGO ID'. It filters out nodes with only a single NUTS3 level and 
    splits nodes with multiple levels into individual columns.

    Parameters:
    - mapping_data (DataFrame): DataFrame containing mapping information with at least 'LEGO ID' and 'NUTS3' columns.

    Returns:
    - DataFrame: Processed DataFrame where nodes with multiple NUTS3 levels are split into separate columns.
    """
    mapping_data_split = mapping_data.assign(NUTS3=mapping_data['NUTS3'].str.split(',')).explode('NUTS3')
    mapping_data_split = mapping_data_split[mapping_data_split['NUTS3'].notna()]
    mapping_data_split['NUTS3'] = mapping_data_split['NUTS3'].str.strip()
    mapping_data_split = mapping_data_split[mapping_data_split['NUTS3'] != '']
    nodes_with_multiple_nuts3 = mapping_data_split.groupby('LEGO ID')['NUTS3'].apply(list).reset_index()
    nodes_with_multiple_nuts3 = nodes_with_multiple_nuts3[nodes_with_multiple_nuts3['NUTS3'].apply(lambda x: len(x) > 1)]
    if nodes_with_multiple_nuts3.empty:
        logger.info("No nodes with multiple NUTS3 levels found.")
        return pd.DataFrame()
    max_nuts3 = int(nodes_with_multiple_nuts3['NUTS3'].apply(len).max())
    for i in range(max_nuts3):
        nodes_with_multiple_nuts3[f'NUTS3_{i+1}'] = nodes_with_multiple_nuts3['NUTS3'].apply(lambda x: x[i] if i < len(x) else None)
    nodes_with_multiple_nuts3 = nodes_with_multiple_nuts3.drop(columns=['NUTS3'])
    return nodes_with_multiple_nuts3

def validate_lego_node_count(mapping_data, distributed_data):
    """
    Validates that the number of LEGO nodes in the mapping data matches those in the distributed result.

    Parameters:
    - mapping_data (DataFrame): The mapping DataFrame containing LEGO node information.
    - distributed_data (DataFrame): The distributed result DataFrame with LEGO node distribution.

    Returns:
    - None: Logs validation results and any missing or extra LEGO node IDs.
    """
    expected_lego_nodes = mapping_data['LEGO ID'].nunique()
    actual_lego_nodes = distributed_data['LEGO ID'].nunique()
    logger.info(f"Number of LEGO nodes from mapping data: {expected_lego_nodes}")
    logger.info(f"Actual number of LEGO nodes in distributed result: {actual_lego_nodes}")
    if expected_lego_nodes == actual_lego_nodes:
        logger.info("Validation Passed: All LEGO nodes are present in the distributed result.")
    else:
        logger.info("Validation Failed: Some LEGO nodes are missing or extra nodes are present in the distributed result.")
        missing_lego_ids = set(mapping_data['LEGO ID']) - set(distributed_data['LEGO ID'])
        extra_lego_ids = set(distributed_data['LEGO ID']) - set(mapping_data['LEGO ID'])
        if missing_lego_ids:
            logger.info(f"Missing LEGO IDs: {missing_lego_ids}")
        if extra_lego_ids:
            logger.info(f"Extra LEGO IDs: {extra_lego_ids}")

def validate_distribution_result(federal_state_name, total_demand_input, processed_time_series, distributed_demand_tracker):
    """
    Validates and debugs the distribution results for a federal state.

    It compares the total demand from the input time series with the total demand distributed 
    to LEGO nodes and logs the differences along with detailed tracking of demand per node.

    Parameters:
    - federal_state_name (str): Name of the federal state.
    - total_demand_input (float): Total demand from the input time series.
    - processed_time_series (DataFrame): DataFrame containing the distributed time series data.
    - distributed_demand_tracker (dict): Dictionary tracking the distributed demand per LEGO node.

    Returns:
    - None: Logs the validation and debugging information.
    """
    total_sum_distributed_time_series = processed_time_series.iloc[:, 1:].sum().sum()
    total_sum_distributed_time_series = round(total_sum_distributed_time_series, 6)
    total_demand_input = round(total_demand_input, 6)
    diff_demand = total_demand_input - total_sum_distributed_time_series
    logger.info("--------------------------------------------------------------------------------")
    logger.info(f"Result validation: {federal_state_name}")
    logger.info(f"Total demand from input time series: {total_demand_input}")
    logger.info(f"Total demand distributed to LEGO nodes: {total_sum_distributed_time_series}")
    logger.info(f"Difference between input demand and distributed demand: {diff_demand}")
    logger.info(f"Demand tracked for each LEGO node: {distributed_demand_tracker}")
    total_tracked_demand = round(sum(distributed_demand_tracker.values()), 6)
    logger.info(f"Total tracked distributed demand: {total_tracked_demand}")
    logger.info("")

# =============================================================================
# DISTRIBUTION FUNCTIONS FOR INDIVIDUAL FEDERAL STATES
# =============================================================================

def distribute_nuts_level_data_burgenland(nuts_data, mapping_data, total_demand_input, federal_state_name, output_folder, filename_result_distribution):
    """
    Distributes time series data for Burgenland to LEGO nodes.

    This function processes the mapping data filtered for Burgenland (NUTS2 code 'AT11'),
    divides the relevant time series data among LEGO nodes (accounting for shared nodes), and
    validates the distribution. Finally, the results are saved to an Excel file.

    Parameters:
    - nuts_data (DataFrame): DataFrame containing time series data indexed by NUTS-ID.
    - mapping_data (DataFrame): Mapping DataFrame with LEGO node assignments.
    - total_demand_input (float): The total demand from the input time series.
    - federal_state_name (str): The name of the federal state (should be 'Burgenland').
    - output_folder (str): Directory to save the resulting Excel file.
    - filename_result_distribution (str): Filename for the distribution result.

    Returns:
    - float: Total distributed demand for validation.
    """
    if federal_state_name != 'Burgenland':
        logger.info(f"The federal state '{federal_state_name}' does not match 'Burgenland'. No action taken.")
        return None
    mapping_data['NUTS2'] = mapping_data['NUTS2'].str.strip()
    mapping_data_burgenland = mapping_data[mapping_data['NUTS2'] == 'AT11']
    pd.set_option('display.max_rows', None)
    logger.info(f"Mapping data for {federal_state_name}:\n{mapping_data_burgenland}")
    processed_data = []
    distributed_demand_tracker = {}
    total_distributed_demand = 0
    for _, row in mapping_data_burgenland.iterrows():
        lego_id = row['LEGO ID']
        nuts_ids = [nuts_id.strip() for nuts_id in row['NUTS3'].split(',')] if isinstance(row['NUTS3'], str) else [row['NUTS3']]
        lego_combined_data = None
        for nuts_id in nuts_ids:
            relevant_data = nuts_data[nuts_data['NUTS-ID'].str.strip() == nuts_id.strip()]
            if not relevant_data.empty:
                relevant_data.iloc[:, 1:] = relevant_data.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')
                lego_nodes_sharing_nuts3 = mapping_data_burgenland[mapping_data_burgenland['NUTS3'].str.contains(nuts_id, na=False)]
                num_sharing_lego_nodes = len(lego_nodes_sharing_nuts3)
                divided_data = relevant_data.iloc[:, 1:] / num_sharing_lego_nodes
                lego_combined_data = divided_data if lego_combined_data is None else lego_combined_data + divided_data.values
        if lego_combined_data is not None:
            lego_sum = lego_combined_data.sum().sum()
            total_distributed_demand += lego_sum
            logger.info(f"Final shape of combined_data for LEGO ID {lego_id}: {lego_combined_data.shape}, Sum of distributed data: {lego_sum}")
            processed_data.append(pd.Series(
                [lego_id] + lego_combined_data.sum(axis=0).tolist(),
                index=['LEGO ID'] + list(nuts_data.columns[1:])
            ))
            distributed_demand_tracker[lego_id] = lego_sum
        else:
            logger.info(f"Error: No data was combined for LEGO ID {lego_id}. Skipping this LEGO ID.")
    processed_time_series = pd.concat(processed_data, axis=1).T if processed_data else pd.DataFrame()
    processed_time_series.iloc[:, 1:] = processed_time_series.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')
    validate_lego_node_count(mapping_data_burgenland, processed_time_series)
    validate_distribution_result(federal_state_name, total_demand_input, processed_time_series, distributed_demand_tracker)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    output_path = os.path.join(output_folder, filename_result_distribution)
    processed_time_series.to_excel(output_path, index=False)
    logger.info(f"Data saved to: {output_path}")
    return total_distributed_demand

def distribute_nuts_level_data_niederoesterreich(nuts_data, mapping_data, total_demand_input, federal_state_name, output_folder, filename_result_distribution):
    if federal_state_name != 'Niederösterreich':
        logger.info(f"The federal state '{federal_state_name}' does not match 'Niederösterreich'. No action taken.")
        return None
    mapping_data['NUTS2'] = mapping_data['NUTS2'].str.strip()
    mapping_data_niederösterreich = mapping_data[mapping_data['NUTS2'] == 'AT12']
    logger.info(f"Mapping data for {federal_state_name}:\n{mapping_data_niederösterreich}")
    processed_data = []
    distributed_demand_tracker = {}
    total_distributed_demand = 0
    for _, row in mapping_data_niederösterreich.iterrows():
        lego_id = row['LEGO ID']
        nuts_ids = [nuts_id.strip() for nuts_id in row['NUTS3'].split(',')] if isinstance(row['NUTS3'], str) else [row['NUTS3']]
        lego_combined_data = None
        for nuts_id in nuts_ids:
            relevant_data = nuts_data[nuts_data['NUTS-ID'].str.strip() == nuts_id.strip()]
            if not relevant_data.empty:
                relevant_data.iloc[:, 1:] = relevant_data.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')
                lego_nodes_sharing_nuts3 = mapping_data_niederösterreich[mapping_data_niederösterreich['NUTS3'].str.contains(nuts_id, na=False)]
                num_sharing_lego_nodes = len(lego_nodes_sharing_nuts3)
                divided_data = relevant_data.iloc[:, 1:] / num_sharing_lego_nodes
                lego_combined_data = divided_data if lego_combined_data is None else lego_combined_data + divided_data.values
        if lego_combined_data is not None:
            lego_sum = lego_combined_data.sum().sum()
            total_distributed_demand += lego_sum
            logger.info(f"Final shape of combined_data for LEGO ID {lego_id}: {lego_combined_data.shape}, Sum of distributed data: {lego_sum}")
            processed_data.append(pd.Series(
                [lego_id] + lego_combined_data.sum(axis=0).tolist(),
                index=['LEGO ID'] + list(nuts_data.columns[1:])
            ))
            distributed_demand_tracker[lego_id] = lego_sum
        else:
            logger.info(f"Error: No data was combined for LEGO ID {lego_id}. Skipping this LEGO ID.")
    processed_time_series = pd.concat(processed_data, axis=1).T if processed_data else pd.DataFrame()
    processed_time_series.iloc[:, 1:] = processed_time_series.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')
    validate_lego_node_count(mapping_data_niederösterreich, processed_time_series)
    validate_distribution_result(federal_state_name, total_demand_input, processed_time_series, distributed_demand_tracker)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    output_path = os.path.join(output_folder, filename_result_distribution)
    processed_time_series.to_excel(output_path, index=False)
    logger.info(f"Data saved to: {output_path}")
    return total_distributed_demand

def distribute_nuts_level_data_oberoesterreich(nuts_data, mapping_data, total_demand_input, federal_state_name, output_folder, filename_result_distribution):
    mapping_data['NUTS2'] = mapping_data['NUTS2'].str.strip()
    mapping_data_oberoesterreich = mapping_data[mapping_data['NUTS2'] == 'AT31']
    pd.set_option('display.max_rows', None)
    logger.info(f"Mapping data {federal_state_name}:\n{mapping_data_oberoesterreich}")
    processed_data = []
    distributed_demand_tracker = {}
    total_distributed_demand = 0
    for _, row in mapping_data_oberoesterreich.iterrows():
        lego_id = row['LEGO ID']
        nuts_ids = [nuts_id.strip() for nuts_id in row['NUTS3'].split(',')] if isinstance(row['NUTS3'], str) else [row['NUTS3']]
        lego_combined_data = None
        for nuts_id in nuts_ids:
            relevant_data = nuts_data[nuts_data['NUTS-ID'].str.strip() == nuts_id.strip()]
            if not relevant_data.empty:
                relevant_data.iloc[:, 1:] = relevant_data.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')
                lego_nodes_sharing_nuts3 = mapping_data_oberoesterreich[mapping_data_oberoesterreich['NUTS3'].str.contains(nuts_id, na=False)]
                num_sharing_lego_nodes = len(lego_nodes_sharing_nuts3)
                divided_data = relevant_data.iloc[:, 1:] / num_sharing_lego_nodes
                lego_combined_data = divided_data if lego_combined_data is None else lego_combined_data + divided_data.values
        if lego_combined_data is not None:
            lego_sum = lego_combined_data.sum().sum()
            total_distributed_demand += lego_sum
            processed_data.append(pd.Series(
                [lego_id] + lego_combined_data.sum(axis=0).tolist(),
                index=['LEGO ID'] + list(nuts_data.columns[1:])
            ))
            distributed_demand_tracker[lego_id] = lego_sum
        else:
            logger.info(f"Error: No data was combined for LEGO ID {lego_id}. Skipping this LEGO ID.")
    processed_time_series = pd.concat(processed_data, axis=1).T if processed_data else pd.DataFrame()
    processed_time_series.iloc[:, 1:] = processed_time_series.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')
    validate_lego_node_count(mapping_data_oberoesterreich, processed_time_series)
    validate_distribution_result(federal_state_name, total_demand_input, processed_time_series, distributed_demand_tracker)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    output_path = os.path.join(output_folder, filename_result_distribution)
    processed_time_series.to_excel(output_path, index=False)
    logger.info(f"Data saved to: {output_path}")
    return total_distributed_demand

def distribute_nuts_level_data_tirol(nuts_data, mapping_data, total_demand_input, federal_state_name, output_folder, filename_result_distribution):
    mapping_data['NUTS2'] = mapping_data['NUTS2'].str.strip()
    mapping_data_tirol = mapping_data[mapping_data['NUTS2'] == 'AT33']
    pd.set_option('display.max_rows', None)
    logger.info(f"Mapping data {federal_state_name}:\n{mapping_data_tirol}\n")
    nodes_with_multiple_nuts3 = find_and_split_nodes_with_multiple_nuts_levels(mapping_data_tirol)
    processed_data = []
    distributed_demand_tracker = {}
    total_distributed_demand = 0
    for _, row in nodes_with_multiple_nuts3.iterrows():
        lego_id = row['LEGO ID']
        nuts_ids = [row.get(f'NUTS3_{i+1}') for i in range(len(row) - 1) if row.get(f'NUTS3_{i+1}')]
        lego_combined_data = None
        for nuts_id in nuts_ids:
            relevant_data = nuts_data[nuts_data['NUTS-ID'].str.strip() == nuts_id.strip()]
            if not relevant_data.empty:
                relevant_data.iloc[:, 1:] = relevant_data.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')
                lego_nodes_sharing_nuts3 = mapping_data_tirol[mapping_data_tirol['NUTS3'].str.contains(nuts_id, na=False)]
                num_sharing_lego_nodes = len(lego_nodes_sharing_nuts3)
                divided_data = relevant_data.iloc[:, 1:] / num_sharing_lego_nodes
                lego_combined_data = divided_data if lego_combined_data is None else lego_combined_data + divided_data.values
        if lego_combined_data is not None:
            lego_sum = lego_combined_data.sum().sum()
            total_distributed_demand += lego_sum
            processed_data.append(pd.Series(
                [lego_id] + lego_combined_data.sum(axis=0).tolist(),
                index=['LEGO ID'] + list(nuts_data.columns[1:])
            ))
            distributed_demand_tracker[lego_id] = lego_sum
        else:
            logger.info(f"Error: No data was combined for LEGO ID {lego_id}. Skipping this LEGO ID.")
    for nuts_id in mapping_data_tirol['NUTS3'].unique():
        if nuts_id not in distributed_demand_tracker:
            relevant_data = nuts_data[nuts_data['NUTS-ID'].str.strip() == nuts_id.strip()]
            if not relevant_data.empty:
                relevant_data.iloc[:, 1:] = relevant_data.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')
                lego_nodes = mapping_data_tirol[mapping_data_tirol['NUTS3'] == nuts_id]['LEGO ID'].unique()
                num_lego_nodes = len(lego_nodes)
                divided_data = relevant_data.iloc[:, 1:].values / num_lego_nodes
                divided_data = np.atleast_2d(divided_data.sum(axis=0))
                for lego_id in lego_nodes:
                    processed_data.append(pd.Series(
                        [lego_id] + divided_data.flatten().tolist(),
                        index=['LEGO ID'] + list(nuts_data.columns[1:])
                    ))
                    lego_sum = divided_data.sum()
                    total_distributed_demand += lego_sum
                    distributed_demand_tracker[lego_id] = lego_sum
    processed_time_series = pd.concat(processed_data, axis=1).T if processed_data else pd.DataFrame()
    processed_time_series.iloc[:, 1:] = processed_time_series.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')
    validate_lego_node_count(mapping_data_tirol, processed_time_series)
    validate_distribution_result(federal_state_name, total_demand_input, processed_time_series, distributed_demand_tracker)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    output_path = os.path.join(output_folder, filename_result_distribution)
    processed_time_series.to_excel(output_path, index=False)
    logger.info(f"Data saved to: {output_path}")
    return total_distributed_demand

def distribute_nuts_level_data_kaernten(nuts_data, mapping_data, total_demand_input, federal_state_name, output_folder, filename_result_distribution):
    """
    Distributes time series data for Kärnten to LEGO nodes and saves the results to an Excel file.
    """
    if federal_state_name != 'Kärnten':
        logger.info(f"The federal state '{federal_state_name}' does not match 'Kärnten'. No action taken.")
        return None
    mapping_data_kaernten = mapping_data[mapping_data['NUTS2'].str.strip() == 'AT21']
    logger.info(f"Mapping data for {federal_state_name}:\n{mapping_data_kaernten}")
    processed_data = []
    distributed_demand_tracker = {}
    total_distributed_demand = 0
    for _, row in mapping_data_kaernten.iterrows():
        lego_id = row['LEGO ID']
        nuts_ids = [nuts_id.strip() for nuts_id in row['NUTS3'].split(',')] if isinstance(row['NUTS3'], str) else [row['NUTS3']]
        lego_combined_data = None
        for nuts_id in nuts_ids:
            relevant_data = nuts_data[nuts_data['NUTS-ID'].str.strip() == nuts_id]
            if not relevant_data.empty:
                relevant_data.iloc[:, 1:] = relevant_data.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')
                lego_nodes_sharing_nuts3 = mapping_data_kaernten[mapping_data_kaernten['NUTS3'].str.contains(nuts_id, na=False)]
                num_sharing_lego_nodes = len(lego_nodes_sharing_nuts3)
                divided_data = relevant_data.iloc[:, 1:] / num_sharing_lego_nodes
                lego_combined_data = divided_data if lego_combined_data is None else lego_combined_data + divided_data.values
        if lego_combined_data is not None:
            lego_sum = lego_combined_data.sum().sum()
            total_distributed_demand += lego_sum
            processed_data.append(pd.Series(
                [lego_id] + lego_combined_data.sum(axis=0).tolist(),
                index=['LEGO ID'] + list(nuts_data.columns[1:])
            ))
            distributed_demand_tracker[lego_id] = lego_sum
        else:
            logger.info(f"Error: No data was combined for LEGO ID {lego_id}. Skipping this LEGO ID.")
    processed_time_series = pd.concat(processed_data, axis=1).T if processed_data else pd.DataFrame()
    processed_time_series.iloc[:, 1:] = processed_time_series.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')
    validate_lego_node_count(mapping_data_kaernten, processed_time_series)
    validate_distribution_result(federal_state_name, total_demand_input, processed_time_series, distributed_demand_tracker)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    output_path = os.path.join(output_folder, filename_result_distribution)
    processed_time_series.to_excel(output_path, index=False)
    logger.info(f"Data saved to: {output_path}")
    return total_distributed_demand

def distribute_nuts_level_data_salzburg(nuts_data, mapping_data, total_demand_input, federal_state_name, output_folder, filename_result_distribution):
    """
    Distributes time series data for Salzburg to LEGO nodes and saves the results to an Excel file.
    """
    if federal_state_name != 'Salzburg':
        logger.info(f"The federal state '{federal_state_name}' does not match 'Salzburg'. No action taken.")
        return None
    mapping_data_salzburg = mapping_data[mapping_data['NUTS2'].str.strip() == 'AT32']
    logger.info(f"Mapping data for {federal_state_name}:\n{mapping_data_salzburg}")
    processed_data = []
    distributed_demand_tracker = {}
    total_distributed_demand = 0
    for _, row in mapping_data_salzburg.iterrows():
        lego_id = row['LEGO ID']
        nuts_ids = [nuts_id.strip() for nuts_id in row['NUTS3'].split(',')] if isinstance(row['NUTS3'], str) else [row['NUTS3']]
        lego_combined_data = None
        for nuts_id in nuts_ids:
            relevant_data = nuts_data[nuts_data['NUTS-ID'].str.strip() == nuts_id]
            if not relevant_data.empty:
                relevant_data.iloc[:, 1:] = relevant_data.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')
                lego_nodes_sharing_nuts3 = mapping_data_salzburg[mapping_data_salzburg['NUTS3'].str.contains(nuts_id, na=False)]
                num_sharing_lego_nodes = len(lego_nodes_sharing_nuts3)
                divided_data = relevant_data.iloc[:, 1:] / num_sharing_lego_nodes
                lego_combined_data = divided_data if lego_combined_data is None else lego_combined_data + divided_data.values
        if lego_combined_data is not None:
            lego_sum = lego_combined_data.sum().sum()
            total_distributed_demand += lego_sum
            processed_data.append(pd.Series(
                [lego_id] + lego_combined_data.sum(axis=0).tolist(),
                index=['LEGO ID'] + list(nuts_data.columns[1:])
            ))
            distributed_demand_tracker[lego_id] = lego_sum
        else:
            logger.info(f"Error: No data was combined for LEGO ID {lego_id}. Skipping this LEGO ID.")
    processed_time_series = pd.concat(processed_data, axis=1).T if processed_data else pd.DataFrame()
    processed_time_series.iloc[:, 1:] = processed_time_series.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')
    validate_lego_node_count(mapping_data_salzburg, processed_time_series)
    validate_distribution_result(federal_state_name, total_demand_input, processed_time_series, distributed_demand_tracker)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    output_path = os.path.join(output_folder, filename_result_distribution)
    processed_time_series.to_excel(output_path, index=False)
    logger.info(f"Data saved to: {output_path}")
    return total_distributed_demand

def distribute_nuts_level_data_vorarlberg(nuts_data, mapping_data, total_demand_input, federal_state_name, output_folder, filename_result_distribution):
    """
    Distributes time series data for Vorarlberg across LEGO nodes and saves the results to an Excel file.
    """
    if federal_state_name != 'Vorarlberg':
        logger.info(f"The federal state '{federal_state_name}' does not match 'Vorarlberg'. No action taken.")
        return None
    mapping_data_vorarlberg = mapping_data[mapping_data['NUTS2'].str.strip() == 'AT34']
    logger.info(f"Mapping data for {federal_state_name}:\n{mapping_data_vorarlberg}")
    processed_data = []
    distributed_demand_tracker = {}
    total_distributed_demand = 0
    for _, row in mapping_data_vorarlberg.iterrows():
        lego_id = row['LEGO ID']
        nuts_ids = [nuts_id.strip() for nuts_id in row['NUTS3'].split(',')] if isinstance(row['NUTS3'], str) else [row['NUTS3']]
        lego_combined_data = None
        for nuts_id in nuts_ids:
            relevant_data = nuts_data[nuts_data['NUTS-ID'].str.strip() == nuts_id]
            if not relevant_data.empty:
                relevant_data.iloc[:, 1:] = relevant_data.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')
                lego_nodes_sharing_nuts3 = mapping_data_vorarlberg[mapping_data_vorarlberg['NUTS3'].str.contains(nuts_id, na=False)]
                num_sharing_lego_nodes = len(lego_nodes_sharing_nuts3)
                divided_data = relevant_data.iloc[:, 1:] / num_sharing_lego_nodes
                lego_combined_data = divided_data if lego_combined_data is None else lego_combined_data + divided_data.values
        if lego_combined_data is not None:
            lego_sum = lego_combined_data.sum().sum()
            total_distributed_demand += lego_sum
            processed_data.append(pd.Series(
                [lego_id] + lego_combined_data.sum(axis=0).tolist(),
                index=['LEGO ID'] + list(nuts_data.columns[1:])
            ))
            distributed_demand_tracker[lego_id] = lego_sum
        else:
            logger.info(f"Error: No data was combined for LEGO ID {lego_id}. Skipping this LEGO ID.")
    processed_time_series = pd.concat(processed_data, axis=1).T if processed_data else pd.DataFrame()
    processed_time_series.iloc[:, 1:] = processed_time_series.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')
    validate_lego_node_count(mapping_data_vorarlberg, processed_time_series)
    validate_distribution_result(federal_state_name, total_demand_input, processed_time_series, distributed_demand_tracker)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    output_path = os.path.join(output_folder, filename_result_distribution)
    processed_time_series.to_excel(output_path, index=False)
    logger.info(f"Data saved to: {output_path}")
    return total_distributed_demand

def distribute_nuts_level_data_wien(nuts_data, mapping_data, total_demand_input, federal_state_name, output_folder, filename_result_distribution):
    """
    Distributes time series data for Wien across LEGO nodes and saves the results to an Excel file.
    """
    if federal_state_name != 'Wien':
        logger.info(f"The federal state '{federal_state_name}' does not match 'Wien'. No action taken.")
        return None
    mapping_data_wien = mapping_data[mapping_data['NUTS2'].str.strip() == 'AT13']
    logger.info(f"Mapping data for {federal_state_name}:\n{mapping_data_wien}")
    processed_data = []
    distributed_demand_tracker = {}
    total_distributed_demand = 0
    for _, row in mapping_data_wien.iterrows():
        lego_id = row['LEGO ID']
        nuts_ids = [nuts_id.strip() for nuts_id in row['NUTS3'].split(',')] if isinstance(row['NUTS3'], str) else [row['NUTS3']]
        lego_combined_data = None
        for nuts_id in nuts_ids:
            relevant_data = nuts_data[nuts_data['NUTS-ID'].str.strip() == nuts_id.strip()]
            if not relevant_data.empty:
                relevant_data.iloc[:, 1:] = relevant_data.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')
                lego_nodes_sharing_nuts3 = mapping_data_wien[mapping_data_wien['NUTS3'].str.contains(nuts_id, na=False)]
                num_sharing_lego_nodes = len(lego_nodes_sharing_nuts3)
                divided_data = relevant_data.iloc[:, 1:] / num_sharing_lego_nodes
                lego_combined_data = divided_data if lego_combined_data is None else lego_combined_data + divided_data.values
        if lego_combined_data is not None:
            lego_sum = lego_combined_data.sum().sum()
            total_distributed_demand += lego_sum
            processed_data.append(pd.Series(
                [lego_id] + lego_combined_data.sum(axis=0).tolist(),
                index=['LEGO ID'] + list(nuts_data.columns[1:])
            ))
            distributed_demand_tracker[lego_id] = lego_sum
        else:
            logger.info(f"Error: No data was combined for LEGO ID {lego_id}. Skipping this LEGO ID.")
    processed_time_series = pd.concat(processed_data, axis=1).T if processed_data else pd.DataFrame()
    processed_time_series.iloc[:, 1:] = processed_time_series.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')
    validate_lego_node_count(mapping_data_wien, processed_time_series)
    validate_distribution_result(federal_state_name, total_demand_input, processed_time_series, distributed_demand_tracker)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    output_path = os.path.join(output_folder, filename_result_distribution)
    processed_time_series.to_excel(output_path, index=False)
    logger.info(f"Data saved to: {output_path}")
    return total_distributed_demand

def distribute_nuts_level_data_steiermark(nuts_data, mapping_data, total_demand_input, federal_state_name, output_folder, filename_result_distribution):
    """
    Distributes time series data for Steiermark across LEGO nodes and saves the results to an Excel file.
    """
    if federal_state_name != 'Steiermark':
        logger.info(f"The federal state '{federal_state_name}' does not match 'Steiermark'. No action taken.")
        return None
    mapping_data_steiermark = mapping_data[mapping_data['NUTS2'].str.strip() == 'AT22']
    logger.info(f"Mapping data for {federal_state_name}:\n{mapping_data_steiermark}")
    processed_data = []
    distributed_demand_tracker = {}
    total_distributed_demand = 0
    for _, row in mapping_data_steiermark.iterrows():
        lego_id = row['LEGO ID']
        nuts_ids = [nuts_id.strip() for nuts_id in row['NUTS3'].split(',')] if isinstance(row['NUTS3'], str) else [row['NUTS3']]
        lego_combined_data = None
        for nuts_id in nuts_ids:
            relevant_data = nuts_data[nuts_data['NUTS-ID'].str.strip() == nuts_id.strip()]
            if not relevant_data.empty:
                relevant_data.iloc[:, 1:] = relevant_data.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')
                lego_nodes_sharing_nuts3 = mapping_data_steiermark[mapping_data_steiermark['NUTS3'].str.contains(nuts_id, na=False)]
                num_sharing_lego_nodes = len(lego_nodes_sharing_nuts3)
                divided_data = relevant_data.iloc[:, 1:] / num_sharing_lego_nodes
                lego_combined_data = divided_data if lego_combined_data is None else lego_combined_data + divided_data.values
        if lego_combined_data is not None:
            lego_sum = lego_combined_data.sum().sum()
            total_distributed_demand += lego_sum
            processed_data.append(pd.Series(
                [lego_id] + lego_combined_data.sum(axis=0).tolist(),
                index=['LEGO ID'] + list(nuts_data.columns[1:])
            ))
            distributed_demand_tracker[lego_id] = lego_sum
        else:
            logger.info(f"Error: No data was combined for LEGO ID {lego_id}. Skipping this LEGO ID.")
    processed_time_series = pd.concat(processed_data, axis=1).T if processed_data else pd.DataFrame()
    processed_time_series.iloc[:, 1:] = processed_time_series.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')
    validate_lego_node_count(mapping_data_steiermark, processed_time_series)
    validate_distribution_result(federal_state_name, total_demand_input, processed_time_series, distributed_demand_tracker)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    output_path = os.path.join(output_folder, filename_result_distribution)
    processed_time_series.to_excel(output_path, index=False)
    logger.info(f"Data saved to: {output_path}")
    return total_distributed_demand
