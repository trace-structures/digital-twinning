import os
import pyuff
import numpy as np
import requests
import pandas as pd
import re
import json
from datetime import datetime

def get_json_1(start_date, end_date, step_hours=1):
    """
    start_date and end_date are required in string format: '2025-03-01'
    earliest start_date: '2025-02-04'
    """
    
    # Define the API endpoint and parameters
    url = "https://upward-prime-turkey.ngrok-free.app/APIRest/plot/fetch_oma_api.php"
    measurements_params = {
        "start_date": start_date,
        "end_date": end_date,
        "step_hours": step_hours
    }

    # Send the GET request to the API
    response = requests.get(url, params=measurements_params)

    # Check if the request was successful
    if response.status_code == 200:
        measurements_data = response.json()  # Parse the JSON response
        print(f"Received {len(measurements_data)} records.")
    else:
        print("Request failed:", response.status_code, response.text)

    return measurements_data

def json_to_unv_1(measurements_data, output_filename):
    """
    measurements_data: output of get_json()
    """

    # --- Check if the output file exists and delete it if it does ---
    if os.path.exists(output_filename):
        print(f"Output file '{output_filename}' already exists. Deleting it.")
        os.remove(output_filename)

    uff_file = pyuff.UFF(output_filename, 'w')

    nodes = measurements_data[0]['sensors']
    node_num = len(nodes)

    # --- Dataset type 164: Unit system ---
    dataset_164 = {'type': 164,
    'units_code': 1,
    'units_description': 'mm/newton',
    'temp_mode': 1,
    'length': 1.0,
    'force': 1.0,
    'temp': 1.0,
    'temp_offset': 1.0
    }

    # --- Dataset type 15: Node coordinates ---
    node_nums = range(1, node_num + 1)
    def_cs=[0]*len(node_nums)
    disp_cs=[0]*len(node_nums)

    # Dummy coordinates for each sensor/node
    coordinates = {
        'x': list(range(1,node_num+1)),
        'y': [0.0] * node_num,
        'z': [0.0] * node_num
    }

    dataset_15 = {'type': 15,
    'node_nums': node_nums,
    'def_cs': def_cs,
    'disp_cs': disp_cs,
    'x': coordinates['x'],
    'y': coordinates['y'],
    'z': coordinates['z']
    }

    # --- Prepare type 55 datasets from OMA data ---
    all_dataset_55_to_write = []

    # Iterate through each OMA result dictionary in the list
    for data in measurements_data:
        oma_id = data.get('OMA_id', 'N/A')
        timestamp_start = data.get('timestamp_start', 'N/A')
        timestamp_end = data.get('timestamp_end', 'N/A')
        sensors = data.get('sensors', []) # Get sensors, default to empty list
        modes = data.get('modes', [])     # Get modes, default to empty list

        if not modes:
            continue

        # Convert sensor IDs to node numbers (assuming they are integers)
        if sensors:
            node_nums = np.array([int(s) for s in sensors])
            num_nodes = len(node_nums)
        else:
            print(f"Warning: No sensors found for OMA_id {oma_id}. Cannot process modes for this result. Skipping.")
            continue

        # Iterate through each mode within the current OMA result
        for mode_data in modes:
            mode_id = mode_data.get('mode_id', 'N/A')
            frequency = mode_data.get('frequency')
            damping = mode_data.get('xi')
            phi_data = mode_data.get('phi', [])

            if frequency is None or damping is None or not phi_data:
                print(f"Warning: Incomplete data for Mode {mode_id} in OMA_id {oma_id}. Skipping mode.")
                continue
            
            # Convert phi data to a numpy array of complex numbers
            if len(phi_data) != num_nodes:
                print(f"Warning: Number of phi values ({len(phi_data)}) does not match number of sensors ({num_nodes}) for Mode {mode_id} in OMA_id {oma_id}. Skipping mode.")
                continue

            r1_data = np.array([complex(p.get('real', 0), p.get('imag', 0)) for p in phi_data], dtype=complex)

            # Create arrays for r2 and r3 (assuming only one component is available in phi)
            r2_data = np.zeros(num_nodes, dtype=complex)
            r3_data = np.zeros(num_nodes, dtype=complex)

            # Construct the dataset 55 dictionary
            dataset_55 = {
                'type': 55,
                'id1': f'OMA id: {oma_id}',
                'id2': f"Mode id: {mode_id}",
                'id3': f"Time from {timestamp_start} to {timestamp_end}",
                'id4': f"Xi: {damping}",
                'id5': None,
                'model_type': 1,         # Structural
                'analysis_type': 2,      # 
                'data_ch': 2,            # Complex data
                'spec_data_type': 8,     # Eigenvalue
                'data_type': 5,          # Displacement
                'n_data_per_node': 3,    # We are providing r1, r2, r3
                'r1': r1_data,
                'r2': r2_data,
                'r3': r3_data,
                'load_case': int(oma_id) if isinstance(oma_id, (int, str)) and str(oma_id).isdigit() else -3, # Use OMA_id as load case if it's a number
                'mode_n': int(mode_id) if isinstance(mode_id, (int, str)) and str(mode_id).isdigit() else -3, # Use mode_id as mode number
                'freq': frequency,
                #'modal_a': 0j,
                #'modal_b': 0j,
                #'eig': eigenvalue,
                'node_nums': node_nums
            }

            all_dataset_55_to_write.append(dataset_55)

    # --- Write all datasets to file in proper order ---
    all_datasets = [dataset_164, dataset_15] + all_dataset_55_to_write
    uff_file.write_sets(all_datasets)
    print(f"Successfully wrote {len(all_dataset_55_to_write)} mode shape datasets to {output_filename}")

    # Reading back to verify
    try:
        read_uff = pyuff.UFF(output_filename)
        read_datasets = read_uff.read_sets()
        print(f"\nSuccessfully read back {len(read_datasets)} datasets from the file.")
    except Exception as e:
        print(f"\nError reading the file back: {e}")

def read_unv_file(file_path):
    uff_file = pyuff.UFF(file_path, 'r')
    read_datasets = uff_file.read_sets()
    return read_datasets

def unv_to_json_1(read_datasets, save_to_file=None):
    
    type55_modes = read_datasets[2:]

    result = {
        'timestamp_start': None,
        'timestamp_end': None,
        'OMA_id': None,
        'flag': 0,
        'sensors': [],
        'temperature': 'N/A',
        'humidity': 'N/A',
        'modes': []
    }

    for i, mode in enumerate(type55_modes):
        # === Parse metadata once ===
        if i == 0:
            # Extract OMA ID
            if 'OMA id:' in mode.get('id1', ''):
                result['OMA_id'] = int(mode['id1'].split(':')[1].strip())

            # Extract timestamps
            if 'Time from' in mode.get('id3', ''):
                time_range = mode['id3'].replace('Time from ', '').split(' to ')
                result['timestamp_start'] = time_range[0]
                result['timestamp_end'] = time_range[1]

            # Sensors = node_nums as strings
            result['sensors'] = [str(n) for n in mode['node_nums']]

        # === Extract mode shape ===
        phi_list = []
        for val in mode['r1']:
            phi_list.append({
                'real': float(val.real),
                'imag': float(val.imag)
            })

        if 'Xi:' in mode.get('id4', ''):
            try:
                xi = float(mode['id4'].split(':')[1].strip())
            except:
                xi = 0

        mode_entry = {
            'mode_id': int(mode['mode_n']),
            'frequency': float(mode['freq']),
            'phi': phi_list,
            'xi': xi
        }

        result['modes'].append(mode_entry)

    if save_to_file:
        with open(save_to_file, 'w') as f:
            json.dump(result, f)

    return result


def unv_to_csv_1(unv_filename, save_to_file=False):
    """
    Reads a UNV file and returns a Pandas DataFrame with the data.
    """
    extracted_data = []
    try:
        read_uff = pyuff.UFF(unv_filename)
        all_datasets = read_uff.read_sets()

        print(f"Successfully read {len(all_datasets)} datasets from '{unv_filename}'.")

        for dataset in all_datasets:
            if dataset.get('type') == 55:

                oma_id_match = re.search(r'OMA id\s*:\s*(\d+)', dataset.get('id1', ''))
                oma_id = int(oma_id_match.group(1)) if oma_id_match else None

                mode_id_match = re.search(r'Mode id\s*:\s*(-?\d+)', dataset.get('id2', ''))
                mode_id = int(mode_id_match.group(1)) if mode_id_match else None

                timestamp_match = re.search(r'Time from\s*(.*?)\s*to', dataset.get('id3', ''))
                time_start = timestamp_match.group(1).strip() if timestamp_match else None

                #freq_match = re.search(r'Frequency\s*:\s*([\d.eE+-]+)', dataset.get('id4', ''))
                #frequency = float(freq_match.group(1)) if freq_match else None
                frequency = dataset.get('freq')

                xi_match = re.search(r'Xi\s*:\s*([\d.eE+-]+)', dataset.get('id4', ''))
                xi = float(xi_match.group(1)) if xi_match else None

                node_nums = dataset.get('node_nums')
                r1_data = dataset.get('r1')

                if oma_id is not None and mode_id is not None and frequency is not None and xi is not None and node_nums is not None and r1_data is not None and len(node_nums) == len(r1_data):
                    row_data = {
                        'OMA id': oma_id,
                        'mode id': mode_id,
                        'time start': time_start,
                        'frequency': frequency,
                        'xi': xi,
                    }
                    for i, node_num in enumerate(node_nums):
                        phi_complex = r1_data[i]

                        # Add real and imaginary parts as separate columns
                        row_data[f'phi_sensor_{node_num}_real'] = phi_complex.real
                        row_data[f'phi_sensor_{node_num}_imag'] = phi_complex.imag

                    extracted_data.append(row_data)
                else:
                    print(f"Warning: Skipping incomplete Dataset 55 (OMA ID: {oma_id}, Mode ID: {mode_id})")

        # Create a Pandas DataFrame from the extracted data
        if extracted_data:
            df = pd.DataFrame(extracted_data)
            if save_to_file:
                df.to_csv(save_to_file, index=False)
            return df
        else:
            print("\nNo Dataset 55 found or extracted data was incomplete. No DataFrame created.")

    except FileNotFoundError:
        print(f"Error: The file '{unv_filename}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")