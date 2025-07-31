import os
import pyuff
import numpy as np
import pandas as pd
import json
from datetime import datetime

def load_json(filename):
    with open(filename) as f:
        d = json.load(f)
    return d

def json_to_unv_2(d, save_to_file):
    """
    d: loaded json file
    """
    num_nodes = 13

    freqs = []
    for dict in d['data']:
        freq = dict['frequency']
        freqs.append(freq)

    disps_dict = {}
    for i in range(len(d['data'])):
        disps = []
        nodes = list(range(1,num_nodes + 1))
        for n in nodes:
            for node_dict in d['data'][i]['nodes']:
                node_id = node_dict['node']
                if f"#node-{n}" in node_id:
                    disps.append({n: node_dict['displacement']})
                    break
        disps_dict[i] = disps
    
    mode_shapes = {}
    for i in disps_dict.keys():
        x, y, z = [], [], []
        for node in disps_dict[i]:
            for _, coord in node.items():
                x.append(float(coord['x']))
                y.append(float(coord['y']))
                z.append(float(coord['z']))
        disp_i = {'x': x, 'y': y, 'z': z}
        mode_shapes[i] = disp_i

    n = 1
    coordinates = []
    while n < num_nodes + 1:
        for node_dict in d['data'][0]['nodes']:
            node_id = node_dict['node']
            if f"#node-{n}" in node_id:
                coordinates.append({node_id: node_dict['coordinates']})
                n += 1

    x, y, z = [], [], []
    for node in coordinates:
        for _, coord in node.items():
            x.append(float(coord['x']))
            y.append(float(coord['y']))
            z.append(float(coord['z']))
    coordinates = {'x': x, 'y': y, 'z': z}

    units_code=1
    units_description='mm/newton'
    temp_mode=1
    length=1.0
    force=1.0
    temp=1.0
    temp_offset=1.0
    def_cs=0
    disp_cs=0
    id1='NONE'
    id2='NONE'
    id3='NONE'
    id4='NONE'
    id5='NONE'
    model_type=1
    analysis_type=2
    data_ch=2
    spec_data_type=8
    data_type=2
    n_data_per_node=3
    load_case=1
    modal_m=0
    modal_damp_vis=0.
    modal_damp_his=0.

    if save_to_file and os.path.exists(save_to_file):
        os.remove(save_to_file)

    uff_datasets = []
    node_nums = list(range(1, num_nodes + 1))

    ## Prepare type 164 data
    data_164 = pyuff.prepare_164(
        units_code=units_code,
        units_description=units_description,
        temp_mode=temp_mode,
        length=length,
        force=force,
        temp=temp,
        temp_offset=temp_offset,
    )
    uff_datasets.append(data_164.copy())
    if save_to_file:
        uffwrite = pyuff.UFF(save_to_file)
        uffwrite._write_set(data_164, 'add')

    ## Prepare type 15 data
    if def_cs==0:
        def_cs=[0]*len(node_nums)
    if disp_cs==0:
        disp_cs=[0]*len(node_nums)
    data_15 = pyuff.prepare_15(
        node_nums=node_nums,
        def_cs=def_cs,
        disp_cs=disp_cs,
        x=coordinates['x'],
        y=coordinates['y'],
        z=coordinates['z'],
    )
    uff_datasets.append(data_15.copy())
    if save_to_file:
        uffwrite = pyuff.UFF(save_to_file)
        uffwrite._write_set(data_15, 'add')

    ## Prepare type 55 data
    for mode_i in range(len(d['data'])):
        data_55 = pyuff.prepare_55(
            id1=id1,
            id2=id2,
            id3=id3,
            id4=id4,
            id5=id5,
            model_type=model_type,
            analysis_type=analysis_type,
            data_ch=data_ch,
            spec_data_type=spec_data_type,
            data_type=data_type,
            n_data_per_node=n_data_per_node,
            load_case=load_case,
            mode_n=mode_i,
            freq=freqs[mode_i],
            modal_m=modal_m,
            modal_damp_vis=modal_damp_vis,
            modal_damp_his=modal_damp_his,
            node_nums=np.array(node_nums),
            r1=mode_shapes[mode_i]['x'],
            r2=mode_shapes[mode_i]['y'],
            r3=mode_shapes[mode_i]['z']
        )

        uff_datasets.append(data_55.copy())
        
        if save_to_file:
            uffwrite = pyuff.UFF(save_to_file)
            uffwrite._write_set(data_55, 'add')

def build_node_lookup(type15):
    """Builds a node_num → coordinates dict from type 15 data."""
    return {
        int(node): {
            'x': type15['x'][i],
            'y': type15['y'][i],
            'z': type15['z'][i]
        }
        for i, node in enumerate(type15['node_nums'])
    }

def unv_to_json_2(read_datasets, save_to_file=None):

    """
    Converts UNV type 15 and 55 data to original JSON-like format.
    
    Returns:
    - list of dictionaries matching the original structure
    """
    
    type15 = read_datasets[1]
    type55_modes = read_datasets[2:]

    node_lookup = build_node_lookup(type15)
    results = []

    for mode in type55_modes:
        mode_entry = {
            'frequency': mode['freq'],
            'nodes': []
        }

        for i, node_num in enumerate(mode['node_nums']):
            node_id = f"#node-{int(node_num)}"
            coord = node_lookup.get(int(node_num), {'x': 0, 'y': 0, 'z': 0})

            displacement = {
                'x': float(mode['r1'][i]),
                'y': float(mode['r2'][i]),
                'z': float(mode['r3'][i])
            }

            damping = {
                'hysteretic': float(mode.get('modal_damp_his', 0.0)),
                'viscous': float(mode.get('modal_damp_vis', 0.0))
            }

            node_data = {
                'node': node_id,
                'coordinates': coord,
                'displacement': displacement,
                'damping': damping
            }

            mode_entry['nodes'].append(node_data)

        results.append(mode_entry)

    if save_to_file:
        with open(save_to_file, 'w') as f:
            json.dump(results, f)

    return results

def json_to_csv_2(json_output, save_to_file=None):
    rows = []
    for mode in json_output:
        freq = mode["frequency"]
        for node in mode["nodes"]:
            rows.append({
                "node_id": node["node"],
                "frequency": freq,
                "coord_x": node["coordinates"]["x"],
                "coord_y": node["coordinates"]["y"],
                "coord_z": node["coordinates"]["z"],
                "disp_x": node["displacement"]["x"],
                "disp_y": node["displacement"]["y"],
                "disp_z": node["displacement"]["z"],
                "damp_h": node["damping"]["hysteretic"],
                "damp_v": node["damping"]["viscous"]
            })

    df = pd.DataFrame(rows)
    df = df.drop_duplicates()

    if save_to_file:
        df.to_csv(save_to_file, index=False)

    return df