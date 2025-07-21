import pyuff
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
from simparameter import SimParameter, UniformDistribution
from simparameter_set import SimParamSet
from surrogate_model import SurrogateModel
import seaborn as sns

def _compute_dot_products(input_eigenvectors : np.ndarray, reference_eigenvectors: np.ndarray) -> np.ndarray:
    ''' Calculates dot products between eigenvectors and reference eigenvectors.
    
        Parameters
        ----------
        input_eigenvectors : ndarray of shape (n_samples, n_nodes)
            Array of eigenvectors.
            
        reference_eigenvectors : ndarray of shape (n_clusters, n_nodes)
            Array of reference_eigenvectors.
            
        Returns
        -------
        dot_product : ndarray of shape (n_samples, 1, n_clusters)
            Matrix of dot products between input eigenvectors and typical eigenvectors.'''
    
    num_of_samples = input_eigenvectors.shape[0]
    num_of_modes = input_eigenvectors.shape[1]
    num_of_components = input_eigenvectors.shape[2]
    input_eigenvectors_reshaped = input_eigenvectors.reshape(-1, num_of_components)
    dot_products = np.dot(input_eigenvectors_reshaped, reference_eigenvectors.T)
    dot_products = dot_products.reshape(num_of_samples, num_of_modes, -1)
    return dot_products

def _flip_eigenvectors(input_eigenvectors: np.ndarray, reference_eigenvectors: np.ndarray | None = None) -> np.ndarray:
    ''' Flips eigenvectors by clusters to the direction of the typical eigenvectors.
    
        Parameters
        ----------
        input_eigenvectors : ndarray of shape (n_samples, n_nodes)
            Array of eigenvectors.
            
        reference_eigenvectors : ndarray of shape (n_clusters, n_nodes), default=None
            Array of reference_eigenvectors.
        
        Returns
        -------
        flipped_eigenvectors : ndarray
            Flipped versions of the input eigenvectors.'''
    
    input_eigenvectors = np.expand_dims(input_eigenvectors, axis=1)
    reference_eigenvectors = input_eigenvectors[0].copy() if reference_eigenvectors is None else reference_eigenvectors
    dot_products = _compute_dot_products(input_eigenvectors, reference_eigenvectors)
    max_abs_indices = np.argmax(np.abs(dot_products), axis=-1)
    max_abs_signs = np.sign(dot_products[np.arange(dot_products.shape[0])[:, None], np.arange(dot_products.shape[1]), max_abs_indices])
    flipped_eigenvectors = input_eigenvectors*max_abs_signs[:,:,np.newaxis]
    flipped_eigenvectors = flipped_eigenvectors.reshape(flipped_eigenvectors.shape[0], flipped_eigenvectors.shape[2])
    return flipped_eigenvectors

sensor_cols = ['phi_sensor_1_real', 'phi_sensor_1_imag', 'phi_sensor_2_real',
       'phi_sensor_2_imag', 'phi_sensor_3_real', 'phi_sensor_3_imag',
       'phi_sensor_4_real', 'phi_sensor_4_imag', 'phi_sensor_5_real',
       'phi_sensor_5_imag', 'phi_sensor_6_real', 'phi_sensor_6_imag']
modeshape_cols = ['frequency'] + sensor_cols

def df_from_unv(unv_filename):
    """
    Reads a UNV file and returns a Pandas DataFrame with the data.
    """
    extracted_data = []
    try:
        read_uff = pyuff.UFF(unv_filename)
        all_datasets = read_uff.read_sets()

        #print(f"Successfully read {len(all_datasets)} datasets from '{unv_filename}'.")

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
                    continue
                    #print(f"Warning: Skipping incomplete Dataset 55 (OMA ID: {oma_id}, Mode ID: {mode_id})")

        # Create a Pandas DataFrame from the extracted data
        if extracted_data:
            df = pd.DataFrame(extracted_data)
            df['Datetime'] = pd.to_datetime(df['time start'])
            df.drop(['OMA id', 'time start'], axis=1, inplace=True)
            return df
        else:
            print("\nNo Dataset 55 found or extracted data was incomplete. No DataFrame created.")

    except FileNotFoundError:
        print(f"Error: The file '{unv_filename}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

def wind_to_vector(direction):
    compass = {
        'N': 0, 'NNE': 22.5, 'NE': 45, 'ENE': 67.5,
        'E': 90, 'ESE': 112.5, 'SE': 135, 'SSE': 157.5,
        'S': 180, 'SSW': 202.5, 'SW': 225, 'WSW': 247.5,
        'W': 270, 'WNW': 292.5, 'NW': 315, 'NNW': 337.5,
        'CALM': None
    }
    angle = compass.get(direction)
    if angle is None:
        return np.array([0.0, 0.0])
    radians = np.deg2rad(angle)
    return np.array([round(np.cos(radians),3), round(np.sin(radians),3)])

def encode_condition(cond):
    cond = cond.lower()
    if "fair" in cond and "windy" not in cond:
        return "Clear"
    elif "cloudy" in cond:
        return "Cloudy"
    elif "rain" in cond:
        return "Rain"
    elif "drizzle" in cond:
        return "Drizzle"
    elif "fog" in cond:
        return "Fog"
    elif "windy" in cond:
        return "Windy"
    else:
        return "Other"

def preprocess_weather_data(weather_data):
    weather_data = weather_data.copy()
    weather_data['Temperature'] = weather_data['Temperature'].str.replace('°F', '').astype(float)
    weather_data['Temperature'] = (weather_data['Temperature'] - 32) * 5 / 9 # convert to Celsius
    weather_data['Dew Point'] = weather_data['Dew Point'].str.replace('°F', '').astype(float)
    weather_data['Humidity'] = weather_data['Humidity'].str.replace('°%', '').astype(float)
    weather_data['Wind Speed'] = weather_data['Wind Speed'].str.replace('°mph', '').str.replace(' mph', '').str.replace('CALM', '0').astype(float)  # handle 'CALM'
    weather_data['Wind Gust'] = weather_data['Wind Gust'].str.replace('°mph', '').str.replace(' mph', '').astype(float)
    weather_data['Pressure'] = weather_data['Pressure'].str.replace('°in', '').str.replace(' in', '').astype(float)
    weather_data['Precip.'] = weather_data['Precip.'].str.replace('°in', '').str.replace(' in', '').str.replace('"', '', regex=False).astype(float)
    weather_data['Wind Gust'] = weather_data['Wind Gust'].fillna(0)

    weather_data['Datetime'] = pd.to_datetime(weather_data['Date'] + ' ' + weather_data['Time'])
    weather_data[['Wind_X', 'Wind_Y']] = weather_data['Wind'].apply(lambda d: pd.Series(wind_to_vector(d)))
    weather_data['Condition_Simplified'] = weather_data['Condition'].apply(encode_condition)
    condition_encoded = pd.get_dummies(weather_data['Condition_Simplified'], prefix='Cond', dtype='int')
    weather_data = pd.concat([weather_data, condition_encoded], axis=1)
    weather_data = weather_data.drop(['Date', 'Time', 'Wind', 'Condition', 'Condition_Simplified'], axis=1)

    return weather_data

def merge_dataframes(df_modeshapes, df_weather):
    """
    Merges building and weather DataFrames on the 'Datetime' column.
    """
    df_a = df_modeshapes.sort_values(by='Datetime')
    df_b = df_weather.sort_values(by='Datetime')
    merged_df = pd.merge_asof(df_a, df_b, on='Datetime', direction='nearest') # merge based on the nearest time
    if 'mode id' in merged_df.columns:
        merged_df = merged_df.rename(columns={'mode id': 'mode_id'})

    return merged_df

def separate_modes(merged_df):
    modeshapes_df = merged_df.copy()#[modeshape_cols+['mode id', 'Datetime', 'Humidity', 'Temperature']].copy()
    modes = set(modeshapes_df['mode id'])
    modeshapes_dict = {}
    for mode in modes:
        modeshapes_dict[mode] = modeshapes_df[modeshapes_df['mode id'] == mode].drop(['mode id'], axis=1).sort_values(by='Datetime')

        new_df = modeshapes_dict[mode]
        cols = new_df.columns
        #df = new_df.drop(columns=['frequency', 'Datetime', 'Humidity', 'Temperature']).copy()
        df = new_df[sensor_cols].copy()
        new_df1 = pd.DataFrame(_flip_eigenvectors(df.values, reference_eigenvectors=None), columns=sensor_cols)
        for col in cols:
            if col not in sensor_cols:
                new_df1[col] = new_df[col].values
        #new_df1['frequency'] = new_df['frequency'].values
        #new_df1['Datetime'] = new_df['Datetime'].values
        #new_df1['Temperature'] = new_df['Temperature'].values
        #new_df1['Humidity'] = new_df['Humidity'].values
        modeshapes_dict[mode] = new_df1.copy()

    return modeshapes_dict

def merge_modes(modeshapes_dict):
    for i in modeshapes_dict.keys():
        df = modeshapes_dict[i]
        df['mode_id'] = pd.Series([i] * df.shape[0])
        modeshapes_dict[i] = df

    modeshapes_df = pd.concat([modeshapes_dict[i] for i in modeshapes_dict.keys()])
    modeshapes_df = modeshapes_df[~modeshapes_df['mode_id'].isin([-1, -2])]
    return modeshapes_df

def select_modes(df, modes, X_cols):
    # Pivot the DataFrame
    if 'Datetime' not in X_cols:
        X_cols = ['Datetime'] + X_cols
    pivoted_df = df.pivot_table(index=X_cols,
                                columns='mode_id',
                                values=modeshape_cols,
                                aggfunc='first') # 'first' works assuming no duplicates for (time, mode_id)

    # Flatten the MultiIndex columns
    pivoted_df.columns = [f'{col[0]}_mode_{col[1]}' for col in pivoted_df.columns]
    final_df = pivoted_df.reset_index()

    ordered_columns = X_cols
    for mode in modes:
        for col_name in modeshape_cols:
            ordered_columns.append(f'{col_name}_mode_{mode}')

    final_df = final_df[ordered_columns]

    return final_df

def select_cols(data, modes, X_cols, y_cols):
    """
    modes: list of mode ids
    X_cols: weather columns to be used
    y_cols can be 'frequency', 'modeshapes', 'frequency+modeshapes', or a list of column names
    """
    modes = sorted(modes)
    df = select_modes(data, modes, X_cols)

    selected_cols = []

    if y_cols == 'frequency':
        for col in df.columns:
            if 'frequency' in col:
                selected_cols.append(col)
    elif y_cols == 'modeshapes':
        for mode in modes:
            selected_cols.extend([f"{prefix}_mode_{mode}" for prefix in sensor_cols])
    elif y_cols == 'frequency+modeshapes':
        for mode in modes:
            selected_cols.extend([f"{prefix}_mode_{mode}" for prefix in modeshape_cols])

    X = df[X_cols]
    y = df[selected_cols]
    Q = SimParamSet()
    for col in X_cols:
        Q.add(SimParameter(col, UniformDistribution(df[col].min(), df[col].max())))
    y_names = selected_cols

    return X, y, Q, y_names

def eval_model(model):
    [df, model_eval] = model.evaluate_model(verbose=True)
    return df, model_eval

def plot_correlation_mx(df, rows=None, cols=None, annot=True):
    corr = df.corr()
    
    # If rows or cols are not specified, use all columns
    if rows is None:
        rows = corr.index.tolist()
    if cols is None:
        cols = corr.columns.tolist()
    
    corr_subset = corr.loc[rows, cols]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_subset, cmap='coolwarm', ax=ax, annot=annot)
    ax.set_title('Correlation Matrix')
    return fig

if __name__ == "__main__":
    unv_filename = '/mnt/idms/home/vemese/sglib_ttb/demo/data/OMA hospital/hospital_data_06_09_new_labels.unv'
    weather_filename = '/mnt/idms/home/vemese/sglib_ttb/demo/data/OMA hospital/LEGR_2025-02-04_2025-06-09.csv'
    weather_data = pd.read_csv(weather_filename, index_col=0)
    df_modeshapes = df_from_unv(unv_filename)
    weather_df = preprocess_weather_data(weather_data)

    merged_df = merge_dataframes(df_modeshapes, weather_df)
    modeshapes_dict = separate_modes(merged_df)
    modeshapes_df = merge_modes(modeshapes_dict)

    # Example training
    modes=[3,6]
    X_cols=['Temperature', 'Humidity', 'Wind_X', 'Cond_Fog']
    y_cols='frequency+modeshapes'
    X, y, Q, y_names = select_cols(modeshapes_df, modes, X_cols, y_cols)
    model_type = 'LinReg'
    model = SurrogateModel(Q, y_names, model_type)
    model.train_test_split(X, y)

    # fill missing modes
    for col in y_names:
        model.y_train[col] = model.y_train[col].fillna(model.y_train[col].mean())
        model.y_test[col] = model.y_test[col].fillna(model.y_train[col].mean())

    model.train(model.X_train, model.y_train)
    df, model_eval = model.evaluate_model(verbose=False)
    print(df)
    print(model_eval)
