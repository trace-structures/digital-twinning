import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns
import matplotlib.dates as mdates
# from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
# import plotly.express as px
from plotly.colors import sample_colorscale
from matplotlib.cm import ScalarMappable
from plotly.subplots import make_subplots

def scatter_plot_modes_subplots(modeshapes_dict, x_key='Humidity', y_key='frequency', modes=[0, 1, 2, 3, 6]):
    num_plots = len(modes)
    rows = (num_plots + 1) // 2  # arrange in 2 columns
    cols = 2 if num_plots > 1 else 1

    fig = make_subplots(rows=rows, cols=cols, subplot_titles=[f"Mode {m}" for m in modes])

    for idx, mode in enumerate(modes):
        row = idx // 2 + 1
        col = idx % 2 + 1

        df = modeshapes_dict[mode]
        x = df[x_key]
        y = df[y_key]

        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode='markers',
                marker=dict(color='royalblue', size=6),
                name=f'Mode {mode}',
                showlegend=False
            ),
            row=row,
            col=col
        )

        fig.update_xaxes(title_text=x_key, row=row, col=col)
        fig.update_yaxes(title_text=y_key, row=row, col=col)

    fig.update_layout(
        height=300 * rows,
        width=900,
        title_text=f'{x_key} vs {y_key} for Selected Modes',
        showlegend=False
    )

    return fig

def plot_parts(modeshapes_dict, mode):
    df = modeshapes_dict[mode]
    parts = [('real', 'orange', 'Real'), ('imag', 'royalblue', 'Imaginary')]
    figures = []

    for part, color, label_prefix in parts:
        t = df['Datetime']
        sensors = range(1, 7)
        fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(20, 12), sharex=True)
        axes = axes.flatten()

        for i, sensor in enumerate(sensors):
            values = df[f'phi_sensor_{sensor}_{part}']
            ax = axes[i]
            ax.scatter(t, values, c=color, label=f'{label_prefix} Sensor {sensor}', s=20)
            ax.plot(t, values, color=color, linestyle='--')

            ax.set_title(f'Sensor {sensor}')
            ax.set_ylabel('Sensor Value', fontsize=12)
            ax.set_xlabel('Time', fontsize=12)
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d-%H:%M'))
            ax.legend()

        fig.suptitle(f'Mode {mode} {label_prefix} Part for All Sensors', fontsize=16)
        fig.autofmt_xdate()
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        figures.append(fig)

    return figures[0], figures[1]

def plot_sensor_complex_colored(modeshapes_dict, mode, sensor, color_by='Temperature', lines=False):
    df = modeshapes_dict[mode]
    t = df['Datetime']
    color_data = df[color_by]
    real = df[f'phi_sensor_{sensor}_real']
    imag = df[f'phi_sensor_{sensor}_imag']

    fig, ax = plt.subplots(figsize=(20, 8))

    sc1 = ax.scatter(t, real, c=color_data, cmap='viridis', label='Real part', marker='o', s=25)
    sc2 = ax.scatter(t, imag, c=color_data, cmap='viridis', label='Imaginary part', marker='x', s=25)

    if lines:
        ax.plot(t, real, color='gray', alpha=0.3, linestyle='--')
        ax.plot(t, imag, color='gray', alpha=0.3, linestyle='--')

    # Colorbar
    cbar = plt.colorbar(sc1, ax=ax)
    cbar.set_label(f'{color_by} (°C)' if color_by.lower() == 'temperature' else color_by)

    # Format x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d-%H:%M'))
    fig.autofmt_xdate()

    ax.set_title(f'Mode {mode} Sensor {sensor} Data Colored by {color_by}', fontsize=14)
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Sensor Value', fontsize=12)
    ax.legend()

    return fig

def plot_complex_plane_all_sensors(modeshapes_dict, mode, color_by='Temperature'):
    df = modeshapes_dict[mode]
    num_sensors = 6

    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    axes = axes.flatten()

    color_data = df[color_by]
    cmap = 'viridis'

    # Dummy ScalarMappable for global colorbar (created before loop)
    norm = plt.Normalize(vmin=color_data.min(), vmax=color_data.max())
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])  # Required to avoid warning in some versions

    for i, sensor in enumerate(range(1, num_sensors + 1)):
        ax = axes[i]
        real = df[f'phi_sensor_{sensor}_real']
        imag = df[f'phi_sensor_{sensor}_imag']

        sc = ax.scatter(real, imag, c=color_data, cmap=cmap, s=20, norm=norm)
        ax.set_title(f'Sensor {sensor}')
        ax.set_xlabel('Real')
        ax.set_ylabel('Imaginary')
        ax.grid(True)
        ax.axis('equal')

    # Shared colorbar placed to the right of all subplots
    cbar = fig.colorbar(sm, ax=axes, location='right', shrink=0.8, pad=0.05)
    cbar.set_label(f'{color_by} (°C)' if color_by.lower() == 'temperature' else color_by)

    fig.suptitle(f'Mode {mode} Sensor Values in Complex Plane Colored by {color_by}', fontsize=14)

    return fig

def plot_mode_shapes_3d(modeshapes_dict, mode, color_by='Temperature'):
    df = modeshapes_dict[mode]
    fig = go.Figure()
    num_sensors = 6
    num_timesteps = len(df)

    color_vals = df[color_by]
    cmin, cmax = color_vals.min(), color_vals.max()

    # Accumulate all marker points
    x_all, y_all, z_all, c_all = [], [], [], []

    for t in range(num_timesteps):
        value = color_vals.iloc[t]
        norm_val = (value - cmin) / (cmax - cmin)
        line_color = sample_colorscale('Viridis', [norm_val])[0]

        x = list(range(1, num_sensors + 1))
        y = [df[f'phi_sensor_{s}_imag'].iloc[t] for s in x]
        z = [df[f'phi_sensor_{s}_real'].iloc[t] for s in x]

        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=z,
            mode='lines',
            line=dict(color=line_color, width=1.5),
            showlegend=False
        ))

        for s in x:
            x_all.append(s)
            y_all.append(df[f'phi_sensor_{s}_imag'].iloc[t])
            z_all.append(df[f'phi_sensor_{s}_real'].iloc[t])
            c_all.append(value)

    # Add temperature-colored markers
    fig.add_trace(go.Scatter3d(
        x=x_all, y=y_all, z=z_all,
        mode='markers',
        marker=dict(
            size=4,
            color=c_all,
            colorscale='Viridis',
            cmin=cmin,
            cmax=cmax,
            colorbar=dict(title=f'{color_by}'),
            showscale=True
        ),
        showlegend=False
    ))

    fig.update_layout(
        width=1000,
        height=800,
        scene=dict(
            xaxis_title='Sensor',
            yaxis_title='Imaginary Part',
            zaxis_title='Real Part',
        ),
        title=f'3D Mode Shape (Mode {mode}) Colored by {color_by}'
    )
    return fig