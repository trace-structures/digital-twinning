import numpy as np
import pandas as pd
import seaborn as sns
import shap
from simparameter_set import SimParamSet
from simparameter import SimParameter
from distributions import UniformDistribution, NormalDistribution
# from scipy.io import loadmat
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import gaussian_kde
from scipy import stats

##########################################################################################
# Functions for toy example
##########################################################################################

def plot_3Dscatter(x, y, z, x_grids=None, y_grids=None, z_plane=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter3D(x.flatten(), y.flatten(), z.flatten(), c=z.flatten(), cmap='jet', s=0.5, zorder=2)
    
    if (x_grids is not None) and (y_grids is not None) and (z_plane is not None):
        ax.plot_surface(x_grids, y_grids, z_plane, color='k', alpha=1, zorder=1)
    
    return fig
    

def plot_3Dscatterplots(K, M, Z_prediction, Z_true, n_train=0):
    fig, ax = plt.subplots(1,2,figsize=(10,10),subplot_kw=dict(projection='3d'))

    sc1 = ax[0].scatter3D(K.flatten(), M.flatten(), Z_prediction.flatten(), c=Z_prediction.flatten(), cmap='jet', s=10)
    #ax[0].set_title('Proxy Surface with ' + str(n_train) + ' train sample')
    ax[0].set_title('Proxy Surface')
    ax[0].set_xlim(0.5, 2.5); ax[0].set_ylim(0.5, 2.5); ax[0].set_zlim(-1, 1);

    sc2 = ax[1].scatter3D(K.flatten(), M.flatten(), Z_true.flatten(), c=Z_true.flatten(), cmap='jet', s=10)
    ax[1].set_title('True Surface')
    ax[1].set_xlim(0.5, 2.5); ax[1].set_ylim(0.5, 2.5); ax[1].set_zlim(-1, 1);
    
    #fig.suptitle('Main title')

## TODO x and y limits: Q bounds

def plot_sobol_sensitivity(sobol_index, y, param_name=None):
    """
    Plots Sobol sensitivity indices with an optional parameter name for a specific plot.
    
    Parameters:
        partial_variance (pd.DataFrame): Partial variance data.
        sobol_index (pd.DataFrame): Sobol index data.
        y (pd.DataFrame): Quantity of interest (QoI).
        param_name (str, optional): Specific parameter to plot (optional).
        ax (matplotlib.axes.Axes, optional): Axis to plot on (optional).
    """
    
 
    y_var = np.broadcast_to(y.var(axis=0).to_numpy().reshape(-1, 1), sobol_index.shape)
    #total_var = np.broadcast_to(total_var, sobol_index.shape)
    y_var_df = pd.DataFrame(y_var, columns=sobol_index.columns, index=sobol_index.index)
    partial_variance = np.multiply(sobol_index, y_var)

    color_map = plt.cm.viridis(np.linspace(0, 1, partial_variance.shape[1]+1))
    colors = {partial_variance.columns[i]: color_map[i] for i in range(len(partial_variance.columns))}
    modes = partial_variance.index
    y_label = 'Labels'
     
    if param_name == 'freq':
        # Find modes starting with 'f'
        f_modes = [mode for mode in partial_variance.index if mode.startswith('f')]
        
        # Create subplots for each mode starting with 'f'
        num_plots = len(f_modes)
        fig, axes = plt.subplots(1, num_plots, figsize=(5 * num_plots, 5))
        
        # If there's only one plot, `axes` is not a list, so make it iterable
        if num_plots == 1:
            axes = [axes]
        
        for i, mode in enumerate(f_modes):
            threshold = 0.01
            loc_par_var = partial_variance.loc[mode]
            df = pd.DataFrame(loc_par_var)
            df[df < 0] = 0
            df['percentage'] = df[mode] / df[mode].sum()
            under_threshold = df.loc[df['percentage'] < threshold].sum()
            remaining = [y_var_df.loc[mode][0], 1] - df.sum()
            others = under_threshold + remaining
            colors['others'] = color_map[-1]
            df = df[df['percentage'] >= threshold]
            df.loc['others'] = others
            if df.loc['others']['percentage'] < 1e-06:
                df.drop(index='others', inplace=True)

            pie_colors = [colors[x] for x in df.index]

            axes[i].pie(df['percentage'], labels=df.index, colors=pie_colors, wedgeprops={"alpha": 0.5})
            axes[i].set_title(f"Sobol sensitivity for: {mode}", fontsize=12)
        
        # Adjust layout so the pie charts don't overlap
        #plt.tight_layout()

        return fig
    
    fig, ax = plt.subplots(figsize=(8, 6))
    if param_name in partial_variance.index:
        # Handle specific parameter case
        threshold = 0.01
        loc_par_var = partial_variance.loc[param_name]
        df = pd.DataFrame(loc_par_var)
        df[df < 0] = 0
        df['percentage'] = df[param_name] / df[param_name].sum()
        under_threshold = df.loc[df['percentage'] < threshold].sum()
        remaining = [y_var_df.loc[param_name][0], 1] - df.sum()
        others = under_threshold + remaining
        colors['others'] = color_map[-1]
        df = df[df['percentage'] >= threshold]
        df.loc['others'] = others
        pie_colors = [colors[x] for x in df.index]

        ax.set_title(f"Sobol sensitivity for: {param_name}", fontsize=16)
        ax.pie(df['percentage'], labels=df.index, colors=pie_colors, wedgeprops={"alpha": 0.5})
        return fig


    elif param_name is not None and param_name not in partial_variance.columns:
        prefix = param_name
        index_names = partial_variance.index
        modes = [item for item in index_names if item.startswith(prefix)]
        if len(modes) == 0:
            raise ValueError(f"There is no mode starting with {prefix}")
        partial_variance = partial_variance.loc[modes]
        y_var_df = y_var_df.loc[modes]
        y_label = param_name
    #else:
    # General case: Stacked area plot
    timesteps = partial_variance.shape[0]

    # Total variance plot
    ax.plot(modes, y_var_df.iloc[:, 0], color='black', label='Total variance', linewidth=0.8)
    #ax.set_xticklabels(modes,rotation=45, ha='right')

    # Stacked area plot
    stackplot_colors = [colors[col] for col in y_var_df.columns]
    ax.stackplot(modes, partial_variance.to_numpy().transpose(), alpha=0.5, labels=partial_variance.columns, colors=stackplot_colors)
    ax.tick_params(axis='x', labelrotation=90)

    # Format the plot
    ax.set_xlim((0, timesteps - 1))
    ax.set_xlabel(y_label, fontsize=14)
    ax.set_ylabel('Variance', fontsize=14)
    ax.set_title('Sobol sensitivity', fontsize=16)
    ax.legend(fontsize=12, loc='upper left', bbox_to_anchor=(1, 1))
    ax.grid(True, alpha=0.3)

    return fig



def plot_shap_single_waterfall(model, q, param_name):
    if type(q) != pd.DataFrame:
        q = pd.DataFrame([q], columns=model.Q.param_names())
            
    explainer = model.model.explainer
    shap_values = explainer(q)
    
    # Data for plotting
    labels = model.Q.param_names()
    param_index = model.QoI_names.index(param_name)
    values = shap_values[0,:,param_index].data  # Corresponding y-values
    bar_lengths = shap_values[0,:,param_index].values  # Bar lengths from the array
    expected_value = shap_values[0,:,param_index].base_values  # Base value

    # Sort bars by absolute value in descending order
    sorted_indices = np.argsort(np.abs(bar_lengths))#[::-1]  # Indices for descending order
    sorted_labels = [labels[i] for i in sorted_indices]
    sorted_values = [values[i] for i in sorted_indices]
    sorted_bar_lengths = [bar_lengths[i] for i in sorted_indices]
    colors = ['#E53935' if val >= 0 else '#1E88E5' for val in sorted_bar_lengths]  # Red for positive, blue for negative

    # Create figure and axis
    #fig, ax = plt.subplots(figsize=(8, 3))
    fig, ax = plt.subplots()
    bar_height=0.915

    # Determine x-axis limits dynamically
    x_min = min(min(expected_value + np.cumsum(sorted_bar_lengths)), expected_value)
    x_max = max(max(expected_value + np.cumsum(sorted_bar_lengths)), expected_value)
    x_range = x_max - x_min
    x_range_ratio = (x_max - x_min)*0.12
    ax.set_xlim(x_min-x_range_ratio, x_max+x_range_ratio)
    #ax.set_xticks(np.linspace(x_min-x_range_ratio, x_max+x_range_ratio, 5))
    
    # Determine y-axis limits dynamically
    y_pos = list(range(len(sorted_labels)))
    y_range_correction = (2/3)*bar_height
    y_min = min(y_pos) - (3/4)*bar_height
    y_max = max(y_pos) + (2/3)*bar_height
    ax.set_ylim(y_min, y_max)
    
    # Plot horizontal bars in cumulative order
    prediction = expected_value  # Start from the base value
    for i in range(len(sorted_labels)):
        bar_length = sorted_bar_lengths[i]
        bar = ax.barh(y_pos[i], bar_length, color=colors[i], height=bar_height, left=prediction, align='center')#, edgecolor='black')
        
        # Position text inside the bar
        text_x = prediction + bar_length
        text_x_correction = (x_max - x_min)*0.02
        sign=''
        if sorted_bar_lengths[i] >= 0:
            x = text_x
            text_x += text_x_correction
            aligment = 'left'
            color = '#E53935'
            sign='+'
        else:
            x = text_x
            text_x -= text_x_correction
            aligment = 'right'
            color = '#1E88E5'
        ax.text(text_x, y_pos[i], f'{sign}{bar_length:.2f}', 
                va='center', ha=aligment, color=color, fontsize=10)
        
        # Update the left position for the next bar
        prediction += bar_length
        
    # Add triangles at the end of bars
        y = y_pos[i]
        half_bar_height = bar_height/2*0.9
        
        if bar_length >= 0:
            triangle = np.array([[x, y - half_bar_height], [x, y + half_bar_height], [x + x_range*0.01, y]])
        else:
            triangle = np.array([[x, y - half_bar_height], [x, y + half_bar_height], [x - x_range*0.01, y]])
        
        ax.add_patch(plt.Polygon(triangle, color=color, label="_nolegend_"))

    # Add reference line for the base value
    ax.axvline(expected_value, color='gray', linestyle='dashed', label=f'Expected Value = {expected_value:.3f}')
    ax.axvline(prediction, color='gray', linestyle='dashed', label=f'Prediction = {prediction:.3f}')
    
    x_label_pos = ax.xaxis.get_ticklabels()[0].get_position()[1]
    ax.text(expected_value, (y_min - half_bar_height)/2, f'Expected Value = {expected_value:.3f}', va='center', fontsize=10, color='black', ha='center')
    ax.text(prediction, y_max + 0.04, f'Prediction = {prediction:.3f}', fontsize=10, color='black', ha='center')
    
    # Labels and formatting
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f'{l} = {v:.3f}' for v, l in zip(sorted_values, sorted_labels)])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_color('black')
    ax.set_title("SHAP Explanation for Individual Prediction", fontsize=16, pad=20)
    
    return fig

def plot_shap_multiple_waterfalls(model, q, param_name=None, show_param_values=False):
    if type(q) != pd.DataFrame:
        q = pd.DataFrame([q], columns=model.Q.param_names())

    explainer = model.model.explainer
    shap_values = explainer(q)
    
    labels = model.Q.param_names()
    QoI_names = model.QoI_names
    num_params = len(QoI_names)  # Number of QoI indices
    num_features = len(labels)  # Number of features (a, b, c)
    
    # Determine x-axis limits dynamically
    all_expected_value = shap_values[0, :, :].base_values
    all_bar_lengths = shap_values[0, :, :].values
    values = shap_values[0, :, 0].data  # Corresponding y-values
    all_labels = np.array([[labels[i]]*num_params for i in range(num_features)])
    
    title = "Prediction of QoI and Effects of Varying Parameters"
    
    if param_name is not None:
        column_names = model.QoI_names
        if param_name == 'freq':     
            freqs = [freq for freq in column_names if freq.startswith('f')]
            QoI_names = freqs
            num_params = len(freqs)
            freqs_indices = [i for i, e in enumerate(column_names) if e in freqs]
            all_expected_value = all_expected_value[freqs_indices]
            all_bar_lengths = all_bar_lengths[:, freqs_indices]
            all_labels = np.array([[labels[i]]*num_params for i in range(num_features)])
            
            title = "Prediction of Frequencies and Effects of Varying Parameters"
        elif param_name not in column_names:
            prefix = param_name
            modes = [mode for mode in column_names if mode.startswith(prefix)]
            QoI_names = modes
            num_params = len(modes)
            if num_params == 0:
                raise ValueError(f"There is no mode starting with {prefix}")
            modes_indices = [i for i, e in enumerate(column_names) if e in modes]
            all_expected_value = all_expected_value[modes_indices]
            all_bar_lengths = all_bar_lengths[:, modes_indices]
            all_labels = np.array([[labels[i]]*num_params for i in range(num_features)])
            
            title = f"Prediction of Mode {param_name} and Effects of Varying Parameters"
    
    # Sort bars by absolute value in descending order
    sorted_indices = np.argsort(np.abs(all_bar_lengths), axis=0)#[::-1]  # Indices for descending order
    sorted_all_labels = np.take_along_axis(all_labels, sorted_indices, axis=0)
    sorted_bar_lengths = np.take_along_axis(all_bar_lengths, sorted_indices, axis=0)
    
    fig, ax = plt.subplots(figsize=(num_params*1.8, 5))
    
    bar_width = (1-0.2)/num_features  # Width of each bar
    x_pos = np.arange(num_features * num_params)  # Positions for bars
    
    y_min = np.min(all_expected_value + np.cumsum(sorted_bar_lengths, axis=0))
    y_max = np.max(all_expected_value + np.cumsum(sorted_bar_lengths, axis=0))
    y_range = y_max - y_min
    y_range_ratio = (y_max - y_min)*0.22
    ax.set_ylim(y_min-y_range_ratio, y_max+y_range_ratio)

    all_colors = []  # Colors for bars
    all_bottoms = [] # y coordinates of bars
    all_predictions = all_expected_value + np.sum(sorted_bar_lengths, axis=0)

    bottom = all_expected_value
    for i in range(num_features):
        all_bottoms.append(list(bottom))
        colors = []
        bar_lengths = sorted_bar_lengths[i,:]
        bottom += bar_lengths
        
        for param_index in range(num_params):
            colors.append('#E53935' if bar_lengths[param_index] >= 0 else '#1E88E5')
        all_colors.append(colors)

    br = [np.arange(num_params)]
    for i in range(1, num_features):
        br.append([x + bar_width for x in br[i - 1]])
    
    ax.plot(br[-1], all_predictions, color='black', linewidth=0.8, marker='x', label="Prediction")
        
    for param_index in range(num_features):
        compensation = [-y_range*0.01 if x >= 0 else y_range*0.01 for x in sorted_bar_lengths[param_index]]
        bars = plt.bar((br[param_index]), sorted_bar_lengths[param_index], bottom=all_bottoms[param_index], color=all_colors[param_index], width=bar_width, label="_nolegend_")
    
        # Add triangles at the end of bars
        for bar, height, color in zip(bars, sorted_bar_lengths[param_index], all_colors[param_index]):
            x = bar.get_x() + bar.get_width()/2
            y = bar.get_y() + bar.get_height()
            half_bar_width = bar.get_width()/2 *0.9
            
            if height >= 0:
                triangle = np.array([[x - half_bar_width, y], [x + half_bar_width, y], [x, y + y_range*0.01]])
            else:
                triangle = np.array([[x - half_bar_width, y], [x + half_bar_width, y], [x, y - y_range*0.01]])
            
            ax.add_patch(plt.Polygon(triangle, color=color, label="_nolegend_"))

        # Add text labels
        for i, (x, bottom, height, color) in enumerate(zip(br[param_index], all_bottoms[param_index], sorted_bar_lengths[param_index], all_colors[param_index])):
            sign = '+' if height >= 0 else ''
            text_x_correction = y_range*0.03 if height >= 0 else -y_range*0.03
            ax.text(x, (bottom+height) + text_x_correction, f'{sign}{height:.2f}', rotation=90, ha='center', 
                    va='bottom' if height >= 0 else 'top', color=color, fontsize=9)
            
        
        if show_param_values:
            ax.plot([], [], ' ', label= f"{labels[param_index]} = {values[param_index]:.6f}")
    
    ax.plot([], [], label='Positive effect', marker='s', markersize=8,
            markeredgecolor='#E53935', markerfacecolor='#E53935', linestyle='')
    ax.plot([], [], label='Negative effect', marker='s', markersize=8,
            markeredgecolor='#1E88E5', markerfacecolor='#1E88E5', linestyle='')

    # Formatting
    x_pos = [item for sublist in br for item in sublist]
    ax.set_xticks(x_pos)
    ax.set_xticklabels(sorted_all_labels.flatten(), rotation=45, fontsize=10)
    
    # Add a second x-axis label
    br = np.array(br)
    group_labels = QoI_names
    group_positions = np.mean(br, axis=0)  # Midpoints for t_x labels

    for pos, label in zip(group_positions, group_labels):
        ax.text(pos, y_min - y_range_ratio*2, label, ha='center', fontsize=10, fontweight='bold')
    ax.set_ylabel("Prediction with SHAP Value Impact")
    ax.set_title(title, fontsize=14)
    ax.legend()
    ax.set_xlim(min(x_pos) - 2*bar_width, max(x_pos) + 2*bar_width)

    return fig

def plot_shap_beeswarm(model, param_name, q=None):
    if q is not None:
        explainer = model.model.explainer
        shap_values = explainer(q)
    else:
        shap_values = model.shap_values

    fig, ax = plt.subplots(figsize=(8, 6))
    
    if param_name is not None:
        param_index = model.QoI_names.index(param_name)
        QoI_shap = shap_values[:, :, param_index]

        ax.set_title(f"SHAP Explanation for {param_name} Quantity of Interest", fontsize=16)
        shap.summary_plot(QoI_shap, q, show=False)
    return fig

def plot_effects(effects, xticks=True):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(effects.transpose())
    ax.set_title('Predictions with Subtraced Effects', fontsize=16)
    if not xticks:
        ax.set_xticklabels([])
    ax.set_xlabel('Predicted features', fontsize=12)
    return fig
    
def plot_MCMC_results(Q, sampler, num_param, nwalkers, highlight_point=None, scale=1):
    param_bounds = Q.get_bounds()

    post_samples = sampler.get_chain(flat=True)

    # PairGrid plot
    h = sns.PairGrid(pd.DataFrame(post_samples[:, :], columns=Q.param_names()))
    h.map_diag(plt.hist, color="#2f779dff", bins=15, linewidth=0.3)
    h.map_upper(sns.scatterplot, color="#2f779dff", s=10, linewidth=0.3)
    h.map_lower(sns.scatterplot, color="#2f779dff", s=10, linewidth=0.3)

    # Adjust plot limits based on parameter bounds
    for i in range(num_param):
        for j in range(num_param):
            h.axes[i, j].set_xlim(param_bounds[j])
            if i != j:
                h.axes[i, j].set_ylim(param_bounds[i])
                # Add red scatter point if highlight_point is provided
                if highlight_point is not None:
                    highlight_point = highlight_point.flatten()
                    h.axes[i, j].scatter(
                        highlight_point[j],
                        highlight_point[i],
                        color="red",
                        s=15,
                        edgecolor="black",
                        zorder=5
                    )
    
def plot_MCMC(model_obj, DigitalTwin, nwalkers=None, extra_point=None, extra_point_name='Extra point', map_point=None, custom_xlim=None, custom_ylim=None):
    # Prepare data
    prior_samples = pd.DataFrame(model_obj.X_train, columns=model_obj.Q.param_names())
    posterior_samples = (
        pd.DataFrame(DigitalTwin.sampler.get_chain(flat=True), columns=model_obj.Q.param_names()) 
        if DigitalTwin else None
    )
    if extra_point is not None and not isinstance(extra_point, pd.DataFrame):
        extra_point = pd.DataFrame(extra_point, columns=model_obj.Q.param_names())
        
    mean_value = posterior_samples.mean()
    param_names = model_obj.Q.param_names()
    n_params = len(param_names)
    prior_color = '#b8b8b8'
    posterior_color = '#1a80bb'
    MAP_color = '#ea801c'
    
    # Combine datasets with labels
    prior_samples['Dataset'] = 'Prior'
    if posterior_samples is not None:
        posterior_samples['Dataset'] = 'Posterior'
        combined = pd.concat([prior_samples, posterior_samples])
    else:
        combined = prior_samples

    # Create pairplot
    fig = sns.pairplot(
        combined,
        hue='Dataset',
        palette={'Prior': prior_color, 'Posterior': posterior_color},
        diag_kind='kde',
        plot_kws={'alpha': 0.5, 's': 10},
        diag_kws={'alpha': 0.2, 'common_norm': False, 'zorder': 3}
    )

    # Add true/MAP points (without labels)
    #if selected_point is not None or map_point is not None:
    for i in range(n_params):
        for j in range(n_params):
            ax = fig.axes[i, j]
            if i != j:
                ax.scatter(mean_value[j], mean_value[i], 
                            color='red', s=10, zorder=1, marker='o')
                if map_point is not None:
                    ax.scatter(map_point.iloc[0, j], map_point.iloc[0, i], 
                                color=MAP_color, s=10, zorder=3, marker='o')  # No label here
                if extra_point is not None:
                    ax.scatter(extra_point.iloc[0, j], extra_point.iloc[0, i], 
                                color='purple', s=10, zorder=3, marker='o')  # No label here
            else:
                ax.vlines(mean_value[i], ax.get_ylim()[0], ax.get_ylim()[1], 
                            color='red', zorder=1)
                if map_point is not None:    
                    ax.vlines(map_point.iloc[0, i], ax.get_ylim()[0], ax.get_ylim()[1], 
                                color=MAP_color, zorder=1)
                if extra_point is not None:
                    ax.vlines(extra_point.iloc[0, i], ax.get_ylim()[0], ax.get_ylim()[1], 
                                color='purple', zorder=1)

    # Set axis limits using parameter bounds
    param_bounds = model_obj.Q.get_bounds()
    for i in range(n_params):
        for j in range(n_params):
            fig.axes[i, j].set_xlim(param_bounds[j])
            if i != j:
                fig.axes[i, j].set_ylim(param_bounds[i])

    # ========== KEY ADDITION: COMBINED LEGEND ==========
    # Create custom legend elements
    legend_elements = []
    
    # Add Dataset entries
    legend_elements.append(Line2D([0], [0], marker='o', color='w', 
                                markerfacecolor=prior_color, markersize=8, label='Prior'))
    if posterior_samples is not None:
        legend_elements.append(Line2D([0], [0], marker='o', color='w', 
                                    markerfacecolor=posterior_color, markersize=8, label='Posterior'))
    legend_elements.append(Line2D([0], [0], marker='o', color='w', 
                                    markerfacecolor='red', markersize=8, label='Mean Value'))
    
    # Add True/MAP entries if needed
    if map_point is not None:
        legend_elements.append(Line2D([0], [0], marker='o', color='w', 
                                    markerfacecolor=MAP_color, markersize=8, label='MAP Point'))
    if extra_point is not None:
        legend_elements.append(Line2D([0], [0], marker='o', color='w', 
                                    markerfacecolor='purple', markersize=8, label=extra_point_name))

    # Remove original legend and add combined one
    if fig._legend is not None:
        fig._legend.remove()
    fig.fig.legend(handles=legend_elements, loc='lower right', bbox_to_anchor=(0.95, 0.95))

    plt.suptitle('Prior and Posterior Distributions', y=1.02)
    return fig

def plot_posterior(model_obj, DigitalTwin, nwalkers, true_point=None, map_point=None, scale=1):
    if true_point is not None and type(true_point) != pd.DataFrame:
        true_point = pd.DataFrame(true_point, columns=model_obj.Q.param_names())
    sampler = DigitalTwin.sampler
    nwalkers = DigitalTwin.p0
    param_bounds = model_obj.Q.get_bounds()
    columns = model_obj.Q.param_names()
    num_param = model_obj.Q.num_params()

    post_samples = sampler.get_chain(flat=True)

    # PairGrid plot
    h = sns.PairGrid(pd.DataFrame(post_samples[:, :], columns=columns))
    h.map_diag(plt.hist, color="#2f779dff", bins=15, linewidth=0.3)
    h.map_upper(sns.scatterplot, color="#2f779dff", s=10, linewidth=0.3, alpha=0.1)
    h.map_lower(sns.scatterplot, color="#2f779dff", s=10, linewidth=0.3, alpha=0.1)

    # Adjust plot limits based on parameter bounds
    for i in range(num_param):
        for j in range(num_param):
            h.axes[i, j].set_xlim(param_bounds[j])
            if i != j:
                h.axes[i, j].set_ylim(param_bounds[i])
                # Add red scatter point if highlight_point is provided
                if true_point is not None:
                    #highlight_point = highlight_point.iloc[0]
                    #true_point = true_point.flatten()
                    h.axes[i, j].scatter(
                        true_point.iloc[0][j],
                        true_point.iloc[0][i],
                        color="red",
                        s=15,
                        #edgecolor="black",
                        zorder=5,
                        label="True Point" if i == 0 and j == 1 else ""
                    )
                if map_point is not None:
                    h.axes[i, j].scatter(
                        map_point.iloc[0][j],
                        map_point.iloc[0][i],
                        color="orange",
                        s=15,
                        #edgecolor="black",
                        zorder=5,
                        label="MAP Point" if i == 0 and j == 1 else ""
                    )
    h.fig.subplots_adjust(top=0.9)
    h.fig.suptitle('Posterior distributions')
    
    handles, labels = h.axes[0, 1].get_legend_handles_labels()
    h.fig.legend(handles, labels, loc='upper right')
    
    
    # Calculate and print means
    means = np.zeros((num_param, 1))
    for i in range(num_param):
        means[i] = np.mean(post_samples[:, i])

    print('Means from MCMC:')
    print(means * scale)
    
    return h.fig

def plot_prior(model_obj, true_point=None, scale=1):
    if true_point is not None and type(true_point) != pd.DataFrame:
        true_point = pd.DataFrame(true_point, columns=model_obj.Q.param_names())
    num_param = model_obj.Q.num_params()
    param_bounds = model_obj.Q.get_bounds()
    prios_samples = model_obj.X_train

    # PairGrid plot
    h = sns.PairGrid(prios_samples)
    h.map_diag(plt.hist, color="#2f779dff", bins=15, linewidth=0.3)
    h.map_upper(sns.scatterplot, color="#2f779dff", s=10, linewidth=0.3)
    h.map_lower(sns.scatterplot, color="#2f779dff", s=10, linewidth=0.3)

    # Adjust plot limits based on parameter bounds
    for i in range(num_param):
        for j in range(num_param):
            h.axes[i, j].set_xlim(param_bounds[j])
            if i != j:
                h.axes[i, j].set_ylim(param_bounds[i])
                # Add red scatter point if highlight_point is provided
                if true_point is not None:
                    #true_point = true_point.flatten()
                    h.axes[i, j].scatter(
                        true_point.iloc[0][j],
                        true_point.iloc[0][i],
                        color="red",
                        s=15,
                        #edgecolor="black",
                        zorder=5,
                        label="True Point" if i == 0 and j == 1 else ""
                    )
                    
    h.fig.subplots_adjust(top=0.9)
    h.fig.suptitle('Prior distributions')
    
    handles, labels = h.axes[0, 1].get_legend_handles_labels()
    h.fig.legend(handles, labels, loc='upper right')
    
    return h.fig
    
def plot_remained_effects(higher_bound, lower_bound, margin, higher_measured_bound, lower_measured_bound, QoI_names, xticks=True):
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(0, len(higher_bound))
    ax.fill_between(x, higher_bound + margin, lower_bound - margin, color = '#1a80bb', alpha=0.3, label='Acceptable region')

    ax.fill_between(x, higher_measured_bound, lower_measured_bound, color = 'gray', alpha=0.3)


    ax.plot(higher_bound, color='#1a80bb', label='Reference min-max')
    ax.plot(lower_bound, color='#1a80bb')

    ax.plot(higher_measured_bound, color ='black', label='Monitoring min-max')
    ax.plot(lower_measured_bound, color ='black')

    up_up_indexes = np.where(higher_measured_bound > higher_bound + margin)[0]
    up_low_indexes = np.where(higher_measured_bound < lower_bound - margin)[0]
    low_up_indexes = np.where(lower_measured_bound > higher_bound + margin)[0]
    low_low_indexes = np.where(lower_measured_bound < lower_bound - margin)[0]
    
    upper_indexes = np.sort(np.concatenate((up_up_indexes, up_low_indexes)))
    lower_indexes = np.sort(np.concatenate((low_up_indexes, low_low_indexes)))

    if (len(upper_indexes) > 0) and (len(lower_indexes) > 0):
        ax.plot(upper_indexes, higher_measured_bound[upper_indexes], 'ro', label='Point outside acceptable region')
        ax.plot(lower_indexes, lower_measured_bound[lower_indexes], 'ro')

    elif len(upper_indexes) > 0:
        ax.plot(upper_indexes, higher_measured_bound[upper_indexes], 'ro', label='Point outside acceptable region')
    elif len(lower_indexes) > 0:
        ax.plot(lower_indexes, lower_measured_bound[lower_indexes], 'ro', label='Point outside acceptable region')

    if xticks:
        ax.set_xticks(x, QoI_names)
    else:
        ax.set_xticklabels([])
    ax.set_xlabel('Predicted features', fontsize=12)
    fig.legend()
    return fig

def plot_2D_with_plane_highlight(K, M, Z, Y_m, epsilon=0.1):
    # Create a custom colormap (white → blue)
    colors = [(1, 1, 1), (0, 0, 1)]  # white, blue
    cmap = LinearSegmentedColormap.from_list("custom_cmap", colors, N=256)
    
    # Create a mask to highlight values within Z = Y_m ± epsilon
    mask = np.abs(Z - Y_m) <= epsilon
    Z_highlight = np.where(mask, Z, np.nan)  # Retain only values in the range
    
    # Plot the highlighted region
    plt.imshow(Z_highlight, extent=(K.min(), K.max(), M.min(), M.max()), origin='lower',
               cmap=cmap, aspect='auto')
    
    # Add a contour line where Z = Y_m
    plt.contour(K, M, Z, levels=[Y_m], colors='blue', linewidths=2)
    
    # Add axis labels and a title
    #plt.colorbar(label='Z values in highlight range')  # Uncomment if a colorbar is needed
    plt.xlabel('K')
    plt.ylabel('M')
    plt.title(f'Highlight around Z = {Y_m.item()} ± {epsilon}')
    plt.grid(True)


def plot_multibuilding_MCMC(jointManager, extra_point=None, extra_point_name='Extra point', map_point=True, model_names=None, formatted_param_names=None):
    figures = []
    for m in range(len(jointManager.models)):
        prior_samples = pd.DataFrame(jointManager.models[m].X_train, columns=jointManager.models[m].Q.param_names())
        posterior_samples = pd.DataFrame(jointManager.sampler.get_chain(flat=True)[:, jointManager.indices[m]] * jointManager.scale[m] + jointManager.shift[m], columns=jointManager.models[m].Q.param_names())
        
        param_names = jointManager.models[m].Q.param_names()
        mean_value = posterior_samples.mean()
        n_params = len(param_names)
        prior_color = '#b8b8b8'
        posterior_color = '#1a80bb'
        MAP_color = '#ea801c'

        if map_point:
            map_p = []
            for p in range(len(jointManager.indices[m])):
                map_p.append(estimate_maxima(jointManager.sampler.get_chain(flat=True)[:, jointManager.indices[m][p]] * jointManager.scale[m][p] + jointManager.shift[m][p]))
                # map_p = jointManager.get_MAP(m).values[:, jointManager.indices[m]] * jointManager.scale[m] + jointManager.shift[m]

            # Combine datasets with labels
        prior_samples['Dataset'] = 'Prior'
        if posterior_samples is not None:
            posterior_samples['Dataset'] = 'Posterior'
            combined = pd.concat([prior_samples, posterior_samples])
        else:
            combined = prior_samples

        # Create pairplot
        fig = sns.pairplot(
            combined,
            hue='Dataset',
            palette={'Prior': prior_color, 'Posterior': posterior_color},
            diag_kind='kde',
            plot_kws={'alpha': 0.5, 's': 10},
            diag_kws={'alpha': 0.2, 'common_norm': False, 'zorder': 3}
        )

        # Add true/MAP points (without labels)
        #if selected_point is not None or map_point is not None:
        for i in range(n_params):
            for j in range(n_params):
                ax = fig.axes[i, j]
                if formatted_param_names is not None and j == 0:
                    ax.set_ylabel(formatted_param_names[m][i])

                if formatted_param_names is not None and i == n_params-1:
                    ax.set_xlabel(formatted_param_names[m][j])
                    ax.tick_params(axis='x', labelrotation=90)

                if i != j:
                    ax.scatter(mean_value[j], mean_value[i], 
                            color='red', s=10, zorder=1, marker='o')
                    if map_point:
                        # map_p = jointManager.models[m].get_MAP()
                        ax.scatter(map_p[j], map_p[i], 
                                    color=MAP_color, s=10, zorder=3, marker='o')  # No label here
                    if extra_point is not None:
                        ax.scatter(extra_point.iloc[0, j], extra_point.iloc[0, i], 
                                    color='purple', s=10, zorder=3, marker='o')  # No label here
                else:
                    ax.vlines(mean_value[i], ax.get_ylim()[0], ax.get_ylim()[1], 
                                color='red', zorder=1)
                    if map_point:
                        # map_p = jointManager.models[m].get_MAP()
                        ax.vlines(map_p[i], ax.get_ylim()[0], ax.get_ylim()[1], 
                                    color=MAP_color, zorder=1)
                    if extra_point is not None:
                        ax.vlines(extra_point.iloc[0, i], ax.get_ylim()[0], ax.get_ylim()[1], 
                                    color='purple', zorder=1)

        # Set axis limits using parameter bounds
        param_bounds = jointManager.models[m].Q.get_bounds()
        for i in range(n_params):
            for j in range(n_params):
                fig.axes[i, j].set_xlim(param_bounds[j])
                if i != j:
                    fig.axes[i, j].set_ylim(param_bounds[i])

        # ========== KEY ADDITION: COMBINED LEGEND ==========
        # Create custom legend elements
        legend_elements = []
        
        # Add Dataset entries
        legend_elements.append(Line2D([0], [0], marker='o', color='w', 
                                    markerfacecolor=prior_color, markersize=8, label='Prior'))
        if posterior_samples is not None:
            legend_elements.append(Line2D([0], [0], marker='o', color='w', 
                                        markerfacecolor=posterior_color, markersize=8, label='Posterior'))
        
        legend_elements.append(Line2D([0], [0], marker='o', color='w', 
                                    markerfacecolor='red', markersize=8, label='Mean Value'))
        
        # Add True/MAP entries if needed
        if map_point:
            legend_elements.append(Line2D([0], [0], marker='o', color='w', 
                                        markerfacecolor=MAP_color, markersize=8, label='MAP Point'))
        if extra_point is not None:
            legend_elements.append(Line2D([0], [0], marker='o', color='w', 
                                        markerfacecolor='purple', markersize=8, label=extra_point_name))

        # Remove original legend and add combined one
        if fig._legend is not None:
            fig._legend.remove()
        fig.fig.legend(handles=legend_elements, loc='lower right', bbox_to_anchor=(0.95, 0.95))

        if model_names is None:
            plt.suptitle('Prior and Posterior Distributions of model {}.'.format(m+1), y=1.02)
        else:
            plt.suptitle('Prior and Posterior Distributions of the {} model'.format(model_names[m]), y=1.02)
        figures.append(fig)
    return figures

def estimate_maxima(data):
    kde = gaussian_kde(data)
    no_samples = 200
    samples = np.linspace(min(data), max(data), no_samples)
    probs = kde.evaluate(samples)
    maxima_index = probs.argmax()
    maxima = samples[maxima_index]
    
    return maxima


def plot_update_points_and_surface(Q, sampler, K, M, Z, Y_m, highlight_point, epsilon=0.1):
    param_x, param_y = 1, 0
    param_bounds = Q.get_bounds()
    post_samples = sampler.get_chain(flat=True)
    
    # Paraméterek kiválasztása
    x_samples = post_samples[:, param_x]
    y_samples = post_samples[:, param_y]
    
    # Új scatter plot
    plt.figure(figsize=(6, 6))
    sns.scatterplot(x=x_samples, y=y_samples, color="#2f779dff", s=10, linewidth=0.3, alpha=0.1)
    
    # Paraméterek korlátai
    plt.xlim(param_bounds[param_x])
    plt.ylim(param_bounds[param_y])
    
    # Kiugró pont hozzáadása
    if highlight_point is not None:
        highlight_point = highlight_point.flatten()
        plt.scatter(
            highlight_point[param_x],
            highlight_point[param_y],
            color="red",
            s=15,
            edgecolor="black",
            zorder=5
        )
    
    # Create a custom colormap (white → blue)
    colors = [(1, 1, 1), (0, 0, 1)]  # white, blue
    cmap = LinearSegmentedColormap.from_list("custom_cmap", colors, N=256)
    
    # Create a mask to highlight values within Z = Y_m ± epsilon
    mask = np.abs(Z - Y_m) <= epsilon
    Z_highlight = np.where(mask, Z, np.nan)  # Retain only values in the range
    
    # Plot the highlighted region
    plt.imshow(Z_highlight, extent=(K.min(), K.max(), M.min(), M.max()), origin='lower',
               cmap=cmap, aspect='auto')
    
    # Add a contour line where Z = Y_m
    plt.contour(K, M, Z, levels=[Y_m], colors='blue', linewidths=2)
    
    # Add axis labels and a title
    #plt.colorbar(label='Z values in highlight range')  # Uncomment if a colorbar is needed
    plt.xlabel('K')
    plt.ylabel('M')
    plt.title(f'Highlight around Z = {Y_m.item()} ± {epsilon}')
    plt.grid(True)

def plot_grid_update_points_and_surfaces(Q, sampler, model, y_m, q_true, epsilon=0.1):
        if type(q_true) == pd.DataFrame:
            q_true = q_true.to_numpy().squeeze()
        if type(y_m) == pd.DataFrame:
            y_m = y_m.to_numpy().squeeze()
            
        names = Q.param_names()
        bounds = Q.get_bounds()

        n_params = len(names)
        n_features = len(y_m)

        epsilon = epsilon

        post_samples = sampler.get_chain(flat=True)

        _, ax = plt.subplots(n_params, n_params, figsize=(10, 10))
        for i in range(n_params):
            for j in range(n_params):
                ax[i, j].set_xlim(bounds[j])
                if i != j:
                    ax[i, j].set_ylim(bounds[i])
                    x_samples = post_samples[:, j]
                    y_samples = post_samples[:, i]

                    ax[i, j].scatter(x=x_samples, y=y_samples, color="#2f779dff", s=10, linewidth=0.3, alpha=0.1)
                    ax[i, j].scatter(q_true[j], q_true[i], color='red', zorder=3)

                    x_grid = np.linspace(bounds[j, 0], bounds[j, 1], 200)
                    y_grid = np.linspace(bounds[i, 0], bounds[i, 1], 200)

                    X, Y = np.meshgrid(x_grid, y_grid)
                    array_len = len(X.reshape(-1))
                    param_array = np.zeros((array_len, n_params))
                    for k in range(n_params):
                        if k == j:
                            param_array[:, k] = X.reshape(-1)
                        elif k == i:
                            param_array[:, k] = Y.reshape(-1)
                        else:
                            param_array[:, k] = np.ones(array_len) * q_true[k]

                    Z_model = model.predict(param_array)
                    Z_model = Z_model.reshape(X.shape[0], X.shape[1], -1)

                    for f in range(n_features):
                        f_Z = Z_model[:, :, f]
                        f_y = y_m[f]
                        #ax[i, j].contour(X, Y, f_Z, levels=[f_y], colors='blue', linewidths=2)

                    ax[i, j].grid(True)

                else:
                    ax[i, j].hist(post_samples[:, i], color="#2f779dff", bins=15, linewidth=0.3)
                if j == 0:
                    ax[i, j].set_ylabel(names[i])
                if i == n_params-1:
                    ax[i, j].set_xlabel(names[j])


############################################################################################
#                           SimParamSet from data
############################################################################################

def gaussian(x, mu, sig):
    return 1./(np.sqrt(2.*np.pi)*sig)*np.exp(-np.power((x - mu)/sig, 2.)/2)

def get_distribution(X, parameter_idx, verbose=False):
    hist, bin_edges = np.histogram((X.values[:, parameter_idx]), density=True, bins=50)
    mu = np.mean(X.values[:, parameter_idx])
    sigma = np.std(X.values[:, parameter_idx])
    gaussian_curve = np.zeros(len(bin_edges))
    uniform_line = np.zeros(len(bin_edges))
    for i in range(len(bin_edges)):
        gaussian_curve[i] = gaussian(bin_edges[i], mu, sigma)
        uniform_line[i] = np.mean(hist)
    gaussian_error = np.mean(np.sqrt((hist - gaussian_curve[:-1])**2))
    uniform_error = np.mean(np.sqrt((hist - uniform_line[:-1])**2))
    if verbose:
        print('Gaussian error: {}'.format(gaussian_error))
        print('Uniform error: {}'.format(uniform_error))
    if gaussian_error > uniform_error:
        return 'Uniform'
    else:
        return 'Normal'
    
def get_simparamset_from_data(data, verbose=False):
    Q = SimParamSet()
    for i in range(data.values.shape[1]):
        dist = get_distribution(data, i)    
        if dist == 'Uniform':
            l_bound = np.min(data.values, axis=0)[i]
            u_bound = np.max(data.values, axis=0)[i]
            P = SimParameter(data.columns[i], UniformDistribution(l_bound, u_bound))
            if verbose:
                print('{} - UniformDistribution({}, {})'.format(data.columns[i], l_bound, u_bound))
        elif dist == 'Normal':
            mu = np.mean(data.values[:, i])
            sigma = np.std(data.values[:, i])
            P = SimParameter(data.columns[i], NormalDistribution(mu, sigma))
            if verbose:
                print('{} - NormalDistribution({}, {})'.format(data.columns[i], mu, sigma))
        Q.add(P)
    return Q

def generate_stdrn_simparamset(sigma):
    Q = SimParamSet()
    for i in range(len(sigma)):
        s = SimParameter('pn_' + str(i+1), NormalDistribution(0, sigma[i]))
        Q.add(s)
    return Q

# ##########################################################################################
#                           Load data
# ##########################################################################################
# def load_data():
#     # Pathname where data is stored
#     data_path = "/home/noemi/windoc/Projects/SFB880/flow_trhough_porous_media/SFB_project/points_and_solutions"
#     # directories to read in
#     dirs = ["/first_scan_points", "/best_point", "/second_phase_points"]
#     # "/best_point/island_around"
#     q_l = []
#     y_l = []
#     for dir in dirs:
#         file_name = data_path + dir + '/points_and_sol.mat'
#         q_i, y_i = read_in_data(file_name)
#         q_l.append(q_i)
#         y_l.append(y_i)
#     q = np.concatenate(q_l, axis=0)
#     y = np.concatenate(y_l, axis=0)
#     # take out the obvious outlyer from from the data
#     ind_bad = np.argmax(y[:,150, 1])
#     q = np.delete(q, ind_bad, axis=0)
#     y = np.delete(y, ind_bad, axis=0)
#     return q, y

# def read_in_data(file_name):
#     data = loadmat(file_name)
#     y = data["u"].astype(np.float32)
#     if y.ndim>2:
#         y = np.rollaxis(y, 2, 0)
#     else:
#         y = np.expand_dims(y, axis=0)
#     q = data["x"].astype(np.float32)  # input data (model coefficients)
#     q = np.rollaxis(q, 1, 0)

#     if 'nonconv_ind' in data.keys():
#         conv_ind = ~(np.in1d(np.arange(len(y)), data['nonconv_ind'] - 1))
#         y = y[conv_ind]  # output data (response)
#         q = q[conv_ind]
#     return q, y

# ##########################################################################################
#                           Load experimental data
# ##########################################################################################
# def load_experiment_sol_and_coord():
#     # Pathname where data is stored
#     data_path = '/home/noemi/windoc/Projects/SFB880/flow_trhough_porous_media/SFB_project/points_and_solutions/experiment/exp_sol_and_coord'
#     # Load data
#     data = loadmat(data_path)
#     y = data["u"].astype(np.float32)
#     x = data["y"].astype(np.float32)  # bemeno adat (model coefficients)
#     return x, y

# ##########################################################################################
#                           Set prior paramset
# ##########################################################################################
# def set_prior_paramset():
#     Q = SimParamSet()
#     Q.add(SimParameter('beta', UniformDistribution(-10, 0)))
#     Q.add(SimParameter('beta_T', UniformDistribution(-1, 1)))
#     Q.add(SimParameter('c_t', UniformDistribution(0, 0.2)))
#     Q.add(SimParameter('c_eh', UniformDistribution(0, 0.3)))
#     Q.add(SimParameter('c_wd', UniformDistribution(0, 30)))
#     Q.add(SimParameter('c_dp', UniformDistribution(0, 0.4)))
#     return Q

# ##########################################################################################
#                           Y scaler
# ##########################################################################################
class y_scaler_svd():
    def __init__(self, y, n):
        from sklearn.decomposition import PCA
        [n_s, n_x, n_var] = y.shape
        y = y.reshape([n_s, n_x * n_var])
        self.n_s = n_s
        self.n_x = n_x
        self.n_var = n_var
        self.n_svd = n
        self.pca = PCA(n, whiten=True)
        self.pca.fit(y)
        self.P = self.pca.components_
        self.S = np.diag(self.pca.explained_variance_)

    def scale_reshape_y(self, y):
        if len(y.shape) == 2:
            y = np.reshape(y, (1, -1))
        else:
            y = np.reshape(y, (y.shape[0], -1))
        # y = np.dot(np.dot(y-self.pca.mean_, self.P.T), np.linalg.inv(self.S))
        u = self.pca.transform(y)
        return u

    def reshape_rescale_u(self, u):
        y = self.pca.inverse_transform(u)
        # y = np.dot( np.dot(u, self.S), self.P)
        y = np.reshape(y, (-1, self.n_x, self.n_var))
        return y

# class y_scaler():
#     def __init__(self, y):
#         self.min_y = np.min(y, axis=(0, 1))
#         self.range_y = np.max(y, axis=(0, 1))-self.min_y
#         [n_s, n_x, n_var] = y.shape
#         self.n_s = n_s
#         self.n_x = n_x
#         self.n_var = n_var

#     def scale_reshape_y(self, y):
#         y = y - self.min_y
#         y = 2 * y / self.range_y
#         y = y - 1
#         y = np.reshape(y, (y.shape[0], -1))
#         return y

#     def reshape_rescale_y(self, y):
#         y = np.reshape(y, (-1, self.n_x, self.n_var))
#         y = y + 1
#         y = y * self.range_y / 2
#         y = y + self.min_y
#         return y

class y_scaler():
    def __init__(self, y):
        self.min_y = np.min(y, axis=1)
        self.range_y = np.max(y, axis=1)-self.min_y
        [n_var, n_s] = y.shape
        self.n_s = n_s
        self.n_var = n_var

    def scale_reshape_y(self, y):
        y = y - self.min_y
        y = 2 * y / self.range_y
        y = y - 1
        y = np.reshape(y, (y.shape[0], -1))
        return y

    def reshape_rescale_y(self, y):
        y = np.reshape(y, (-1, self.n_var))
        y = y + 1
        y = y * self.range_y / 2
        y = y + self.min_y
        return y


def test_paramset(simparamset, sample):
    """
    Performs Kolmogorov-Smirnov test to ensure that the sample matches the distributions defined in the SimParamSet.
    :param simparamset: SimParamSet object containing the parameter distributions
    :param sample: DataFrame containing the sample data
    """
    if len(sample.columns) != simparamset.num_params():
        return {"valid": False, "message": "Number of columns in sample must match number of parameters in SimParamSet."}
    
    for param in simparamset.params.keys():
        dist = simparamset.params[param]
        ks_statistic, p_value = stats.kstest(sample[param], dist.cdf)
        if p_value < 0.05:
            return {"valid": False, "message": f"Sample does not match the distribution of parameter {param}."}
        
    return {"valid": True, "message": "Correct"}


# ##########################################################################################
#                           measurement noise
# ##########################################################################################
# def stds_of_measurement_noise(x, y_exp, scale_error=1):
#     sigma_scale = [0.5, 2, 2, 2, 1.5]
#     # percentages of error of the DNS results
#     perc_sigma = 3 * np.ones([len(x),1]) + 2 * np.exp(-(x-0.9)**2/(2*(0.1/3)**2))
#     err_sigma = np.max(abs(y_exp), axis=0) * sigma_scale * perc_sigma / 200 * scale_error
#     return err_sigma

# ##########################################################################################
#                           plot response
# ##########################################################################################
# def plot_response(err, y_exp=[], x=[]):
#     fig,a = plt.subplots(5,1)
#     if len(x) == 0:
#         x = np.arange(err.shape[1])
#     for i in range(5):
#         a[i].plot(x, err[:, :, i].transpose(), '--')
#         if len(y_exp) != 0:
#             a[i].plot(x, y_exp[:, i], 'k', linewidth=2)
#         a[i].axhline(y=0., color='k', linestyle='-', linewidth=2)

# ##########################################################################################
#                           MATRIXPLOT FOR PARAM VALUES
# ##########################################################################################
# def plot_param_values(q, Q, ind_gauss):
#     import seaborn as sns
#     import pandas as pd
#     n = q.shape[0]
#     sns.set_theme(style="ticks")
#     #sns.set_theme(style="white")
#     # generate prior samples and give an identifyer
#     q_prior = Q.sample(n).T


#     # catenate prior with samples and add indices
#     T = np.concatenate((q_prior, q), axis=0)
#     df = pd.DataFrame(T, columns=Q.param_names())

#     ind_q = ["q_prior"] * n
#     ind_q.extend(["q_post"] * n)
#     ind_q[0] ="q_MAP"
#     ind_q[1] = "q_mean"
#     df['samples'] = ind_q

#     df['samples'] = df['samples'].astype(str)

#     g = sns.pairplot(df, vars=Q.param_names(), hue='samples', diag_kind="hist")
#     for i, ax in enumerate(g.axes):
#         ax[0].set_ylim(Q.params[Q.param_names()[i]].a, Q.params[Q.param_names()[i]].b)
#     for i, ax in enumerate(g.axes.T):
#         ax[0].set_xlim(Q.params[Q.param_names()[i]].a, Q.params[Q.param_names()[i]].b)

# if __name__ == "__main__":
#     load_data()