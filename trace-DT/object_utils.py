import numpy as np
from simparameter_set import SimParamSet
from simparameter import SimParameter
from distributions import UniformDistribution
from distributions import NormalDistribution
#from gpc_surrogate import GpcSurrogateModel
from matplotlib.ticker import ScalarFormatter
import matplotlib.pyplot as plt
import emcee
import time
import seaborn as sns
import pandas as pd
import shap

def gpc_multiindex2param_names(I_s, param_names):
    """
    Translates a multi-index set to a list of indexed parameter names.

    Parameters:
        I_s (numpy.ndarray): Multi-index set (rows are indices, columns correspond to dimensions).
        param_names (list of str): List of parameter names.

    Returns:
        list of str: List of indexed parameter names.
    """
    indexed_param_names = [None] * I_s.shape[0]  # Initialize output list
    max_d = np.max(np.sum(I_s, axis=1))  # Maximum Sobol order
    min_d = np.min(np.sum(I_s, axis=1))  # Minimum Sobol order

    ind_start = 0
    for i in range(min_d, max_d + 1):
        # Extract rows corresponding to the current degree (i)
        I_s_i = I_s[np.sum(I_s, axis=1) == i, :]
        l_i = I_s_i.shape[0]

        # Find indices where I_s_i equals 1
        indr, indc = np.where(I_s[np.sum(I_s, axis=1) == i, :] == 1)

        # Sort and reshape indices into groups
        IND = np.column_stack((indr, indc))
        IND = IND[np.argsort(IND[:, 0])]
        IND = IND[:, 1].reshape((l_i, i))  # Extract column indices

        # Map indices to parameter names
        param_names_i = ["".join([param_names[idx] for idx in row]) for row in IND]

        # Add spaces between parameter names if degree > 1
        if i != 1:
            p_names_i = []
            for row in IND:
                p_names_i.append(" ".join(param_names[idx] for idx in row))
            indexed_param_names[ind_start:ind_start + l_i] = p_names_i
        else:
            indexed_param_names[ind_start:ind_start + l_i] = param_names_i

        ind_start += l_i
    return indexed_param_names

# def plot_sobol_sensitivity(partial_variance, sobol_index, y, param_name=None):
#     y_var = np.broadcast_to(y.var(axis=0).to_numpy().reshape(-1,1), sobol_index.shape)
#     y_var_df = pd.DataFrame(y_var, columns=partial_variance.columns, index=partial_variance.index)
#     partial_variance = np.multiply(sobol_index, y_var)
#     # Colors for plots
#     color_map = plt.cm.viridis(np.linspace(0, 1, partial_variance.shape[1]+1))
#     colors = {partial_variance.columns[i]: color_map[i] for i in range(len(partial_variance.columns))}
    
#     if param_name is not None:
#         # Making a dataframe for calculating percentage and thresholds
#         threshold = 0.1
#         loc_par_var = partial_variance.loc[param_name]
#         df = pd.DataFrame(loc_par_var)
#         df['percentage'] = df[param_name]/df[param_name].sum()
#         under_threshold = df.loc[df['percentage'] < threshold].sum()
#         remaining = [y_var_df.loc[param_name][0], 1] - df.sum()
#         others = under_threshold + remaining
#         colors['others'] = color_map[-1]
#         df = df[df['percentage'] >= threshold]
#         df.loc['others'] = others
#         pie_colors = [colors[x] for x in df.index]
        
#         plt.title("Used Quantity of Interest parameter: " + str(param_name), fontsize=16)
#         plt.pie(df[param_name], labels=df.index, colors=pie_colors, wedgeprops={"alpha": 0.5})
#         plt.show()        
#         pass
    
    
#     timesteps = sobol_index.shape[0]
#     time = np.arange(0, timesteps, 1)
#     # Stacked area plot létrehozása
#     plt.figure(figsize=(8, 6))

#     # plt.yticks(np.linspace(0,7.0*10**(-7),36), minor=True)
#     # plt.yticks(np.linspace(0,7.0*10**(-7),8), minor=False)

#     # plt.xticks(np.linspace(0,timesteps,21), minor=True)
#     # plt.xticks(np.linspace(0,timesteps,5), minor=False)

#     plt.tick_params(axis = 'both', which = 'minor', labelsize = 0)
#     plt.tick_params(axis = 'both', which = 'major', labelsize = 12)
#     plt.grid(which = 'minor', alpha = 0.3)
#     plt.grid(which = 'major', alpha = 0.3)

#     # Teljes variancia plot
#     plt.plot(time, y_var[:,0], color='black', label='Total variance', linewidth=0.8)
    
#     # Stacked area plot
#     stackplot_colors = [colors[x] for x in y_var_df.columns]
#     plt.stackplot(time, partial_variance.to_numpy().transpose(), alpha=0.5, labels=sobol_index.columns, colors=stackplot_colors)

#     # Use ScalarFormatter and set the format
    
#     formatter = ScalarFormatter(useMathText=True)
#     plt.gca().yaxis.set_major_formatter(formatter)

#     # Grafikon beállítások
#     plt.xlim((0,timesteps-1))
#     #plt.ylim((0,7.0*10**(-7)))
#     plt.xlabel('Timestep', fontsize=14)
#     plt.ylabel('Variance', fontsize=14)
#     plt.legend(fontsize=14)

#     #plt.grid(np.linspace(0,100,20),np.linspace(0,3.5*10**(-5),35))
#     plt.show()

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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
    fig, ax = plt.subplots(figsize=(8, 6))
 
    y_var = np.broadcast_to(y.var(axis=0).to_numpy().reshape(-1, 1), sobol_index.shape)
    #total_var = np.broadcast_to(total_var, sobol_index.shape)
    y_var_df = pd.DataFrame(y_var, columns=sobol_index.columns, index=sobol_index.index)
    partial_variance = np.multiply(sobol_index, y_var)

    color_map = plt.cm.viridis(np.linspace(0, 1, partial_variance.shape[1]+1))
    colors = {partial_variance.columns[i]: color_map[i] for i in range(len(partial_variance.columns))}

    if param_name is not None:
        # Handle specific parameter case
        threshold = 0.1
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

        ax.set_title(f"Used Quantity of Interest parameter: {param_name}", fontsize=16)
        ax.pie(df[param_name], labels=df.index, colors=pie_colors, wedgeprops={"alpha": 0.5})
    else:
        # General case: Stacked area plot
        timesteps = sobol_index.shape[0]
        time = np.arange(0, timesteps, 1)

        # Total variance plot
        ax.plot(time, y_var[:, 0], color='black', label='Total variance', linewidth=0.8)

        # Stacked area plot
        stackplot_colors = [colors[col] for col in y_var_df.columns]
        ax.stackplot(time, partial_variance.to_numpy().transpose(), alpha=0.5, labels=sobol_index.columns, colors=stackplot_colors)

        # Format the plot
        ax.set_xlim((0, timesteps - 1))
        ax.set_xlabel('Timestep', fontsize=14)
        ax.set_ylabel('Variance', fontsize=14)
        ax.legend(fontsize=14)
        ax.grid(True, alpha=0.3)

    return fig




def plot_shap_values(model, q=None, param_name=None):
    """
    Plots SHAP values with optional axis input.

    Parameters:
        model: Model with SHAP explainer.
        q (pd.DataFrame, optional): Input data (optional).
        param_name (str, optional): Specific parameter name to plot (optional).
        ax (matplotlib.axes.Axes, optional): Axis to plot on (optional).
    """
    if q is None:
        q = model.X_train
    else:
        q = pd.DataFrame(q, columns=model.Q.param_names())
    
    #xi = model.get_scaled_q(q)
    #xi = pd.DataFrame(xi, columns=model.Q.param_names())
    #explainer = model.model.explainer
    xi = q
    explainer = model.model.explainer
    shap_values = explainer(xi)

    fig, ax = plt.subplots(figsize=(8, 6))
    
    if param_name is not None:
        param_index = model.QoI_names.index(param_name)
        QoI_shap = shap_values[:, :, param_index]

        if QoI_shap.shape[0] == 1:
            ax.set_title(f"Used Quantity of Interest parameter: {param_name}", fontsize=16)
            shap.plots.waterfall(QoI_shap[0, :], show=False)
        else:
            ax.set_title(f"Used Quantity of Interest parameter: {param_name}", fontsize=16)
            shap.summary_plot(QoI_shap, xi, show=False)
    else:
        ax.set_title(f"Average Influence of {shap_values.shape[0]} sample of Parameters on  Quantity of Interest", fontsize=16)
        shap.summary_plot(shap_values[:, :, 0], xi, plot_type="bar", show=False)

    return fig



# def plot_shap_values(model, q=None, param_name=None):
#     if q is None:
#         q = model.X_train
#     else:
#         q = pd.DataFrame(q, columns=model.Q.param_names())
#     xi = model.get_scaled_q(q)
#     xi = pd.DataFrame(xi, columns=model.Q.param_names())
#     explainer = model.model.explainer
#     shap_values = explainer(xi)
    
#     if param_name is not None:
#         param_index = model.QoI_names.index(param_name)
#         QoI_shap = shap_values[:,:,param_index]
    
#         if QoI_shap.shape[0] == 1:
#             plt.title("Used Quantity of Interest parameter: " + str(param_name), fontsize=16)
#             #shap.plots.waterfall(shap_values[0,:,param_index])
#             shap.plots.waterfall(QoI_shap[0,:])
#         else:
#             plt.title("Used Quantity of Interest parameter: " + str(param_name), fontsize=16)
#             #shap.plots.waterfall(shap_values[:,:,param_index])
#             shap.summary_plot(QoI_shap, xi)
#     else:
#         #plt.title("Used Quantity of Interest parameter: " + str(param_name), fontsize=16)
#         # shap.summary_plot(QoI_shap, xi)
#         #shap.summary_plot(shap_values, xi)
#         #shap.plots.force(explainer.expected_value[0], shap_values.values[:,:,0])
#         #shap.plots.bar(shap_values[0])
#         plt.title("Average Influence of Features on Outputs")
#         shap.summary_plot(shap_values[:,:,0], xi, plot_type="bar")

#         # shap.summary_plot(shap_values.values[0])
#         # shap.summary_plot(shap_values[:,:,0], xi, plot_type="bar")
#         # shap.summary_plot(shap_values, xi, plot_type="bar")
#         # Átlagolt hatások kiszámítása jellemzőnként

#     # plt.title("Used Quantity of Interest parameter: " + str(param_name), fontsize=16)
#     # shap.summary_plot(QoI_shap, xi, plot_type="bar")

#     plt.show()
    

def plot_MCMC_results(Q, sampler, num_param, nwalkers, scale):
    param_bounds = Q.get_bounds()

    # post_samples = sampler.get_chain(flat = False)
    # fig, ax = plt.subplots(num_param, figsize = (10, 10))
    # for j in range(nwalkers):
    #     for i in range(Q.num_params()):
    #         ax[i].plot(post_samples[:, j, i])

    ###
    post_samples = sampler.get_chain(flat = True)
    h = sns.PairGrid(pd.DataFrame(post_samples[:, :], columns = Q.param_names()))
    h.map_diag(plt.hist, color="#2f779dff", bins = 15, linewidth = 0.3)
    h.map_upper(sns.scatterplot, color = "#2f779dff", s = 10, linewidth=0.3)
    h.map_lower(sns.scatterplot, color = "#2f779dff", s = 10, linewidth=0.3)
    for i in range(num_param):
        for j in range(num_param):
            h.axes[i, j].set_xlim(param_bounds[j])
            if i != j:
                h.axes[i, j].set_ylim(param_bounds[i])

    means = np.zeros((num_param, 1))
    for i in range(num_param):
        means[i] = np.mean(post_samples[:, i])

    print('Means from MCMC:')
    print(means*scale)
    plt.show()

def MCMC_run_and_plot(nwalkers, niter, nburn, scale, manager):

    Q = manager.Q
    num_param = Q.num_params()
    pdf_func = manager.pdf_func

    p0 = Q.sample(nwalkers)#.T
    
    print('MCMC creating')
    sampler = emcee.EnsembleSampler(nwalkers, num_param, pdf_func)
    start_time = time.time()

    print('Burning period')
    state = sampler.run_mcmc(p0, nburn, progress = True)
    sampler.reset()

    print('MCMC running')
    sampler.run_mcmc(state, niter, progress = True)

    print("--- %s seconds ---" % (time.time() - start_time))

    plot_MCMC_results(Q, sampler, num_param, nwalkers, scale)

# def set_GPC_model_dictionary(n, Q, p):
#     model_dict = {}
#     for i in range(n):
#         model_dict[i+1] = GpcSurrogateModel(Q, p[i])
#     return model_dict

def generate_stdrn_simparamset(sigma):
    Q = SimParamSet()
    for i in range(len(sigma)):
        s = SimParameter('pn_' + str(i+1), NormalDistribution(0, sigma[i]))
        Q.add(s)
    return Q

def set_prior_paramset(name, a, b):
    Q = SimParamSet()
    assert(len(name)==len(a)==len(b))
    for i in range(len(name)):
        Q.add(SimParameter(name[i], UniformDistribution(a[i], b[i])))
    return Q
