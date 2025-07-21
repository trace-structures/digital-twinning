import asyncio
from gpc_surrogate import GpcSurrogateModel
from gbt_surrogate import GenGBT
from dnn_surrogate import DNNModel
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import FunctionTransformer
import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from digital_twin import DigitalTwin
import utils
import os
import pickle
from sklearn.metrics import mean_squared_error
from linreg_surrogate import LinRegSurrogate
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from scipy.stats import kendalltau, pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error

class SurrogateModel:
    def __init__(self, Q, QoI_names, method, **kwargs):
        self.Q = Q # Parameter Set
        self.QoI_names = QoI_names
        # self.params2germ = lambda q: Q.params2germ(q.transpose()).transpose()
        # self.germ2params = lambda x: Q.germ2params(x.transpose()).transpose()
        self.method = method
        self.init_config = kwargs
        self.model = self.get_model()
        
    # # functions for getting model    
    def get_model(self):
        match self.method:
            case "DNN":
                self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                print(f"Using device: {self.device}") # TODO use GPU in dnn_surrogate
                DNN = DNNModel(self.Q.num_params(), len(self.QoI_names), **self.init_config)
                DNN = DNN.double() # modify all datatype of model's parameters to torch.float64
                return DNN
            case "gPCE":
                gPCE = GpcSurrogateModel(self.Q, **self.init_config)
                return gPCE
            case "GBT":
                GBT = GenGBT(self.Q, self.QoI_names, **self.init_config)
                return GBT
            case "LinReg":
                LinReg = LinRegSurrogate(self.Q, self.QoI_names)
                return LinReg
            case _:
                raise ValueError(f"There is no method type: {self.method}")
    
    # splitting train and test dataset
    def train_test_split(self, X_data, y_data, train_test_ratio=0.2, random_seed=1997, split_type='shuffle'):
        self.samples = X_data
        self.QoI = y_data        
        if split_type == 'shuffle':
            X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=train_test_ratio, random_state=random_seed, shuffle=True)
        elif split_type == 'no_shuffle':
            X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=train_test_ratio, random_state=random_seed, shuffle=False)
        elif split_type == 'reverse':
            X_test, X_train, y_test, y_train = train_test_split(X_data, y_data, train_size=train_test_ratio, random_state=random_seed, shuffle=False)
        
        self.X_train, self.X_test, self.y_train, self.y_test = X_train, X_test, y_train, y_test        
        return X_train, X_test, y_train, y_test

    # function for training model
    def train(self, q_train, y_train, q_scaler=None, y_scaler=None, k_fold=None, **params):
        self.train_config = params
        self.get_q_scaler(q_train, q_scaler)
        self.get_y_scaler(y_train, y_scaler)
        xi = self.get_scaled_q(q_train)
        yt = self.get_scaled_y(y_train) # y transformed
        print(f'----- Training started for \'{self.method}\' model -----')
        
        if k_fold is not None:
            train_losses, val_losses = self.cross_validation(xi, yt, k_fold, **params)
        else: # if there is no crossvalidation, then split 80% data for training and 20% for validation
            xi_train, xi_val, yt_train, yt_val = train_test_split(xi, yt, test_size=0.2, random_state=12)
            train_losses, val_losses = self.train_and_validate(xi_train, yt_train, xi_val, yt_val, **params)
        print(f"Average train loss: {np.mean(train_losses):.14f}, Average valid loss: {np.mean(val_losses):.14f}")      
        print(f'----- Training ended for \'{self.method}\' model -----')
        
    def cross_validation(self, xi, yt, k_fold, **params):
        train_losses, val_losses = [], []
        kf = KFold(n_splits=k_fold, shuffle=True)
        for fold, (train_indices, val_indices) in enumerate(kf.split(xi)):
            print(f"Fold {fold + 1}/{k_fold}")
            xi_train, xi_val = xi[train_indices], xi[val_indices]
            yt_train, yt_val = yt[train_indices], yt[val_indices]
            tr_loss, vl_loss = self.train_and_validate(xi_train, yt_train, xi_val, yt_val, **params)
            train_losses.append(tr_loss), val_losses.append(vl_loss)
        return train_losses, val_losses
    
    def train_and_validate(self, xi_train, yt_train, xi_val, yt_val, **params):
        match self.method:
            case "DNN":
                xi_tr,xi_vl = torch.tensor(xi_train, dtype=torch.float64), torch.tensor(xi_val, dtype=torch.float64)
                yt_tr,yt_vl = torch.tensor(yt_train, dtype=torch.float64), torch.tensor(yt_val, dtype=torch.float64)
                tr_loss, vl_loss = self.model.train_and_validate(xi_tr, yt_tr, xi_vl, yt_vl, **params)
            case "gPCE":
                xi_tr, xi_vl = torch.tensor(xi_train, dtype=torch.float64), torch.tensor(xi_val, dtype=torch.float64)
                yt_tr, yt_vl = torch.tensor(yt_train, dtype=torch.float64), torch.tensor(yt_val, dtype=torch.float64)
                tr_loss, vl_loss = self.model.train_and_evaluate(xi_tr, yt_tr, xi_vl, yt_vl, **params)
            case "GBT":
                xi_tr, xi_vl = pd.DataFrame(xi_train, columns=self.Q.param_names()), pd.DataFrame(xi_val, columns=self.Q.param_names())
                yt_tr, yt_vl = pd.DataFrame(yt_train, columns=self.QoI_names), pd.DataFrame(yt_val, columns=self.QoI_names)                 
                tr_loss, vl_loss = self.model.train_and_validate(xi_tr, yt_tr, xi_vl, yt_vl, **params)
            case "LinReg":
                xi_tr, xi_vl = pd.DataFrame(xi_train, columns=self.Q.param_names()), pd.DataFrame(xi_val, columns=self.Q.param_names())
                yt_tr, yt_vl = pd.DataFrame(yt_train, columns=self.QoI_names), pd.DataFrame(yt_val, columns=self.QoI_names)
                tr_loss, vl_loss = self.model.train_and_evaluate(xi_tr, yt_tr, xi_vl, yt_vl)
            case _:
                raise ValueError(f"There is no method type: {self.method}")
        return tr_loss, vl_loss      
    
    def save_model(self, name, path=None):
        name = name + ".sm" # ?? or name + ".ddm" ???
        if path is not None:
            name = os.path.join(path, name)
        with open(name, 'wb') as file:
            pickle.dump(self, file)
        print(f"Model saved to {name}")

    ## function for predicting   
    def predict(self, q, **params):
        if type(q) != pd.DataFrame:
            q = pd.DataFrame(q, columns=self.Q.param_names())
        xi = self.get_scaled_q(q)
        match self.method:
            case "DNN":
                xi = torch.tensor(xi, dtype=torch.float64)
                #y_t = self.model.predict(xi, **params)
            case "gPCE":
                xi = torch.tensor(xi, dtype=torch.float64)
                #y_t = self.model.predict(xi, **params)
            case "GBT":
                xi = pd.DataFrame(xi, columns=self.Q.param_names())
                #y_t = self.model.predict(xi, **params)
            case "LinReg":
                xi = pd.DataFrame(xi, columns=self.Q.param_names())
            case _:
                raise ValueError(f"There is no method type: {self.method}")
        y_t = self.model.predict(xi, **params)
        y = self.get_orig_y(y_t)
        return y
    
    def get_mean_and_var(self, n_sample=1000):
        if self.method == "DNN" or self.method == "GBT" or self.method == "LinReg":
            q = self.Q.sample(n_sample)
            q_df = pd.DataFrame(q, columns=self.Q.param_names())
            y_predict = self.predict(q_df)
            mean, var = y_predict.mean(axis=0), y_predict.var(axis=0)
        elif self.method == "gPCE":
            mean, var = self.model.mean(), self.model.variance()
        return mean, var

    # functions for scaling q     
    def get_q_scaler(self, q, q_scaler, scale_method='default'):
        if scale_method == 'default':
            match self.method:
                case "DNN":
                    scale_method = 'minmax'
                case "gPCE":
                    scale_method = 'minmax'
                case "GBT":
                    scale_method = 'identity'
                case "LinReg":
                    scale_method = 'identity'
                    
        if scale_method == 'identity':
            self.q_scaler = FunctionTransformer(func=identity_func, inverse_func=identity_inverse_func)
        elif scale_method == 'minmax':
            self.q_scaler = MinMaxScaler((0.05, 0.95))
            self.q_scaler.fit(q)
        return self.q_scaler
    
    def get_scaled_q(self, q):
        xi = self.q_scaler.transform(q)
        return xi
    
    def get_orig_q(self, xi):
        q = self.q_scaler.inverse_transform(xi)
        return q
    
    # functions for scaling y
    def get_y_scaler(self, y, y_scaler, scale_method='default'):
        if scale_method == 'default':
            match self.method:
                case "DNN":
                    scale_method = 'minmax'
                case "gPCE":
                    scale_method = 'minmax'
                case "GBT":
                    scale_method = 'identity'
                case "LinReg":
                    scale_method = 'identity'

        if scale_method == 'identity':
            self.y_scaler = FunctionTransformer(func=identity_func, inverse_func=identity_inverse_func)
        elif scale_method == 'minmax':
            self.y_scaler = MinMaxScaler((0.05, 0.95))
            self.y_scaler.fit(y)
        return self.y_scaler
    
    def get_scaled_y(self, y):
        y_t = self.y_scaler.transform(y)
        return y_t
    
    def get_orig_y(self, y_t):
        y = self.y_scaler.inverse_transform(y_t)
        return y
    
    def evaluate_model(self, verbose=False):
        pred = self.predict(self.X_test)

        # Ensure y_test is a DataFrame for consistent column-wise iteration
        if isinstance(self.y_test, pd.Series):
            y_test_df = self.y_test.to_frame()
        elif isinstance(self.y_test, np.ndarray) and self.y_test.ndim == 1:
            y_test_df = pd.DataFrame(self.y_test)
        else: # Already a DataFrame or multi-dim array
            y_test_df = self.y_test if isinstance(self.y_test, pd.DataFrame) else pd.DataFrame(self.y_test)

        # Ensure y_train is a DataFrame for consistent column-wise iteration
        if isinstance(self.y_train, pd.Series):
            y_train_df = self.y_train.to_frame()
        elif isinstance(self.y_train, np.ndarray) and self.y_train.ndim == 1:
            y_train_df = pd.DataFrame(self.y_train)
        else: # Already a DataFrame or multi-dim array
            y_train_df = self.y_train if isinstance(self.y_train, pd.DataFrame) else pd.DataFrame(self.y_train)

        # Ensure prediction is a DataFrame for consistent column-wise iteration
        if isinstance(pred, pd.Series):
            pred_df = pred.to_frame()
        elif isinstance(pred, np.ndarray) and pred.ndim == 1:
            pred_df = pd.DataFrame(pred)
        else: # Already a DataFrame or multi-dim array
            pred_df = pred if isinstance(pred, pd.DataFrame) else pd.DataFrame(pred)

        num_outputs = y_test_df.shape[1]
        output_column_names = y_test_df.columns.tolist()
        if num_outputs == 1:
            output_column_names[0] = self.QoI_names[0]

        # List to store dictionaries for each output's metrics (for DataFrame creation)
        individual_metrics_list = []

        for i, col_name in enumerate(output_column_names):
            y_true_dim = y_test_df.iloc[:, i]
            y_pred_dim = pred_df.iloc[:, i]
            y_train_dim = y_train_df.iloc[:, i]

            # Check for constant inputs before calculating correlations
            is_y_true_constant = (y_true_dim.nunique() <= 1)
            is_y_pred_constant = (y_pred_dim.nunique() <= 1)

            if is_y_true_constant or is_y_pred_constant:
                # If either input is constant, correlation is undefined
                kendall_tau_val = np.nan
                pearson_val = np.nan
                spearman_val = np.nan
            else:
                try:
                    kendall_tau_val = kendalltau(y_true_dim, y_pred_dim)[0]
                except ValueError:
                    kendall_tau_val = np.nan
                try:
                    pearson_val = pearsonr(y_true_dim, y_pred_dim)[0]
                except ValueError:
                    pearson_val = np.nan
                try:
                    spearman_val = spearmanr(y_true_dim, y_pred_dim)[0]
                except ValueError:
                    spearman_val = np.nan

            mse_val = mean_squared_error(y_true_dim, y_pred_dim)
            mae_val = mean_absolute_error(y_true_dim, y_pred_dim)
            rmse_val = np.sqrt(mse_val)
            norm_rmse = rmse_val / (max(y_train_dim) - min(y_train_dim))
            norm_mae = mae_val / (max(y_train_dim) - min(y_train_dim))

            # Get STD of label for the specific training output column
            if isinstance(self.y_train, pd.Series):
                std_of_label_val = self.y_train.std()
            elif isinstance(self.y_train, np.ndarray) and self.y_train.ndim == 1:
                std_of_label_val = pd.Series(self.y_train).std()
            else: # It's a DataFrame or multi-dim array
                y_train_df_temp = self.y_train if isinstance(self.y_train, pd.DataFrame) else pd.DataFrame(self.y_train)
                std_of_label_val = y_train_df_temp.iloc[:, i].std()

            current_output_metrics = {
                "Kendall_tau": kendall_tau_val,
                "Pearson": pearson_val,
                "Spearman": spearman_val,
                "MSE": mse_val,
                "MAE": mae_val,
                "RMSE": rmse_val,
                "STD_of_label": std_of_label_val,
                "NRMSE": norm_rmse,
                "NMAE": norm_mae
            }
            # Add relative RMSE and summed metric for each output
            # Check for zero STD_of_label before division
            if std_of_label_val == 0 or np.isnan(std_of_label_val):
                current_output_metrics["rel_RMSE"] = np.nan
            else:
                current_output_metrics["rel_RMSE"] = (
                    current_output_metrics["RMSE"] /
                    current_output_metrics["STD_of_label"]
                )

            # Ensure correlation metrics are not NaN for summed metric calculation
            k_tau = 0 if np.isnan(kendall_tau_val) else kendall_tau_val
            p_corr = 0 if np.isnan(pearson_val) else pearson_val
            s_corr = 0 if np.isnan(spearman_val) else spearman_val
            rel_rmse = current_output_metrics["rel_RMSE"] if not np.isnan(current_output_metrics["rel_RMSE"]) else 0

            current_output_metrics["summed_metric"] = (
                k_tau + p_corr + s_corr - rel_rmse
            ) / 3

            # Add the output name/label to this dictionary
            current_output_metrics['Output_Name'] = col_name
            individual_metrics_list.append(current_output_metrics)

        # Create DataFrame for individual output metrics
        df_individual_metrics = pd.DataFrame(individual_metrics_list)
        df_individual_metrics = df_individual_metrics.set_index('Output_Name')

        # --- Conditional Aggregated Metrics ---
        model_eval_aggregated = {}
        if num_outputs > 1: # Only calculate aggregated metrics if there are multiple outputs
            for metric_name in ["Kendall_tau", "Pearson", "Spearman", "MSE", "MAE", "RMSE", "rel_RMSE", 'NRMSE', 'NMAE', "summed_metric"]:
                values = df_individual_metrics[metric_name].dropna().tolist()
                if values:
                    model_eval_aggregated[f"Aggregated_{metric_name}_Mean"] = np.mean(values)
                    model_eval_aggregated[f"Aggregated_{metric_name}_Min"] = np.min(values)
                    model_eval_aggregated[f"Aggregated_{metric_name}_Max"] = np.max(values)
                else:
                    model_eval_aggregated[f"Aggregated_{metric_name}_Mean"] = np.nan
                    model_eval_aggregated[f"Aggregated_{metric_name}_Min"] = np.nan
                    model_eval_aggregated[f"Aggregated_{metric_name}_Max"] = np.nan

        if verbose:
            print("\n--- Individual Output Metrics (Table) ---")
            print(df_individual_metrics.round(3))

            if num_outputs > 1:
                print("\n--- Aggregated Metrics ---")
                for key, value in model_eval_aggregated.items():
                    print(f'{key}: {value:.3f}')
            else:
                print("\n(Aggregated metrics not shown for single output.)")

        return [df_individual_metrics, model_eval_aggregated]
    
    def get_sobol_sensitivity(self, max_index=1): # max_of_max_index=max_number_of_parameters
        if (self.method == "DNN" or self.method == "GBT") and max_index > 2:
            print(f'Warning: The maximum value of max_index for {self.method} is 2. It is automatically set to 2.')
            max_index = 2
        if hasattr(self, 'max_index') == False:
            self.max_index = max_index
        partial_var_df, sobol_index_df, total_var = self.model.compute_partial_vars(self, max_index)
        self.partial_var_df, self.sobol_index_df, self.total_var = partial_var_df, sobol_index_df, total_var
        return partial_var_df, sobol_index_df
            
    async def get_shap_values(self, q, mean=False, sample_size_from_q=100):
        if type(sample_size_from_q) is str and sample_size_from_q == 'all':
            print(f'Message: sample size for shap values is set to {q.shape[0]}, which is the number of samples.')
            sample_size_from_q = q.shape[0]
        elif type(sample_size_from_q) is int:
            if q.shape[0] < sample_size_from_q:
                print(f'Warning: sample_size_from_q is larger than the number of samples. It is automatically set to {q.shape[0]}, which is the number of samples.')
                sample_size_from_q = q.shape[0]
            else:
                print(f'Message: sample size for shap values is set to {sample_size_from_q}.')
            q = q.iloc[:sample_size_from_q]
        shap_values = await asyncio.to_thread(self.model.get_shap_values, self.predict, q)
        self.shap_values = shap_values
        if mean:
            mean_shap_values = np.mean(np.abs(shap_values.values), axis=0)
            mean_shap_values_df = pd.DataFrame(mean_shap_values, index=q.columns, columns=self.QoI_names)
            return mean_shap_values_df
        return shap_values
    
    async def subtract_effects(self, q, QoI, subtracted_params):
        QoI = pd.DataFrame(QoI, columns=self.QoI_names)
        shap_values = await self.get_shap_values(q, mean=False, sample_size_from_q='all')
        columns = q.columns.tolist()
        indices = []
        for x in subtracted_params:
            if x in columns:
                indices.append(columns.index(x))
            else:
                raise ValueError(f"There is no parameter called \'{x}\'")
        result = shap_values.values[:, indices, :].sum(axis=1)
        result_df = pd.DataFrame(result, columns=QoI.columns, index=QoI.index)
        #subtract_shap_values = shap_values[subtracted_params]
        remained_effects = QoI - result_df
        return remained_effects
    
    async def plot_subtract_effects_and_alert(self, q, QoI, subtracted_params, threshold_ratio=0.1):
        alert = False
        remained_train_effects = await self.subtract_effects(self.X_train.iloc[:100], pd.DataFrame(self.y_train).iloc[:100], subtracted_params)
        #remained_train_effects = await self.subtract_effects(self.X_train.iloc[:101], self.y_train.iloc[:101], subtracted_params)
        QoI_names = remained_train_effects.columns
        remained_train_effects = remained_train_effects.values

        higher_bound = np.max(remained_train_effects, axis=0)
        lower_bound = np.min(remained_train_effects, axis=0)
        margin = (higher_bound - lower_bound) * threshold_ratio

        remained_measured_effects = await self.subtract_effects(q, QoI, subtracted_params)
        remained_measured_effects = remained_measured_effects.values

        higher_measured_bound = np.max(remained_measured_effects, axis=0)
        lower_measured_bound = np.min(remained_measured_effects, axis=0)


        for i in range(len(higher_measured_bound)):
            if (higher_measured_bound[i] > higher_bound[i] + margin[i]) or (lower_measured_bound[i] < lower_bound[i] - margin[i]):
                alert = True
                break
        fig = utils.plot_remained_effects(higher_bound, lower_bound, margin, higher_measured_bound, lower_measured_bound, QoI_names)
        return fig, alert

    
    def plot_sobol_sensitivity(self, max_index=1, **kwargs):
        if hasattr(self, 'max_index') == False:
            self.max_index = max_index
        if (self.method == "DNN" or self.method == "GBT") and max_index > 2:
            print(f'Warning: The maximum value of max_index for {self.method} is 2. It is automatically set to 2.')
            max_index = 2
        if self.max_index < max_index:
            self.get_sobol_sensitivity(max_index)
        fig = utils.plot_sobol_sensitivity(self.sobol_index_df, self.y_train, **kwargs)
        return fig
    
    async def plot_shap_single_waterfall(self, **kwargs):
        if hasattr(self, 'shap_values') == False:
            self.shap_values = await self.get_shap_values(self.X_test)
            print('The SHAP values have been computed automatically!')
        fig = utils.plot_shap_single_waterfall(self, **kwargs)
        return fig
    
    async def plot_shap_multiple_waterfalls(self, **kwargs):
        if hasattr(self, 'shap_values') == False:
            self.shap_values = await self.get_shap_values(self.X_test)
            print('The SHAP values have been computed automatically!')
        fig = utils.plot_shap_multiple_waterfalls(self, **kwargs)
        return fig
    
    async def plot_shap_beeswarm(self, **kwargs):
        if hasattr(self, 'shap_values') == False:
            self.shap_values = await self.get_shap_values(self.X_test)
            print('The SHAP values have been computed automatically!')
        fig = utils.plot_shap_beeswarm(self, **kwargs)
        return fig
       
    async def plot_effects(self, effects):
        fig = utils.plot_effects(effects)
        return fig
    
    def create_mx_from_tuple(self, pairs, measured_names):
        """
        Create binary matrix H from tuple of QoI names and measured variable names

        Parameters:
        - pairs: List of tuples containing pairs of names.
        - measured_names: Header names of measured data.
        
        Returns:
        - A binary matrix where rows correspond to QoI_names and columns to measured_names.
        """
        if len(measured_names) < len(self.QoI_names):
            raise ValueError("Measured data must have at least as many columns as QoI_names.")

        # Create index mappings for rows and columns
        row_idx = {name: i for i, name in enumerate(self.QoI_names)}
        col_idx = {name: j for j, name in enumerate(measured_names)}
        
        H = np.zeros((len(self.QoI_names), len(measured_names)), dtype=int)
        for a, b in pairs:
            H[row_idx[a], col_idx[b]] = 1
        
        return H
            
# class y_scaler(scaler_type='minmax'):
#     def __init__(self, y_train):
#         self.scaler(y_train)
        
#     def scaler(self, y_train):
#         y_min = y_train.min().min()
#         y_max = y_train.max().max()
#         self.y_min = y_min
#         self.y_max = y_max
        
#     def transform(self, y):
#         y_scaled = (y - self.y_min)*0.95 / (self.y_max - self.y_min)
#         return y_scaled
    
#     def inverse_transform(self, y_scaled):
#         y_unscaled = y_scaled * (self.y_max - self.y_min) / 0.95 + self.y_min
#         return y_unscaled

def identity_func(x):
    if isinstance(x, np.ndarray):
        return x
    return x.to_numpy()

def identity_inverse_func(x):
    return x


########################################################################################################################
#                                                         TEST                                                         #
########################################################################################################################

from simparameter import SimParameter
from simparameter_set import SimParamSet
from distributions import UniformDistribution, NormalDistribution, TranslatedDistribution, ExponentialDistribution

async def main():
    #############################################################################################
    generated_data = False
    method = "DNN"
    
    sobol = False
    shap = True
    calc_effects = False
    update = False
    
    #############################################################################################
    if generated_data:
        # Create SimParameters and SimParameterSet
        dist_orig = NormalDistribution(2, 3)
        dist_orig.fix_bounds(1, 2)
        trans_dist = TranslatedDistribution(dist_orig, dist_orig.shift, dist_orig.scale)
        #P1 = SimParameter('a', trans_dist)
        P1 = SimParameter('a', NormalDistribution(1,0.01))
        P2 = SimParameter('b', UniformDistribution(0.9,1.1))
        P3 = SimParameter('c', NormalDistribution(0,0.01))
        
        # P1 = SimParameter('a', NormalDistribution(0,1))
        # P2 = SimParameter('b', UniformDistribution(-3,3))
        # P3 = SimParameter('c', NormalDistribution(1,1))
        
        Q = SimParamSet()
        Q.add(P1)
        Q.add(P2)
        Q.add(P3)
        
        # Sampling
        x_sample = Q.sample(1000, method='MC', random_seed=1997)
        x_sample = torch.from_numpy(x_sample)
        x_df = pd.DataFrame(x_sample, columns=Q.param_names())
        #x_df.to_csv('../demo/data/sine_data/example_x_df.csv', index=False)

        # Create/calculate target values
        def generate_y_values(x_sample, t):
            y_values = torch.stack([a * np.sin((b*t)) + c for a, b, c in zip(x_sample[:, 0], x_sample[:, 1], x_sample[:, 2])])
            return y_values
        time_steps = 10
        t = torch.linspace(0, np.pi, time_steps, dtype=torch.float64)
        y_values = generate_y_values(x_sample, t)
        QoI_names = [f"t_{i}" for i in range(time_steps)]
        
        y_df = pd.DataFrame(y_values, columns=QoI_names) 
        y_df.to_csv('demo/data/sine_data/example_y_df.csv', index=False)
        
        q = Q.sample(1, method='MC', random_seed=1997)
        q_df = pd.DataFrame(q, columns=Q.param_names())
        q_df.to_csv('demo/data/sine_data/example_q_df.csv', index=False)
    
        QoI_param = 't_2'
        
        # For Update
        z_m = generate_y_values(q, t)
        z_m_df = pd.DataFrame(z_m, columns=QoI_names)

        z_m_df.to_csv('demo/data/sine_data/example_z_m_df.csv', index=False)
        s = 0.2
        sigma = (np.ones(time_steps)*s)
        sigma = y_df.describe().loc['std']/3
        sigma.to_csv('demo/data/sine_data/example_sigma_df.csv', index=False)
        
        subtracted_effects = ['a','b','c']
        
    else:
        import os
        #os.chdir('libraries/')
        cwd = os.getcwd()
        files = os.listdir(cwd)
        print("Files in %r: %s" % (cwd, files))
        
        x_df = pd.read_csv('demo/data/YB_data/x_df.csv')
        y_df = pd.read_csv('demo/data/YB_data/y_df.csv')
        
        Q = utils.get_simparamset_from_data(x_df)
        QoI_names = y_df.columns.to_list()
        
        QoI_param = 'f.4'
        
        #q = np.load('libraries/example_q.npy')
        q = x_df.iloc[0].to_numpy().reshape(1,-1)
        q_df = pd.DataFrame(q, columns=Q.param_names())
        
        # For Update
        z_m_df = pd.read_csv('demo/data/YB_data/z_m_df.csv')
        sigma = pd.read_csv('demo/data/YB_data/sigma.csv').T
        
        subtracted_effects = ['e1','e2','e3','g1','g2','q']
    #############################################################################################    
        
    split_config = {
            'train_test_ratio': 0.2, 
            'random_seed': 1997,
            'split_type': 'no_shuffle'
            }
    
    max_index = 1
    
    # For Update
    E = utils.generate_stdrn_simparamset(sigma.values.flatten())
    nwalkers = 64
    nburn = 500
    niter = 100
    #############################################################################################
    # Model configurations
    match method:
        # DNN model configurations
        case "DNN":
            config = {  
                'init_config' : {
                    'layers': [
                        {'neurons': 512, 'activation': 'relu', 'dropout': 0.2},
                        {'neurons': 256, 'activation': 'sigmoid', 'dropout': 0.2},
                        {'neurons': 256, 'activation': 'sigmoid', 'dropout': 0.2},
                        {'neurons': 128, 'activation': 'relu', 'dropout': None},
                        ],
                    'outputAF': 'tanh'
                    },
                'train_config' : {
                    'optimizer': 'Adam',
                    'loss': 'MSE',
                    'epochs': 100,
                    'batch_size': 32,
                    'k_fold': None,
                    'early_stopping': {
                        'patience': 25,
                        'min_delta': 0.0001}
                    }
                    }
        # gPCE model configurations
        case "gPCE":
            config = {
                'init_config' : {
                'p' : 4
                }, 
                'train_config' : {
                    'k_fold': 9
                    }
            }
        # GBT model configurations
        case "GBT":
            config = {
                'init_config' : {
                    'gbt_method': 'xgboost'
                },
                'train_config' : {
                    'max_depth': 6,
                    'num_of_iter': 250,
                    'k_fold': 9
                    }
            }

    # Surrogate Model
    model = SurrogateModel(Q, QoI_names, method, **config['init_config'])  
    model.train_test_split(x_df, y_df, **split_config)
    model.train(model.X_train, model.y_train, **config['train_config'])
    mean, var = model.get_mean_and_var()
    if sobol:
        partial_variance, sobol_index = model.get_sobol_sensitivity(max_index) #np.inf
        model.plot_sobol_sensitivity(max_index=max_index)
        model.plot_sobol_sensitivity(max_index=max_index, param_name='S.1')
        model.plot_sobol_sensitivity(max_index=max_index, param_name='freq')
        model.plot_sobol_sensitivity(max_index=max_index, param_name=QoI_param)
    if shap:
        shap = await model.get_shap_values(model.X_test)
        fig = await model.plot_shap_single_waterfall(q=q_df, param_name=QoI_param)
        fig = await model.plot_shap_multiple_waterfalls(q=q_df)
        fig = await model.plot_shap_multiple_waterfalls(q=q_df, param_name='S.1')
        fig = await model.plot_shap_multiple_waterfalls(q=q_df, param_name='freq')
        fig = await model.plot_shap_beeswarm(param_name=QoI_param)
    if calc_effects:
        effects = await model.subtract_effects(model.X_test.iloc[:100], model.y_test.iloc[:100], subtracted_effects) #q_df.columns.to_list()
        #fig = await model.plot_effects(effects)
        fig, alert = await model.plot_subtract_effects_and_alert(model.X_test.iloc[:1], model.y_test.iloc[:1], subtracted_effects, threshold_ratio=0.1)
    model.predict(q_df)
    
    # # Update
    if update:
        DT = DigitalTwin(model, E)
        DT.update(z_m_df, nwalkers=nwalkers, nburn=nburn, niter=niter)
        DT.get_mean_and_var_of_posterior()
        DT.get_MAP()
        map = DT.get_MAP()
        utils.plot_MCMC(model, DT, nwalkers=nwalkers, map_point=map)
        utils.plot_grid_update_points_and_surfaces(model.Q, DT.sampler, model, z_m_df, q_df)  
        
    
    return y_df, model
       
if __name__ == "__main__":
    
    import nest_asyncio

    nest_asyncio.apply()

    asyncio.run(main())  
