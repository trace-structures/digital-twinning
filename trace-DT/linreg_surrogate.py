from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
import shap
from SALib.analyze import sobol
from sklearn.linear_model import LinearRegression
from scipy.stats import kendalltau, pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error

class LinRegSurrogate:
    def __init__(self, Q, QoI_names):
        self.model = LinearRegression()
        self.Q = Q # Parameter Set
        self.QoI_names = QoI_names

    def train_and_evaluate(self, X_train, y_train, X_val, y_val):
        self.model.fit(X_train, y_train)
        y_pred_tr = self.model.predict(X_train)
        tr_loss = mean_squared_error(y_train, y_pred_tr)
        y_pred_vl = self.model.predict(X_val)
        vl_loss = mean_squared_error(y_val, y_pred_vl)
        return tr_loss, vl_loss

    def predict(self, X):
        return self.model.predict(X)

    def score(self, X, y):
        return mean_squared_error(y, self.predict(X))
    
    def evaluate_model(self, y_train, X_test, y_test, verbose=False):
        pred = self.model.predict(X_test)
        model_eval = {
            "Kendall_tau": kendalltau(y_test, pred)[0],
            "Pearson":  pearsonr(y_test.squeeze(), pred.squeeze())[0],
            "Spearman": spearmanr(y_test, pred)[0],
            "MSE":  mean_squared_error(y_test, pred),
            "MAE":  mean_absolute_error(y_test, pred),
            "RMSE": np.sqrt(mean_squared_error(y_test, pred)),
            "STD_of_label": y_train.std()
            }
        if verbose==True:
            print('Kendall tau correlation - measure of correspondance between two rankings: %.3f' %model_eval["Kendall_tau"])
            print('Pearson correlation - measure of linear realationship (cov normalised): %.3f' %model_eval["Pearson"])
            print('Spearman correaltion - cov(rank(y1), rank(y2)/stdv(rank(y1))): %.3f' %model_eval["Spearman"])
            print("mean squared error:", model_eval["RMSE"])
            print("STD of label:", model_eval["STD_of_label"])
        df = pd.DataFrame(model_eval, index=[0])
        df['label'] = y_train.name if hasattr(y_train, 'name') else 'label'
        df['rel_RMSE'] = df['RMSE'] / df['STD_of_label']
        df['summed_metric'] = (df['Kendall_tau'] + df['Pearson'] + df['Spearman'] - df['rel_RMSE']) / 3
        
        return [df, model_eval]

    def compute_partial_vars(self, model_obj, max_index):
        paramset = model_obj.Q
        QoI_names = model_obj.QoI_names
        # problem = {
        #     'num_vars': paramset.num_params(),
        #     'names': paramset.param_names(),
        #     'bounds': [[0, 1] for _ in range(paramset.num_params())]
        # }
        problem = {
            'num_vars': paramset.num_params(), 'names': paramset.param_names(), 'dists': paramset.dist_types, 'bounds': paramset.dist_params
            } 
        
        d = paramset.num_params()
        q = paramset.sample(method='Sobol_saltelli', n=8192) # saltelli working only for uniform distribution # N * (2D + 2)
        # https://salib.readthedocs.io/en/latest/user_guide/advanced.html
        #q = paramset.sample(method='QMC_LHS', n=10000)
        #xi = model_obj.get_scaled_q(q)      
        #y = model_obj.model.predict(q)
        y = model_obj.predict(q)
        #y = model_obj.get_orig_y(y_t)
        
        # Run model
        S1 = []
        S2 = []
        for i in range(y.shape[1]):
            y_i = y[:,i]

            # Sobol analysis
            Si_i = sobol.analyze(problem, y_i)
            T_Si, first_Si, (idx, second_Si) = sobol.Si_to_pandas_dict(Si_i)
            df = Si_i.to_df()
            cols_S1 = list(df[1].index)
            cols_S2 = list(df[2].index)

            S1.append(first_Si['S1'])
            S2.append(second_Si['S2'])

        S1 = np.array(S1)
        S2 = np.array(S2)

        col_names = cols_S1
        sobol_index = S1
        if max_index == 2:
            sobol_index = np.concatenate([S1, S2], axis=1)
            col_names = cols_S1 + cols_S2
            col_names = [f"{x[0]} {x[1]}" if isinstance(x, tuple) else x for x in col_names]
                    
        # Compute partial variances
        y_var = y.var(axis=0).reshape(-1, 1)
        partial_variance = sobol_index * y_var
             
        partial_var_df, sobol_index_df = pd.DataFrame(partial_variance, columns=col_names, index=QoI_names), pd.DataFrame(sobol_index, columns=col_names, index=QoI_names)

        return partial_var_df, sobol_index_df, y_var
    
    def get_shap_values(self, predict_fn, X):

        explainer = shap.KernelExplainer(predict_fn, X)
        self.explainer = explainer
        shap_values = self.explainer(X)

        return shap_values

    def to_jsonld(self, model_id: str):
        """
        :param model_id: Unique ID of the model, e.g., 'LinReg001'
        """

        jsonld = {
            "@context": {
                "mls": "https://ml-schema.github.io/documentation/mls.html",
                "rdfs": "http://www.w3.org/2000/01/rdf-schema#"
            },

            "@id": f"https://example.org/models/{model_id}",
            "@type": "mls:Model",
            "mls:implementsAlgorithm": {
                "@id": "https://en.wikipedia.org/wiki/Linear_regression",
                "@type": "mls:Algorithm",
                "rdfs:label": "Linear Regression"
            },

            "mls:hasHyperParameter": [],

            "mls:hasInput": [
                {
                    "@type": "mls:Feature",
                    "mls:featureName": name,
                    "mls:hasDistribution": {
                        "@type": "mls:Distribution",
                        "mls:distributionType": dist.get_type(),
                        "mls:params": str(dist.dist_params),
                    }
                }
                for (name, dist) in self.Q.params.items()
            ]
        }
    
        return jsonld