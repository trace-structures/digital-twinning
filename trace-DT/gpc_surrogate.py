import numpy as np
import pandas as pd
import torch
from multiindex import multiindex
from gpc_functions import syschar_to_polysys
from sklearn.metrics import mean_squared_error
from scipy.sparse import csr_matrix
import copy
from object_utils import gpc_multiindex2param_names
import shap


# ##########################################################################################
#                           GPC SURROGATE MODEL
#                ‘Generalized Polynomial Chaos Expansion’
# ##########################################################################################
class GpcSurrogateModel:
    def __init__(self, Q, p=0, I="default", full_tensor=False, **kwargs):
        self.basis = GpcBasis(Q, p=p, I="default", full_tensor=False)
        self.Q = Q
        self.u_i_alpha = []

    def __repr__(self):
        attrs = vars(self)
        return ', '.join("%s: %s" % item for item in attrs.items())

    def compute_coeffs_by_regression(self, q_j_k, u_i_k):
        q_j_k = torch.t(q_j_k)
        u_i_k = torch.t(u_i_k)
        xi_j_k = self.Q.params2germ(q_j_k)
        phi_alpha_k = self.basis.evaluate(xi_j_k)
        u_i_alpha = np.matmul(u_i_k, np.linalg.pinv(phi_alpha_k))
        self.u_i_alpha = u_i_alpha

    def compute_coeffs_by_projection(self, q_j_k, u_i_k, w_k):
        xi_j_k = self.Q.params2germ(q_j_k)
        phi_alpha_k = self.basis.evaluate(xi_j_k)
        u_i_alpha = np.matmul(u_i_k, np.diag(w_k), phi_alpha_k.transpose())
        self.u_i_alpha = u_i_alpha

    def predict_response(self, q_j_k):
        if not isinstance(q_j_k, torch.Tensor):
            q_j_k = torch.tensor(q_j_k, dtype=torch.float64)           
        q_j_k = torch.t(q_j_k)
        xi_j_k = self.Q.params2germ(q_j_k)
        phi_alpha_k = self.basis.evaluate(xi_j_k)
        u_i_j = np.matmul(self.u_i_alpha, phi_alpha_k)
        if u_i_j.ndim == 1:  # Check if the array is 1D
            u_i_j = u_i_j.reshape(-1, 1)  # Reshape to (x, 1)
        u_i_j = np.array(u_i_j)
        return u_i_j
    
    def predict(self, q_j_k):
        if not isinstance(q_j_k, torch.Tensor):
            q_j_k = torch.tensor(q_j_k.values, dtype=torch.float64)           
        q_j_k = torch.t(q_j_k)
        xi_j_k = self.Q.params2germ(q_j_k)
        phi_alpha_k = self.basis.evaluate(xi_j_k)
        u_i_j = np.matmul(self.u_i_alpha, phi_alpha_k)
        u_i_j = torch.t(u_i_j)
        if u_i_j.ndim == 1:  # Check if the array is 1D
            u_i_j = u_i_j.reshape(-1, 1)  # Reshape to (x, 1)
        u_i_j = np.array(u_i_j)
        return u_i_j
    
    def train(self, q_train, y_train):
        self.compute_coeffs_by_regression(q_train, y_train)
        y = self.predict_response(q_train)
        mse = mean_squared_error(y_train, y)
        print(f"Validation MSE: {mse:.4f}")
        return 
    
    def train_and_evaluate(self, q_train, y_train, q_val, y_val):
        self.compute_coeffs_by_regression(q_train, y_train)
        y_tr_pred = (self.predict_response(q_train)).transpose()
        y_vl_pred = (self.predict_response(q_val)).transpose()
        mse_tr = mean_squared_error(y_train, y_tr_pred)
        mse_vl = mean_squared_error(y_val, y_vl_pred)
        return mse_tr, mse_vl
    
    def mean(self):
        coeffs = self.u_i_alpha
        means = coeffs[:, 0]
        return means
    
    def variance(self):
        coeffs = self.u_i_alpha
        variances = torch.sum(coeffs[:, 1:]**2, axis=1)
        return variances
    
    def compute_partial_vars_original(self, model_obj, max_index=1):
        a_i_alpha = model_obj.model.u_i_alpha
        V_a_orig = model_obj.model.basis
        QoI_names = model_obj.QoI_names
        # Get the multiindex set and convert to a logical array
        V_a = copy.deepcopy(V_a_orig)
        I_a = V_a.I
        I_s = I_a.astype(bool)
        
        # Calculate total variance
        sqr_norm = V_a.norm(do_sqrt=False)
        var_row = np.multiply(a_i_alpha**2, sqr_norm.transpose())
        total_var = torch.sum(var_row, axis=1)[:, None]

        # Get rid of indices with too high an order and the mean
        sobol_order = np.sum(I_s, axis=1)
        ind = (sobol_order > 0) & (sobol_order <= max_index)

        I_a = I_a[ind, :]
        I_s = I_s[ind, :]
        a_i_alpha = a_i_alpha[:, ind]
        V_a.I = I_a

        # Calculate variance of the a_i_alpha * F_alpha polynomials
        sqr_norm = V_a.norm(do_sqrt=False)
        var_row = np.multiply(a_i_alpha**2, sqr_norm.transpose())

        # Get unique rows from Sobol indices
        I_s, _, ind2 = np.unique(I_s, axis=0, return_index=True, return_inverse=True)
        M = V_a.size()[0]
        U = csr_matrix((np.ones(M), (np.arange(M), ind2)), shape=(M, len(I_s)))
        
        # Compute the partial variance
        partial_var = var_row @ U.toarray()

        # Sort partial variances by Sobol order
        order_criterion = np.column_stack([np.sum(I_s, axis=1), np.flip(I_s, axis=1)])
        sortind = np.argsort(order_criterion.view([('', order_criterion.dtype)] * order_criterion.shape[1]), axis=0).flatten()
        I_s = I_s[sortind]
        I_s = I_s.astype(int)
        partial_var = partial_var[:, sortind]
        
        indexed_param_names = gpc_multiindex2param_names(I_s, self.Q.param_names())
        # Compute the ratios (optional)
        sobol_index = np.multiply(partial_var, 1. / total_var)
        
        ###
        y_pred = model_obj.predict(model_obj.X_train)
        y_var = np.var(y_pred, axis=0).reshape(-1, 1)
        #partial_var = np.multiply(sobol_index, y_var)
        ###
        
        
        partial_var_df = pd.DataFrame(partial_var, columns=indexed_param_names, index=QoI_names)
        sobol_index_df = pd.DataFrame(sobol_index, columns=indexed_param_names, index=QoI_names)
                
        return partial_var_df, sobol_index_df, y_var
    
    def compute_partial_vars(self, model_obj, max_index=1):
        a_i_alpha = model_obj.model.u_i_alpha
        V_a_orig = model_obj.model.basis
        QoI_names = model_obj.QoI_names
        # Get the multiindex set and convert to a logical array
        V_a = copy.deepcopy(V_a_orig)
        I_a = V_a.I
        I_s = I_a.astype(bool)
        
        # Identify the alpha=0 term (mean) to exclude from total variance
        is_alpha0 = np.all(I_a == 0, axis=1)
        
        # Calculate total variance excluding alpha=0
        a_i_alpha_nonzero = a_i_alpha[:, ~is_alpha0]
        sqr_norm_nonzero = V_a_orig.norm(do_sqrt=False)[~is_alpha0]
        var_row_nonzero = np.multiply(a_i_alpha_nonzero**2, sqr_norm_nonzero.transpose())
        total_var = torch.sum(var_row_nonzero, axis=1)[:, np.newaxis]

        # Get rid of indices with too high an order and the mean
        sobol_order = np.sum(I_s, axis=1)
        ind = (sobol_order > 0) & (sobol_order <= max_index)

        I_a = I_a[ind, :]
        I_s = I_s[ind, :]
        a_i_alpha = a_i_alpha[:, ind]
        V_a.I = I_a

        # Calculate variance of the a_i_alpha * F_alpha polynomials
        sqr_norm = V_a.norm(do_sqrt=False)
        var_row = np.multiply(a_i_alpha**2, sqr_norm.transpose())

        # Get unique rows from Sobol indices
        I_s, _, ind2 = np.unique(I_s, axis=0, return_index=True, return_inverse=True)
        M = V_a.size()[0]
        U = csr_matrix((np.ones(M), (np.arange(M), ind2)), shape=(M, len(I_s)))
        
        # Compute the partial variance
        partial_var = var_row @ U.toarray()

        # Sort partial variances by Sobol order
        order_criterion = np.column_stack([np.sum(I_s, axis=1), np.flip(I_s, axis=1)])
        sortind = np.lexsort([order_criterion[:, i] for i in reversed(range(order_criterion.shape[1]))])
        I_s = I_s[sortind]
        partial_var = partial_var[:, sortind]
        
        indexed_param_names = gpc_multiindex2param_names(I_s, self.Q.param_names())
        
        # Compute the Sobol indices
        sobol_index = np.divide(partial_var, total_var)
        
        partial_var_df = pd.DataFrame(partial_var, columns=indexed_param_names, index=QoI_names)
        sobol_index_df = pd.DataFrame(sobol_index, columns=indexed_param_names, index=QoI_names)
                
        return partial_var_df, sobol_index_df, total_var
    
    def get_shap_values(self, predict_fn, q, forced=False, explainer_type="kernelexplainer"):
        if explainer_type == "kernelexplainer":
            if hasattr(self, 'explainer') == False or forced == True:
                explainer = shap.KernelExplainer(predict_fn, q)
                self.explainer = explainer
            shap_values = self.explainer(q)
        return shap_values


# ##########################################################################################
#                           GPC BASIS
# ##########################################################################################
class GpcBasis:
    # ---------------------Initialization---------------------------------------------------
    def __init__(self, Q, p=0, I="default", full_tensor=False, **kwargs):
        m = Q.num_params()
        self.m = m

        self.syschars = Q.get_gpc_syschars()
        self.p = p

        if I == "default":
            self.I = multiindex(self.m, p, full_tensor=full_tensor)
        else:
            self.I = I

    # ---------------------set how gpc looks when printed ---------------------------------------------------
    def __repr__(self):
        attrs = vars(self)
        return ', '.join("%s: %s" % item for item in attrs.items())

    # ----------------------------------------- size --------------------------------------------------
    def size(self):
        return self.I.shape

    # ----------------------------Evaluate basis functions ---------------------------------------------------
    def evaluate(self, xi, dual=False):
        syschars = self.syschars
        I = self.I
        m = self.m
        M = self.I.shape[0]
        if xi.ndim == 1:
            xi = xi.reshape(-1, 1)
        k = xi.shape[1]
        deg = max(self.I.flatten())

        p = np.zeros([m, k, deg + 2])
        p[:, :, 0] = np.zeros(xi.shape)
        p[:, :, 1] = np.ones(xi.shape)

        if len(syschars) == 1:
            polysys = syschar_to_polysys(syschars)
            r = polysys.recur_coeff(deg)
            for d in range(deg):
                p[:, :, d + 2] = (r[d, 0] + xi * r[d, 1]) * p[:, :, d + 1] - r[d, 2] * p[:, :, d]
        else:
            for j, syschar in enumerate(syschars):
                polysys = syschar_to_polysys(syschar)
                r = polysys.recur_coeff(deg)
                for d in range(deg):
                    p[j, :, d + 2] = (r[d, 0] + xi[j, :] * r[d, 1]) * p[j, :, d + 1] - r[d, 2] * p[j, :, d]

        y_alpha_j = np.ones([M, k])
        for j in range(m):
            y_alpha_j = y_alpha_j * p[j, :, I[:, j] + 1]

        if dual:
            nrm2 = self.norm(do_sqrt=False)
            y_alpha_j = (y_alpha_j / nrm2.reshape(-1, 1)).transpose()
        return y_alpha_j

    # ------------------------Compute the norm of the basis functions-----------------------
    def norm(self, do_sqrt=True):
        syschars = self.syschars
        I = self.I
        m = self.m
        M = self.I.shape[0]

        if syschars == syschars.lower():
            norm_I = np.ones([M, 1])
            return norm_I

        if len(syschars) == 1:
            # max degree of univariate polynomials
            deg = max(self.I.flatten())
            polysys = syschar_to_polysys(syschars)
            nrm = polysys.sqnorm(range(deg + 1))
            norm2_I = np.prod(nrm[I].reshape(I.shape), axis=1)

        else:
            norm2_I = np.ones([M])
            for j in range(m):
                deg = max(I[:, j])
                polysys = syschar_to_polysys(syschars[j])
                nrm2 = polysys.sqnorm(np.arange(deg + 1))
                norm2_I = norm2_I * nrm2[I[:, j]]
        if do_sqrt:
            norm_I = np.sqrt(norm2_I)
        else:
            norm_I = norm2_I

        return norm_I


# ##########################################################################################
#                           UTILS
# ##########################################################################################

#
# ##########################################################################################
#                           TEST
# ##########################################################################################
from simparameter import SimParameter
from simparameter_set import SimParamSet
from distributions import UniformDistribution, NormalDistribution
def main():
    #print(multiindex(3, 4))
    P1 = SimParameter('p1', UniformDistribution(-2,2))
    P2 = SimParameter('p2', NormalDistribution(-2,2))

    Q = SimParamSet()
    Q.add(P1)
    Q.add(P2)
    gPCE = GpcSurrogateModel(Q, p=3)
    gPCE.basis.norm()
    print(gPCE.basis.evaluate(np.array([np.arange(-1, 1, 0.1)] * 2)))
    #print(gPCE)


if __name__ == "__main__":
    main()
