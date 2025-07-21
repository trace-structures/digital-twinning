import numpy as np
from simparameter_set import SimParamSet
from simparameter import SimParameter
from distributions import UniformDistribution
from gpc_surrogate import GpcSurrogateModel
from scipy.io import loadmat
import matplotlib.pyplot as plt

def set_prior_paramset():
    Q = SimParamSet()
    Q.add(SimParameter('beta', UniformDistribution(-10, 0)))
    Q.add(SimParameter('beta_T', UniformDistribution(-1, 1)))
    Q.add(SimParameter('c_t', UniformDistribution(0, 0.2)))
    Q.add(SimParameter('c_eh', UniformDistribution(0, 0.3)))
    Q.add(SimParameter('c_wd', UniformDistribution(0, 30)))
    Q.add(SimParameter('c_dp', UniformDistribution(0, 0.4)))
    return Q

# prior simparamset and y scaler
Q = set_prior_paramset()

# Read simulation data
#q, y = load_data()  # maybe also the coordinates of the displacements (x, y, z)

# Initiate surrogate model
p=2  # degree of approximation
model = GpcSurrogateModel(Q, p) # Initiate model, and store basis in model.basis
# Training the model (compute coefficients which can be reached by model.u_i_alpha)
#model.compute_coeffs_by_regression(q, y)
# Predict model response for any values of the parameters
#y = model.predict_response(q)
