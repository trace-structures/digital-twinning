import numpy as np
from scipy.stats.qmc import Halton
from scipy.stats.qmc import LatinHypercube as LHS
from scipy.stats.qmc import Sobol
from SALib.sample import saltelli

class SimParamSet():
    def __init__(self, normalized=True):
        self.normalized = normalized
        self.params = {}
        self.dist_types = []
        self.dist_params = []

    def add(self, simparam):
        if simparam.name in self.params.keys():
            raise("parameter name {} already exists in SimParamSet".format(simparam.name))
        self.params[simparam.name] = simparam.dist
        self.dist_types.append(simparam.dist.dist_type)
        self.dist_params.append(simparam.dist.dist_params)

    def filter(self, param_names):
        new_paramset = SimParamSet(normalized=self.normalized)
        new_params = {}
        new_dist_types = []
        new_dist_params = []
        # To avoid confusion, use a list of original names to map indices
        original_names = list(self.params.keys())
        for name in param_names:
            if name not in self.params:
                raise KeyError(f"Parameter {name} not found in SimParamSet")
            idx = original_names.index(name)
            new_params[name] = self.params[name]
            new_dist_types.append(self.dist_types[idx])
            new_dist_params.append(self.dist_params[idx])
        new_paramset.params = new_params
        new_paramset.dist_types = new_dist_types
        new_paramset.dist_params = new_dist_params
        return new_paramset

    def num_params(self):
        return len(self.params)

    def param_names(self):
        return list(self.params.keys())
    
    def get_params(self):
        return self.params

    def mean(self):
        m = self.num_params()
        params = self.params
        q_mean = np.zeros([m,1])
        for i, dist in enumerate(params.values()):
            q_mean[i] = dist.mean()
        return q_mean

    def var(self):
        m = self.num_params()
        params = self.params
        var = np.zeros([m, 1])
        for i, dist in enumerate(params.values()):
            var[i] = dist.var()
        return var

    def pdf(self, q):
        m = self.num_params()
        assert(q.shape[0] == m)
        n = q.shape[1]

        params = self.params
        p_q = np.ones([1, n])
        for i, dist in enumerate(params.values()):
            p_q = p_q * dist.pdf(q[i])
        return p_q

    def logpdf(self, q):
        m = self.num_params()
        assert(q.shape[0] == m)
        n = q.shape[1]

        params = self.params
        p_q = np.zeros([1, n])
        for i, dist in enumerate(params.values()):
            p_q = p_q + dist.logpdf(q[i])
        return p_q

    def cdf(self, q):
        m = self.num_params()
        assert (q.shape[0] == m)
        n = q.shape[1]

        params = self.params
        p_q = np.empty([m, n])
        for i, dist in enumerate(params.values()):
            p_q[i,:] = dist.cdf(q[i,:])

    def get_gpc_syschars(self):
        params = self.params
        syschar = ''
        for i, dist in enumerate(params.values()):
            syschar = syschar + dist.get_base_dist().orth_polysys_syschar(True)
        return syschar

    def germ2params(self, xi_i_k):
        m = self.num_params()
        assert (xi_i_k.shape[0] == m)
        n = xi_i_k.shape[1]

        params = self.params
        q_i_k = np.zeros([m, n])
        for i, dist in enumerate(params.values()):
            q_i_k[i, :] = dist.base2dist(xi_i_k[i,:])
        return q_i_k

    def params2germ(self, q_i_k):
        m = self.num_params()
        assert (q_i_k.shape[0] == m)
        n = q_i_k.shape[1]

        params = self.params
        xi_i_k = np.zeros([m, n])
        for i, dist in enumerate(params.values()):
            xi_i_k[i, :] = dist.dist2base(q_i_k[i,:])
        return xi_i_k
    
    def get_bounds(self):
        m = self.num_params()
        bounds = np.zeros((m, 2))
        for i, dist in enumerate(self.params.values()):
            bounds[i, :] = dist.get_bounds()
        return bounds


    def sample(self, n, method='MC', random_seed=None, **kwargs):
        m = self.num_params()
        q_i_k = np.zeros([m, n])
        params = self.params
        if method == 'MC':
            np.random.seed(random_seed)
            xi = np.random.rand(m,n)
        # QMC methods
        elif method == 'QMC_Halton':
            gen = Halton(m, seed=random_seed)
            xi = gen.random(n).T
        elif method == 'QMC_LHS':
            sampler = LHS(d=m, seed=random_seed)
            xi = sampler.random(n).T
        elif method == 'QMC_Sobol':
            sampler = Sobol(d=m, seed=random_seed)
            xi = sampler.random(n).T
        elif method == 'Sobol_saltelli':
            #problem ={'num_vars': m, 'names': self.param_names(), 'bounds': self.get_bounds()}
            problem = {'num_vars': m, 'names': self.param_names(), 'dists': self.dist_types, 'bounds': self.dist_params}
            #xi = saltelli.sample(problem, n)[:n,:].T
            xi = saltelli.sample(problem, n).T
            return xi.transpose()
            
        for i, dist in enumerate(params.values()):
            q_i_k[i,:] = dist.invcdf(xi[i,:])
            
        return q_i_k.transpose()
    
    def diminished_paramset(self, indexes):
        assert self.num_params() >= len(indexes)
        # assert indexes.all() <= self.num_params()

        diminished_paramnames = np.array(self.param_names())[indexes]
        new_paramset = SimParamSet(normalized=self.normalized)
        new_params = {}
        new_dist_types = []
        new_dist_params = []
        for i in range(len(indexes)):
            new_params[diminished_paramnames[i]] = self.params[diminished_paramnames[i]]
            new_dist_types.append(self.dist_types[indexes[i]])
            new_dist_params.append(self.dist_params[indexes[i]])
        new_paramset.params = new_params
        new_paramset.dist_types = new_dist_types
        new_paramset.dist_params = new_dist_params
        return new_paramset


class MySimParamSet(SimParamSet):
    def __init__(self, normalized=True):
        super().__init__(normalized)

    def add(self, param, *args):
        """
        Add a parameter to the param set. If the parameter is a string,
        create a new SimParameter with the given arguments.
        """
        if isinstance(param, str):
            from simparameter import SimParameter  # Feltételezve, hogy a SimParameter importálható
            param = SimParameter(param, *args)
        
        if param.name in self.params:
            print(f"Warning: Parameter name {param.name} already exists in SimParamSet and will be overwritten.")
        
        super().add(param)

    def params2struct(self, x):
        """
        Convert parameter values into a dictionary.
        """
        assert x.shape[0] == self.num_params(), \
            f"x must have {self.num_params()} rows, but got {x.shape[0]}"
        
        param_struct = {}
        for i, name in enumerate(self.param_names()):
            param_struct[name] = x[i, :]
        return param_struct

    def stdnor2params(self, xi):
        """
        Convert standard normal variables to parameter values.
        """
        assert xi.shape[0] == self.num_params(), \
            f"xi must have {self.num_params()} rows, but got {xi.shape[0]}"
        
        n = xi.shape[1]
        params = np.zeros((self.num_params(), n))
        
        for i, (name, dist) in enumerate(self.params.items()):
            if isinstance(dist, NormalDistribution):
                params[i, :] = dist.base2dist(xi[i, :])
            else:
                params[i, :] = dist.stdnor(xi[i, :])
        
        return params

    def params2stdnor(self, params):
        """
        Convert parameter values to standard normal variables.
        """
        assert params.shape[0] == self.num_params(), \
            f"params must have {self.num_params()} rows, but got {params.shape[0]}"
        
        n = params.shape[1]
        xi = np.zeros((self.num_params(), n))
        
        for i, (name, dist) in enumerate(self.params.items()):
            if isinstance(dist, NormalDistribution):
                xi[i, :] = dist.dist2base(params[i, :])
            else:
                xi[i, :] = dist.dist2stdnor(params[i, :])
        
        return xi

    def logpdf(self, q):
        """
        Calculate the log of the probability density function of the parameter set.
        """
        assert q.shape[0] == self.num_params(), \
            f"q must have {self.num_params()} rows, but got {q.shape[0]}"
        
        n = q.shape[1]
        log_p_q = np.zeros(n)
        
        for i, (name, dist) in enumerate(self.params.items()):
            log_p_q += dist.logpdf(q[i, :])
        
        return log_p_q

    def set_all_to_mean(self):
        """
        Set all parameters to the mean value of their distribution.
        """
        for name, dist in self.params.items():
            if hasattr(dist, 'mean'):
                mean_value = dist.mean()
                self.params[name].set_fixed(mean_value)


if __name__ == "__main__":
    from distributions import UniformDistribution
    from distributions import NormalDistribution
    from simparameter import SimParameter
    P1 = SimParameter('p1', UniformDistribution(-2,2))
    P2 = SimParameter('p2', NormalDistribution(0,2))

    Q = SimParamSet()
    Q.add(P1)
    Q.add(P2)

    print(Q.mean())
    print(Q.pdf(np.array([-3, -2, -1, 0, 1, 2, 3]*2).reshape(2,-1)))
    print(Q.cdf(np.array([-3, -2, -1, 0, 1, 2, 3]*2).reshape(2,-1)))
    print(Q.get_gpc_syschars())
    print(Q.params2germ(np.array([-3, -2, -1, 0, 1, 2, 3]*2).reshape(2,-1)))
    # print(Q.germ2params(np.array([-2, -1, -0.5, 0, 0.5, 1, 2]*2).reshape(2,-1)))
    Q.sample(10)
    Q.sample(10, method='Sobol_saltelli')

    print(Q.get_bounds())




