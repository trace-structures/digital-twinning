import numpy as np
from distributions import UniformDistribution
from gpc_functions import syschar_to_polysys
class SimParameter():
    def __init__(self, name, dist_or_num, *args, **kwargs):
        self.name = name
        self.dist = dist_or_num if hasattr(dist_or_num, 'pdf') else None
        self.fixed_val = dist_or_num if not hasattr(dist_or_num, 'pdf') else None
        self.is_fixed = self.fixed_val is not None

    def __repr__(self):
        return 'Param({}, {})'.format(self.name, self.dist.__repr__())

    def mean(self):
        return self.dist.mean()

    def var(self):
        return self.dist.var()

    def moments(self):
        return self.dist.moments()

    def pdf(self, x):
        return self.dist.pdf(x)

    def cdf(self, p):
        return self.dist.cdf(p)
    
    def ppf(self, q): # same as invcdf
        return self.dist.invcdf(q)

    def sample(self, n):
        return self.dist.sample(n)

    def get_gpc_dist(self):
        return self.dist.get_base_dist()

    def get_gpc_polysys(self, normalized):
        syschar = self.get_gpc_syschar(normalized)
        return syschar_to_polysys(syschar)

    def get_gpc_syschar(self, normalized):
        return self.dist.get_base_dist().orth_polysys_syschar(normalized)

    def germ2param(self,x):
        q = self.dist.base2dist(x)
        return q

    def param2germ(self, q):
        x = self.dist.dist2base(q)
        return x
    
    
class MySimParameter(SimParameter):
    def __init__(self, name, dist_or_num, *args, **kwargs):
        super().__init__(name, dist_or_num, *args, **kwargs)

    def gpc_expand(self, normalized=True, syschar=None, expand_options=None):
        """
        Expands the parameter in the default polynomial system of the distribution.
        """
        if syschar is None:
            syschar = self.get_gpc_syschar(normalized)
        if expand_options is None:
            expand_options = {}
        # `gpc_param_expand` function is not defined yet
        #q_alpha, V_q, varerr = gpc_param_expand(self.dist, syschar, expand_options)
        #return q_alpha, V_q, varerr

    def linspace(self, n):
        """
        Generate n evenly spaced values between the parameter bounds.
        """
        y = np.array([0, 1])
        x_bounds = self.dist.ppf(y)  # Use inverse CDF (percent-point function)
        if np.isinf(x_bounds[0]):
            x_bounds[0] = self.dist.ppf(0.01)
        if np.isinf(x_bounds[1]):
            x_bounds[1] = self.dist.ppf(0.99)
        return np.linspace(x_bounds[0], x_bounds[1], n)

    def logpdf(self, x):
        """
        Return the log PDF of the parameter.
        """
        if self.is_fixed:
            abstol = 1e-10
            reltol = 1e-10
            x0 = self.fixed_val
            y = np.log((np.abs(x - x0) <= abstol + np.abs(x0) * reltol).astype(float))
        else:
            y = np.log(self.dist.pdf(x))
        return y

    def get_gpc_syschar(self, normalized):
        """
        Placeholder for GPC system characteristic retrieval.
        """
        # Implementáld az osztályod szerint!
        return "default_syschar"  # Példa

if __name__ == "__main__":
    from distributions import UniformDistribution
    P = SimParameter('p', UniformDistribution(-2,2))
    print(P.moments())
    print(P.pdf(np.array([-3, -2, -1, 0, 1, 2, 3])))
    print(P.cdf(np.array([-3, -2, -1, 0, 1, 2, 3])))
    print(P.get_gpc_polysys(True))
    print(P.param2germ(np.array([-3, -2, -1, 0, 1, 2, 3])))
    print(P.germ2param(np.array([-2, -1, -0.5, 0, 0.5, 1, 2])))
    print(P.get_gpc_syschar(False))
