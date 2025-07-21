import numpy as np
from scipy.stats.qmc import Halton
from scipy.stats.qmc import LatinHypercube as LHS
from scipy.stats.qmc import Sobol
from SALib.sample import saltelli
import scipy.special as sc
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt

# Set random seed
np.random.seed(seed=1234)

# Give back the item, if the array has only one element, else give back the array
def unwrap_if_scalar(arr):
    output = arr.item() if arr.size == 1 else arr
    return output

class Distribution(ABC):
    @abstractmethod
    def pdf(self, x): pass

    @abstractmethod
    def cdf(self, x): pass

    @abstractmethod
    def invcdf(self, y): pass

    @abstractmethod
    def mean(self): pass

    @abstractmethod
    def var(self): pass

    @abstractmethod
    def skew(self): pass

    @abstractmethod
    def kurt(self): pass

    #@abstractmethod
    #def orth_polysys_syschar(self, normalized): pass

    def moments(self):
        return [self.mean(), self.var(), self.skew(), self.kurt()]
    
    def logpdf(self, x):
        pdf = self.pdf(x)
        y = np.log(pdf)
        return y
    
    # def sample(self, n, method ='MC'): 
    #     sample_mu = 0
    #     sample_sigma = 1
    #     yi = np.random.rand(n)
    #     yi = (yi * sample_sigma) + sample_mu
    #     xi = self.invcdf(yi)
    #     return xi
    
    def sample(self, n, method='MC', **params):
        if method == 'MC':
            yi = np.random.rand(n)
        elif method == 'QMC_Halton':
            sampler = Halton(d=1)
            yi = sampler.random(n)
        elif method == 'QMC_LHS':
            sampler = LHS(d=1)
            yi = sampler.random(n)
        elif method == 'QMC_Sobol':
            sampler = Sobol(d=1)
            yi = sampler.random(n)
        # elif method == 'Sobol_saltelli': # It's not well implemented!!!
        #     yi = saltelli.sample(params['problem'], n)
        xi = self.invcdf(yi.squeeze())
        return xi

    def translate(self, shift, scale):
        tdist = TranslatedDistribution(self, shift, scale)
        return tdist
    
    def get_shift(self):
        return self.shift
    
    def get_scale(self):
        return self.scale
    
    def fix_moments(self, mean, var):
        old_mean, old_var = self.mean(), self.var()
        self.shift = mean - old_mean
        self.scale = np.sqrt(var/old_var)
        new_dist = self.translate(self.shift, self.scale)
        return new_dist
    
    def fix_bounds(self, min, max, q0=0, q1=1):
        if not (0 <= q0 <= 1):
            raise ValueError(f"q0 must be between 0 and 1, got {q0}")
        if not (q0 <= q1 <= 1):
            raise ValueError(f"q1 must be between q0 and 1, got {q1}")
        
        old_min = self.invcdf(q0)
        old_max = self.invcdf(q1)
        
        if not np.isfinite(old_min):
            # raise ValueError(f"Lower quantile (q0) gives infinity (unbounded distribution?)")
            print(f"Lower quantile (q0) gives infinity (unbounded distribution?). Using new q0=0.02")
            old_min, _ = self.get_bounds()
        if not np.isfinite(old_max):
            # raise ValueError(f"Upper quantile (q1) gives infinity (unbounded distribution?)")
            print(f"Upper quantile (q1) gives infinity (unbounded distribution?). Using new q1=0.98")
            _, old_max = self.get_bounds()
        
        center = self.mean()
        self.scale = ((max-min)/(old_max-old_min))
        self.shift = min - ((old_min-center)*self.scale + center)
        new_dist = self.translate(self.shift, self.scale)
        return new_dist
            
    def stdnor2base(self, x):
        y = self.invcdf(NormalDistribution().cdf(x))
        return y

    def base2stdnor(self, y):
        x = NormalDistribution().invcdf(self.cdf(y))
        return x
    
    def get_base_dist(self):
        dist_germ = NormalDistribution(0, 1)
        return dist_germ
        
    def base2dist(self, y):
        x = self.invcdf(self.get_base_dist().cdf(y))
        return x
    
    def dist2base(self, x):
        y = self.get_base_dist().invcdf(self.cdf(x))
        return y
    
    def orth_polysys(self):
        raise Exception(f"No polynomial system for this distribution ({self})")
    
    def get_bounds(self, delta=0.02):
        bounds = self.invcdf(np.array([delta, 1 - delta]))
        return bounds

class TranslatedDistribution(Distribution):
    def __init__(self, dist, shift, scale, center=None):
        self.dist = dist
        self.shift = shift
        self.scale = scale
        if center is None:
            self.center = self.dist.moments()[0]
        else:
            self.center = center
        
    def __repr__(self):
        return 'Translated({}, {}, {}, {})'.format(self.dist, self.shift, self.scale, self.center)
    
    def translate_points(self, x, forward):
        if forward:
            y = TranslatedDistribution.translate_points_forward(x, self.shift, self.scale, self.center)
        else:
            y = TranslatedDistribution.translate_points_backwards(x, self.shift, self.scale, self.center)
        return y
    
    def pdf(self, x):
        x = self.translate_points(x, False)
        y = self.dist.pdf(x)/self.scale        
        return y
    
    def cdf(self, x):
        x = self.translate_points(x, False)
        y = self.dist.cdf(x)/1
        return y
    
    def invcdf(self, y):
        x = self.dist.invcdf(y)
        x = self.translate_points(x, True)
        return x/1
    
    def mean(self):
        moments = self.dist.moments()
        moments = TranslatedDistribution.translate_moments(moments, self.shift, self.scale, self.center)
        mean = moments[0]
        return mean

    def var(self):
        moments = self.dist.moments()
        moments = TranslatedDistribution.translate_moments(moments, self.shift, self.scale, self.center)
        var = moments[1]
        return var

    def skew(self):
        moments = self.dist.moments()
        moments = TranslatedDistribution.translate_moments(moments, self.shift, self.scale, self.center)
        skew = moments[2]
        return skew
      
    def kurt(self):
        moments = self.dist.moments()
        moments = TranslatedDistribution.translate_moments(moments, self.shift, self.scale, self.center)
        kurt = moments[3]
        return kurt
    
    def sample(self, n):
        xi = self.dist.sample(n)
        xi = self.translate_points(xi, True)
        return xi
    
    def moments(self):
        moments = self.dist.moments()
        moments = TranslatedDistribution.translate_moments(moments, self.shift, self.scale, self.center)
        return moments
    
    def get_base_dist(self):
        dist_germ = self.dist.get_base_dist()
        return dist_germ
        
    @staticmethod
    def translate_points_forward(x, shift, scale, center):
        y = (x-center)*scale+center+shift
        return y
    
    @staticmethod
    def translate_points_backwards(x, shift, scale, center):
        y = (x-shift-center)/scale+center
        return y
    
    @staticmethod
    def translate_moments(m, shift, scale, center):
        if len(m)>=1:
            m[0] = TranslatedDistribution.translate_points_forward(m[0], shift, scale, center)
        if len(m) >=2:
            m[1] = m[1]*scale**2
        # Higher (standardized) moments like skewness or kurtosis are
        # not affected by neither shift nor scale
        return m

class NormalDistribution(Distribution):
    def __init__(self, mu=0, sigma=1):
        self.dist_type = 'norm'
        assert (sigma > 0)
        self.mu = mu
        self.sigma = sigma
        self.dist_params = [mu, sigma]

    def __repr__(self):
        return 'N({}, {:.2f})'.format(self.mu, self.sigma**2)
    
    def get_type(self):
        return 'Normal'
    
    def pdf(self, x):
        mu = self.mu
        sigma = self.sigma
        root = (x - mu) / sigma
        y_exp = root**2
        y_exp = -1 / 2 * y_exp
        y = np.exp(y_exp) / (sigma*np.sqrt(2*np.pi))
        return y
    
    def logpdf(self, x):
        mu = self.mu
        sigma = self.sigma
        root = (x - mu) / sigma
        y = -1 / 2 * (root**2) - np.log(sigma*np.sqrt(2*np.pi))
        return y
    
    def cdf(self, x):
        mu = self.mu
        sigma = self.sigma
        y = 1 / 2 * (1 + sc.erf( (x-mu) / (sigma * np.sqrt(2))))
        return y

    def invcdf(self, y):
        mu = self.mu
        sigma = self.sigma
        y = np.array(y)
        x = np.full(y.shape, np.nan) #original
        ind = (y >= 0) & (y <= 1)
        x[ind] = mu + sigma * np.sqrt(2) * sc.erfinv(2*y[ind]-1)
        x = x/1
        return x
    
    def sample(self, n, method ='MC', **params):
        if method == 'MC':
            xi = np.random.randn(n)
        else:
            xi = UniformDistribution().sample(n, method, **params)
        return (xi * self.sigma) + self.mu
    
    def mean(self):
        return self.mu

    def var(self):
        return self.sigma * self.sigma

    def skew(self):
        return 0

    def kurt(self):
        return 0
    
    def get_base_dist(self):
        dist_germ = NormalDistribution(0, 1)
        return dist_germ
    
    def translate(self, shift, scale):
        new_dist = NormalDistribution(self.mu+shift, self.sigma*scale)
        return new_dist

    def base2dist(self, y):
        return self.mu + y * self.sigma

    def dist2base(self, x):
        return (x - self.mu) / self.sigma
    
    def orth_polysys(self):
        from polysys import HermitePolynomials
        if self.mu == 0 and self.sigma == 1:
            polysys = HermitePolynomials()
        else:
            polysys = Distribution.orth_polysys(self)
        return polysys
    
    def orth_polysys_syschar(self, normalized):
        if self.mu == 0 and self.sigma == 1:
            if normalized:
                polysys_char = 'h'
            else:
                polysys_char = 'H'
        else:
            polysys_char = []
        return polysys_char

class UniformDistribution(Distribution):
    def __init__(self, a=0, b=1):
        self.dist_type = 'unif'
        self.a = a
        self.b = b
        self.dist_params = [a, b]

    def __repr__(self):
        return 'U({}, {})'.format(self.a, self.b)
    
    def get_type(self):
        return 'Uniform'

    def pdf(self, x):
        a = self.a
        b = self.b
        y = 1 / (b - a) * np.ones(np.size(x))
        #y = np.clip(y, 0, 0)
        y[x < a] = 0
        y[x > b] = 0
        return unwrap_if_scalar(y)
    
    def logpdf(self, x):
        a = self.a
        b = self.b
        pdf = self.pdf(x)
        pdf = np.array(pdf) # OR
        pdf = pdf.reshape(x.shape)
        # pdf = np.array([pdf]).flatten()
        y = np.zeros(x.shape)
        for i in range(len(x)):
            if pdf[i] == 0:
                y[i] = -np.Inf
            else:
                y[i] = np.log(pdf[i])
        return y
    
    def cdf(self, x):
        a = self.a
        b = self.b
        #x = np.asarray(x)
        y = (x - a) / (b - a)
        y = np.clip(y, 0, 1)
        return y

    def invcdf(self, y):
        a = self.a
        b = self.b
        y = np.array(y)
        x = np.full(np.size(y), np.nan)
        ind = (y >= 0) & (y <= 1)
        x[ind] = a + (b - a) * y[ind]
        return unwrap_if_scalar(x)

    # def sample(self, n, method='MC'):
    #     if method == 'MC':
    #         xi = np.random.rand(n)
    #     elif method == 'QMC':
    #         gen = ghalton.Halton(1)
    #         xi = np.array(gen.get(n))
    #     return self.invcdf(xi)

    def mean(self):
        return 0.5 * (self.a + self.b)

    def var(self):
        return (self.b - self.a) ** 2 / 12

    def skew(self):
        return 0

    def kurt(self):
        return -6 / 5

    def translate(self, shift, scale):
        m = (self.a + self.b) / 2
        v = scale * (self.b - self.a) / 2

        a = m + shift - v
        b = m + shift + v
        new_dist = UniformDistribution(a,b)
        return new_dist

    def get_base_dist(self):
        dist_germ = UniformDistribution(-1, 1)
        return dist_germ

    def base2dist(self, y):
        return self.mean() + y * (self.b - self.a) / 2

    def dist2base(self, x):
        return (x - self.mean()) * 2 / (self.b - self.a)
    
    def orth_polysys(self):
        from polysys import LegendrePolynomials
        if self.a == -1 and self.b == 1:
            polysys = LegendrePolynomials()
        else:
            polysys = Distribution.orth_polysys(self)
        return polysys

    def orth_polysys_syschar(self, normalized):
        if self.a == -1 and self.b == 1:
            if normalized:
                polysys_char = 'p'
            else:
                polysys_char = 'P'
        else:
            polysys_char = []
        return polysys_char
    
    def get_bounds(self, delta=0):
        a = self.a
        b = self.b

        ab = b - a
        bounds = np.array([a - ab*delta, b + ab*delta])
        return bounds
    
class LogNormalDistribution(Distribution):
    def __init__(self, mu=0, sigma=1):
        self.dist_type = 'lognorm'
        assert (sigma > 0)
        self.mu = mu
        self.sigma = sigma
        self.dist_params = [mu, sigma]
        
    def __repr__(self):
        return 'lnN({}, {})'.format(self.mu, self.sigma**2)
    
    def pdf(self, x):
        x = np.array(x)
        y = np.zeros(x.shape)
        ind = (x > 0)
        mu = self.mu
        sigma = self.sigma
        root = (np.log(x[ind]) - mu) / sigma
        y_exp = root**2
        y_exp = -1 / 2 * y_exp
        y[ind] = np.exp(y_exp) / (x[ind]*sigma*np.sqrt(2*np.pi))
        return unwrap_if_scalar(y)
    
    def logpdf(self, x):
        y = np.zeros(x.shape)
        ind = (x > 0)        
        mu = self.mu
        sigma = self.sigma
        root = (np.log(x[ind]) - mu) / sigma
        y = -1 / 2 * (root**2) - x[ind]*sigma*np.sqrt(2*np.pi)
        return y
    
    def cdf(self, x):
        x = np.array(x)
        y = np.zeros(x.shape)
        ind = (x > 0)
        mu = self.mu
        sigma = self.sigma
        y[ind] = 1 / 2 * (1 + sc.erf((np.log(x[ind])-mu) / (sigma * np.sqrt(2))))
        return unwrap_if_scalar(y)
    
    def invcdf(self, y):
        y = np.array(y)
        x = np.full(y.shape, np.nan)
        ind = (y >= 0) & (y <= 1)
        mu = self.mu
        sigma = self.sigma
        x[ind] = np.exp(mu + sigma*np.sqrt(2)*sc.erfinv(2*y[ind]-1))
        return x
    
    def sample(self, n, method ='MC', **params):
        if method == 'MC':
            xi = np.random.randn(n)
        else:
            xi = UniformDistribution().sample(n, method, **params)
        return np.exp((xi * self.sigma) + self.mu)
    
    def mean(self):
        return np.exp(self.mu + (self.sigma**2)/2)

    def var(self):
        return (np.exp(self.sigma**2)-1)*(np.exp(2*self.mu + self.sigma**2))

    def skew(self):
        return (np.exp(self.sigma**2)+2)*(np.sqrt(np.exp(self.sigma**2)-1))

    def kurt(self):
        return np.exp(4*self.sigma**2) + 2*np.exp(3*self.sigma**2) + 3*np.exp(2*self.sigma**2)-6
    
    def get_base_dist(self):
        base = NormalDistribution(0, 1)
        return base

    def base2dist(self, y):
        return np.exp(y*self.sigma + self.mu)

    def dist2base(self, x):
        # ignore RuntimeWarning in case x == 0
        with np.errstate(divide='ignore', invalid='ignore'):
            return (np.log(x) - self.mu)/self.sigma
    
    def stdnor2base(self, y): # same as base2dist??
        return np.exp(y*self.sigma + self.mu)
    
    def base2stdnor(self, x): # same as dist2base??
        return (np.log(x) - self.mu)/self.sigma
    
    def orth_polysys(self):
        from polysys import HermitePolynomials
        if self.mu == 0 and self.sigma == 1:
            polysys = HermitePolynomials()
        else:
            polysys = Distribution.orth_polysys(self)
        return polysys
    
    def orth_polysys_syschar(self, normalized):
        if self.mu == 0 and self.sigma == 1:
            if normalized:
                polysys_char = 'h'
            else:
                polysys_char = 'H'
        else:
            polysys_char = []
        return polysys_char
    
class BetaDistribution(Distribution):
    def __init__(self, a, b):
        self.dist_type = 'beta'
        self.a = a
        self.b = b
        self.dist_params = [a, b]
        
    def __repr__(self):
        return 'Beta({}, {})'.format(self.a, self.b)
    
    def pdf(self, x):
        x = TranslatedDistribution.translate_points_backwards(x, -1, 2, 0)
        x = np.array(x)
        y = np.zeros(x.shape)
        ind = (x >= 0) & (x <= 1)
        y[ind] = x[ind]**(self.a-1)*(1-x[ind])**(self.b-1) / sc.beta(self.a,self.b)
        return unwrap_if_scalar(y)
        
    def cdf(self, x):
        x = TranslatedDistribution.translate_points_backwards(x, -1, 2, 0)
        x = np.array(x)
        y = np.zeros(x.shape)
        ind = (x >= 0) & (x <= 1)
        y[ind] = sc.betainc(self.a, self.b, x[ind])
        y[x>1] = 1
        return unwrap_if_scalar(y)
    
    def invcdf(self, y):
        #TODO implementing the Matlab code
        y = np.array(y)
        x = np.full(y.shape, np.nan)
        ind = (y >= 0) & (y <= 1)
        x[ind] = sc.betaincinv(self.a, self.b, y[ind])
        x = TranslatedDistribution.translate_points_forward(x, -1, 2, 0)
        return unwrap_if_scalar(x)
    
    def moments(self):
        mean = self.a/(self.a + self.b)
        var = self.a*self.b/(((self.a+self.b)**2)*(self.a+self.b+1))
        skew = 2*(self.b-self.a)*np.sqrt(self.a+self.b+1)/((self.a+self.b+2)*np.sqrt(self.a*self.b))
        kurt = 6*(self.a**3-(self.a**2)*(2*self.b-1)+(self.b**2)*(self.b+1)-2*self.a*self.b*(self.b+2))/(self.a*self.b*(self.a+self.b+2)*(self.a+self.b+3))
        moments = [mean, var, skew, kurt]
        moments = TranslatedDistribution.translate_moments(moments, -1, 2, 0)
        return moments 
        
    def mean(self):
        mean = BetaDistribution.moments(self)[0]
        return mean
    
    def var(self):
        var = BetaDistribution.moments(self)[1]
        return var
    
    def skew(self):
        skew = BetaDistribution.moments(self)[2]
        return skew
    
    def kurt(self):
        kurt = BetaDistribution.moments(self)[3]
        return kurt
    
    def get_base_dist(self):
        return self
    
    def base2dist(self, y):
        x = y
        return x
    
    def dist2base(self, x):
        y = x
        return y
    
    def orth_polysys(self):
        from polysys import JacobiPolynomials
        polysys = JacobiPolynomials(self.b-1, self.a-1)
        return polysys
    
    # def orth_polysys_syschar(self, normalized):
    #     #TODO
    #     if self.mu == 0 and self.sigma == 1:
    #         if normalized:
    #             polysys_char = 'h'
    #         else:
    #             polysys_char = 'H'
    #     else:
    #         polysys_char = []
    #     return polysys_char
    #     polysys=JacobiPolynomials(dist.b-1, dist.a-1);

class ExponentialDistribution(Distribution):
    def __init__(self, lambda_):
        self.dist_type = 'exp'
        self.lambda_ = lambda_
        self.dist_params = [lambda_]
        
    def __repr__(self):
        return 'Exp({})'.format(self.lambda_)

    def pdf(self, x):
        x = np.array(x)
        y = np.zeros(x.shape)
        ind = (x>=0)
        y[ind] = self.lambda_*np.exp(-self.lambda_*x[ind])
        return unwrap_if_scalar(y)
    
    def cdf(self, x):
        x = np.array(x)
        y = np.zeros(x.shape)
        #y = np.zeros(np.size(x))
        ind = (x>=0)
        y[ind] = 1 - np.exp(-self.lambda_*x[ind])
        y = unwrap_if_scalar(y)
        return y
    
    def invcdf(self, y):
        y = np.array(y)
        x = np.full(np.size(y), np.nan)
        ind = (y>=0) & (y<=1)
        # ignore RuntimeWarning in case x == 0
        with np.errstate(divide='ignore', invalid='ignore'):
            x[ind] = -np.log(1-y[ind])/self.lambda_
        return unwrap_if_scalar(x)
    
    def mean(self):
        mean = 1/self.lambda_
        return mean
    
    def var(self):
        var = 1/self.lambda_**2
        return var
    
    def skew(self):
        return 2
    
    def kurt(self):
        return 6
    
    def sample(self, n, method='MC', **params):
        #uni = UniformDistribution() 
        yi = UniformDistribution().sample(n, method, **params)
        xi = self.invcdf(yi)
        return xi
    
    def orth_polysys(self):
        if self.lambda_:
            from polysys import LaguerrePolynomials
            polysys = LaguerrePolynomials()
        else:
            Distribution.orth_polysys()
        return polysys
    
    def get_base_dist(self):
        base = ExponentialDistribution(1)
        return base
    
    def base2dist(self, y):
        x = y/self.lambda_
        return x
    
    def dist2base(self, x):
        y = x*self.lambda_
        return y
        
    
if __name__ == "__main__":
    arr = np.arange(-10, 10, 1)
    print(arr)
    
    dist = NormalDistribution(0,3)
    print(dist.moments())
    problem = {
        'num_vars': 1,
        'names': ['x1'],
        'bounds': [[-3.14159265359, 3.14159265359]]
    }
    
    dist.sample(16, method='Sobol_saltelli', problem=problem)
    #plt.plot(arr, dist.pdf(arr))
    print(dist.pdf(arr))
    print(dist.logpdf(arr))
    #plt.plot(arr, dist.logpdf(arr))
    print(dist.cdf(arr))
    #plt.plot(arr, dist.cdf(arr))
    print(dist.get_base_dist().mu, dist.get_base_dist().sigma)
    print(dist.dist2base(np.array([-3, -2, -1, 0, 1, 2, 3])))
    print(dist.base2dist(np.array([-2, -1, -0.5, 0, 0.5, 1, 2])))
    print(dist.get_bounds())

    fig1, ax1 = plt.subplots(3)
    ax1[0].plot(arr, dist.pdf(arr))
    ax1[1].plot(arr, dist.logpdf(arr))
    ax1[2].plot(arr, dist.cdf(arr))



    print('------')

    dist = UniformDistribution(0,3)
    print(dist.moments())
    print(dist.pdf(arr))
    #plt.plot(arr, dist.logpdf(arr))
    print(dist.logpdf(arr))
    print(dist.cdf(np.array([-3, -2, -1, 0, 1, 2, 3])))
    print(dist.get_base_dist().a, dist.get_base_dist().b)
    print(dist.dist2base(np.array([-3, -2, -1, 0, 1, 2, 3])))
    print(dist.base2dist(np.array([-2, -1, -0.5, 0, 0.5, 1, 2])))
    print(dist.get_bounds())

    fig2, ax2 = plt.subplots(3)
    ax2[0].plot(arr, dist.pdf(arr))
    ax2[1].plot(arr, dist.logpdf(arr))
    ax2[2].plot(arr, dist.cdf(arr))
    
    
    
    print('------')

    dist = LogNormalDistribution(-3,3)
    print(dist.moments())
    print(dist.pdf(arr))
    #plt.plot(arr, dist.logpdf(arr))
    print(dist.logpdf(arr))
    print(dist.cdf(np.array([-3, -2, -1, 0, 1, 2, 3])))
    print(dist.get_base_dist().mu, dist.get_base_dist().sigma)
    #print(dist.dist2base(np.array([-3, -2, -1, 0, 1, 2, 3])))
    #print(dist.base2dist(np.array([-2, -1, -0.5, 0, 0.5, 1, 2])))
    ##print(dist.get_bounds())

    # fig2, ax2 = plt.subplots(3)
    # ax2[0].plot(arr, dist.pdf(arr))
    # ax2[1].plot(arr, dist.logpdf(arr))
    # ax2[2].plot(arr, dist.cdf(arr))

    plt.show()