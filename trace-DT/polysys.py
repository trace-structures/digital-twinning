import numpy as np
import scipy.special as sp
from distributions import UniformDistribution
from distributions import NormalDistribution
from distributions import BetaDistribution

from abc import ABC, abstractmethod

class PolynomialSystem(ABC):

    def evaluate(self, deg, xi):
        k = np.size(xi)
        p = np.zeros([k, deg+2])
        p[:,0] = 0
        p[:,1] = 1
        r = self.recur_coeff(deg+1)
        for d in range(deg):
           p[:, d+2] = (r[d,0] + xi * r[d,1]) * p[:, d+1] - r[d,2] * p[:,d]
        y_alpha_j = p[:,1:]
        return y_alpha_j

    def sqnorm(self, n):
        deg = max(n.flatten()) + 1
        r = self.recur_coeff(deg)
        nrm2 = self.sqnorm_by_rc(r)
        nrm2 = np.reshape([nrm2[n+1], len(n)])
        return nrm2

    def sqnorm_by_rc(self, rc):
        b = rc[:, 1]
        h = b[0] / b[1:]
        c = rc[1:, 2]
        nrm2 = np.concatenate(np.ones([1]),  h.flatten * np.cumprod(c.flatten()))
        return nrm2

    def normalized(self):
        polysys = NormalizedPolynomials(self)
        return polysys

    @abstractmethod
    def weighting_dist(self):
        pass

    @abstractmethod
    def recur_coeff(self, deg):
        pass


class NormalizedPolynomials(PolynomialSystem):
    def __init__(self, base_polysys):
        self.base_polysys = base_polysys

    def recur_coeff(self, deg):
        r = self.base_polysys.recur_coeff(deg)
        n = np.array(range(deg))
        z = np.concatenate((np.zeros([1]), np.sqrt(self.base_polysys.sqnorm(np.arange(0,deg+1)))), axis=0)
        r = np.array([r[:, 0]*z[n + 1] / z[n + 2],
            r[:, 1] * z[n + 1] / z[n + 2],
            r[:, 2] * z[n] / z[n + 2]])
        return r.transpose()

    def weighting_dist(self):
        dist = self.base_polysys.weighting_dist()
        return dist

class LegendrePolynomials(PolynomialSystem):
    # def __init__(self):

    @classmethod
    def normalized(self):
        polysys = NormalizedPolynomials(self)
        return polysys

    @staticmethod
    def recur_coeff(deg):
        n = np.array(range(deg)).reshape(-1,1)
        zer = np.zeros(n.shape).reshape(-1,1)
        r = np.concatenate((zer, (2*n+1)/(n+1), n/(n+1)), axis=1)
        return r

    @staticmethod
    def sqnorm(n):
        nrm2 = 1/(2*n + 1)
        return nrm2

    @staticmethod
    def weighting_dist():
        dist = UniformDistribution(-1,1)
        return dist

class HermitePolynomials(PolynomialSystem):
    @classmethod
    def normalized(self):
        polysys = NormalizedPolynomials(self)
        return polysys

    @staticmethod
    def recur_coeff(deg):
        n = np.arange(deg)
        one = np.ones_like(n)
        zero = np.zeros_like(n)
        r = np.column_stack((zero, one, n))
        # n = np.array(range(deg)).reshape(-1,1)
        # on = np.ones(n.shape).reshape(-1,1)
        # zer = np.zeros(n.shape).reshape(-1,1)
        # r = np.concatenate((zer, on, n), axis = 1)
        return r

    @staticmethod
    def sqnorm(n):
        nrm2 = sp.factorial(n)
        return nrm2

    @staticmethod
    def weighting_dist():
        dist = NormalDistribution(0,1)
        return dist
    
class JacobiPolynomials(PolynomialSystem):
    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta
        
    def normalized(self):
        polysys = NormalizedPolynomials(self)
        return polysys

    def recur_coeff(self, deg):
        n = np.arange(deg)
        a = self.alpha
        b = self.beta
        
        b_n = (2*n+a+b+1)*(2*n+a+b+2)/( 2*(n+1)*(n+a+b+1) )
        a_n = (a^2-b^2)*(2*n+a+b+1)/( 2*(n+1)*(n+a+b+1)*(2*n+a+b) )
        c_n = (n+a)*(n+b)*(2*n+a+b+2)/( (n+1)*(n+a+b+1)*(2*n+a+b) )
        
        if a+b==0 or a+b==-1:
            b_n[0]=0.5*(a+b)+1
            a_n[0]=0.5*(a-b)
            c_n[0]=0
            
        r = [a_n, b_n, c_n]
            
        return r

    def sqnorm(self, n):
        nrm2 = PolynomialSystem.sqnorm(self, n)
        return nrm2

    def weighting_dist(self):
        dist = BetaDistribution(self.beta+1, self.alpha+1)
        return dist

class ChebyshevTPolynomials(PolynomialSystem):
    @classmethod
    def normalized(self):
        polysys = NormalizedPolynomials(self)
        return polysys

    @staticmethod
    def recur_coeff(deg):
        return []

    @staticmethod
    def sqnorm(n):
        nrm2 = []
        return nrm2

    @staticmethod
    def weighting_dist():
        # dist = UniformDistribution(-1, 1)
        dist = []
        return dist


class ChebyshevUPolynomials(PolynomialSystem):
    @classmethod
    def normalized(self):
        polysys = NormalizedPolynomials(self)
        return polysys

    @staticmethod
    def recur_coeff(deg):
        return []

    @staticmethod
    def sqnorm(n):
        nrm2 = []
        return nrm2

    @staticmethod
    def weighting_dist():
        # dist = UniformDistribution(-1, 1)
        dist = []
        return dist

class LaguerrePolynomials(PolynomialSystem):
    @classmethod
    def normalized(self):
        polysys = NormalizedPolynomials(self)
        return polysys

    @staticmethod
    def recur_coeff(deg):
        return []

    @staticmethod
    def sqnorm(n):
        nrm2 = []
        return nrm2

    @staticmethod
    def weighting_dist():
        # dist = UniformDistribution(-1, 1)
        dist = []
        return dist


class Monomials(PolynomialSystem):
    @classmethod
    def normalized(self):
        polysys = NormalizedPolynomials(self)
        return polysys

    @staticmethod
    def recur_coeff(deg):
        return []

    @staticmethod
    def sqnorm(n):
        nrm2 = []
        return nrm2

    @staticmethod
    def weighting_dist():
        # dist = UniformDistribution(-1, 1)
        dist = []
        return dist

if __name__ == "__main__":
    LegendrePolynomials.recur_coeff(4)
    LegendrePolynomials.normalized()
    LegendrePolynomials.sqnorm(4)
    LegendrePolynomials.sqnorm(3)
    LegendrePolynomials.normalized().recur_coeff(4)




