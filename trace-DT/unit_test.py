from distributions import Distribution, TranslatedDistribution, NormalDistribution, UniformDistribution, LogNormalDistribution, BetaDistribution, ExponentialDistribution
import numpy as np
import math
from scipy.integrate import quad, quad_vec
import scipy.special as sc

def assert_equals(actual, expected, assert_id=None, **kwargs):
    """
    Check the equality of actual and expected values.
    
    Parameters:
        actual: The actual value.
        expected: The expected value.
        assert_id: Identifier for the assertion.
        kwargs: Additional options for comparison, such as 'abstol', 'reltol', 'fuzzy', etc.
    """
      
    # Check if assert_id is None
    if assert_id is None:
        assert_id = ''
    
    fuzzy = kwargs.get('fuzzy', False)
    
    result_list = compare_objects(actual, expected, assert_id, kwargs)
    process_assert_results(result_list, assert_id, fuzzy)

def compare_objects(actual, expected, assert_id, options):
    result_list = compare_types(actual, expected, assert_id)
    if result_list:
        return result_list
    
    result_list = compare_size(actual, expected, assert_id)
    if result_list:
        return result_list
    
    return compare_content(actual, expected, assert_id, options)

def compare_types(actual, expected, assert_id):
    result_list = []
    if not (isinstance(actual, (int, float, np.ndarray)) and isinstance(expected, (int, float, np.ndarray))):
        if type(actual) != type(expected):
            msg = f"Classes don't match: {type(actual)} != {type(expected)}"
            result_list.append((msg, assert_id))
            return result_list
    return result_list

def compare_size(actual, expected, assert_id):
    result_list = []
    if isinstance(actual, np.ndarray) and isinstance(expected, np.ndarray):
        if actual.shape != expected.shape:
            msg = f"Size doesn't match: {actual.shape} != {expected.shape}"
            result_list.append((msg, assert_id))
            return result_list
    return result_list

def compare_content(actual, expected, assert_id, options):
    if isinstance(actual, int) and isinstance(expected, int):
        return compare_int(actual, expected, assert_id)    
    elif isinstance(actual, float) and isinstance(expected, float):
        return compare_float(actual, expected, assert_id, options)
    elif isinstance(actual, np.ndarray) and isinstance(expected, np.ndarray):
        return compare_double(actual, expected, assert_id, options)
    elif isinstance(actual, list) and isinstance(expected, list):
        return compare_list(actual, expected, assert_id, options)
    elif isinstance(actual, bool) and isinstance(expected, bool):
        return compare_logical(actual, expected, assert_id)
    elif isinstance(actual, str) and isinstance(expected, str):
        return compare_char(actual, expected, assert_id)
    else:
        msg = f"Don't know how to compare types {type(actual)} and {type(expected)}"
        return [(msg, assert_id)]
    
def compare_int(actual, expected, assert_id):
    result_list = []
    if actual != expected:
        msg = f"Integer values don't match: {actual} != {expected}"
        result_list.append((msg, assert_id))
    return result_list

def compare_float(actual, expected, assert_id, options):
    result_list = []
    
    # Get the absolute tolerance and relative tolerance from kwargs
    abstol = options.get('abstol', 1e-8)
    reltol = options.get('reltol', 1e-8)
    
    # Compare the floats with tolerance
    if not np.isclose(actual, expected, atol=abstol, rtol=reltol):
        msg = f"Float values don't match: {actual} != {expected} (within tolerance: abstol={abstol}, reltol={reltol})"
        result_list.append((msg, assert_id))
    return result_list

def compare_double(actual, expected, assert_id, options):
    abstol = options.get('abstol', 1e-8)
    reltol = options.get('reltol', 1e-8)
    result_list = []
    
    if not np.allclose(actual, expected, atol=abstol, rtol=reltol):
        msg = f"Values don't match: {actual} != {expected}"
        result_list.append((msg, assert_id))
    
    return result_list

def compare_list(actual, expected, assert_id, options):
    abstol = options.get('abstol', 1e-8)
    reltol = options.get('reltol', 1e-8)
    result_list = []
    
    if not np.allclose(actual, expected, atol=abstol, rtol=reltol):
        msg = f"Values don't match: {actual} != {expected}"
        result_list.append((msg, assert_id))
    
    return result_list

def compare_logical(actual, expected, assert_id):
    result_list = []
    if actual != expected:
        msg = f"Logical values don't match: {actual} != {expected}"
        result_list.append((msg, assert_id))
    return result_list

def compare_char(actual, expected, assert_id):
    result_list = []
    if actual != expected:
        msg = f"String values don't match: '{actual}' != '{expected}'"
        result_list.append((msg, assert_id))
    return result_list

def process_assert_results(result_list, assert_id, fuzzy):
    if result_list:
        for result in result_list:
            msg, assert_id = result
            print(f"Assertion failed [{assert_id}]: {msg}")
            raise AssertionError(f"Assertion failed [{assert_id}]: {msg}")
    else:
        print(f"Assertion passed [{assert_id}]")

        

def compute_moments(dist, tol=1e-6):
    dom = dist.invcdf(np.array([0,1]))
    dom.sort()
    
    # 0th moment: should be one
    fun_m0 = lambda x: dist.pdf(x)
    m0, _ = quad(fun_m0, dom[0], dom[1], epsrel=tol, epsabs=tol)

    # 1st moment: the mean
    fun_m1 = lambda x: x * dist.pdf(x)
    m1, _ = quad(fun_m1, dom[0], dom[1], epsrel=tol, epsabs=tol)

    # 2nd central moment: the variance
    fun_m2 = lambda x: (x - m1)**2 * dist.pdf(x)
    m2, _ = quad(fun_m2, dom[0], dom[1], epsrel=tol, epsabs=tol)

    # 3rd standardized moment: the skewness
    fun_m3 = lambda x: ((x - m1) / np.sqrt(m2))**3 * dist.pdf(x)
    m3, _ = quad(fun_m3, dom[0], dom[1], epsrel=tol, epsabs=tol)

    # 4th standardized moment: the kurtosis (subtract 3 to get excess kurtosis)
    fun_m4 = lambda x: ((x - m1) / np.sqrt(m2))**4 * dist.pdf(x)
    m4, _ = quad(fun_m4, dom[0], dom[1], epsrel=tol, epsabs=tol)
    m4 = m4 - 3

    return [m0, m1, m2, m3, m4]

def normal_stdnor(x, mu=0, sigma=1):
    y = mu+sigma*x
    return y

def linspace_midpoints(x_min, x_max, N=100):
    x = np.linspace(x_min, x_max, N+1)
    xmid = (x[1:] + x[:-1]) / 2
    return xmid

def compute_gramian(polysys, dist=None, N=4, tol=1e-6):
    if dist is None:
        dist = polysys.weighting_dist()
    dom = dist.invcdf([0,1])
    def fun(x):
        poly_eval = polysys.evaluate(N, x)
        out = np.outer(poly_eval, poly_eval) * dist.pdf(x)
        #return np.outer(poly_eval, poly_eval) * dist.pdf(x)
        #print(out[0])
        return out
    Q, _ = quad_vec(fun, dom[0], dom[1], epsrel=tol, epsabs=tol)
    #Q = quad(fun, dom[0], dom[1], epsrel=tol, epsabs=tol)
    return Q
    
def pdf_integrate(f,F,x):
    F0_array = F[0].reshape(-1,1)
    f_array = f.reshape(-1,1)
    F2 = np.cumsum(np.concatenate([F0_array, f_array])) * ([x[1]-x[0]])
    return F2

def uniform_stdnor(x,a=0,b=1):
    y = a+0.5*(b-a)*(1+sc.erf(x/np.sqrt(2)))
    return y
    
def lognormal_stdnor(x, mu=0, sigma=1):
    y = np.exp(mu+sigma*x)
    return y


###########################################################################################################################
####################################################     UNITTESTS     ####################################################
###########################################################################################################################

################################################## TranslatedDistribution #################################################

def unittest_TranslatedDistribution():
    ## Initialization
    N = NormalDistribution(0.4,0.2)
    T = TranslatedDistribution(N,4,5)

    assert_equals( T.shift, 4, 'Initialization shift' )
    assert_equals( T.scale, 5, 'Initialization scale' )
    assert_equals( T.center, 0.4, 'Initialization center=mean' )
    assert_equals( T.__repr__(), 'Translated(N(0.4, 0.04), 4, 5, 0.4)', '__repr__' )

    T = TranslatedDistribution(N,4,5,1)
    assert_equals( T.center, 1, 'Initialization center' )

    ## Mean & Var
    T = TranslatedDistribution(N,4,5)
    assert_equals( T.mean(), 0.4 + 4, 'mean' )
    assert_equals( T.var(), 0.04 * 25, 'var' )

    T = TranslatedDistribution(N,4,5,1)
    assert_equals( T.mean(), (0.4-1)*5 + 4 + 1, 'mean' )
    assert_equals( T.var(), 0.04 * 25, 'var' )

    ## Moments
    T = TranslatedDistribution(N, 4, 5, 1)
    m_act = [1, 0, 0, 0, 0]
    m_act[1:] = T.moments()
    m_ex = compute_moments(T)
    assert_equals( m_act, m_ex, 'moments' )

    ## PDF
    LN = LogNormalDistribution(3,3)
    T = TranslatedDistribution(LN,0.25,1)
    assert_equals( T.pdf(1/2), 0.1826612, 'pdf_median', abstol=0.001 ) 

    ## CDF
    LN = LogNormalDistribution(3,3)
    T = TranslatedDistribution(LN,0.25,1)
    assert_equals( T.cdf(1/2), 0.07185716, 'cdf_median', abstol=1e-08 ) ## abstol

    # ## INVCDF

    # ## Sample
    # munit_control_rand('seed', 1234)
    E = ExponentialDistribution(1.3)
    T = TranslatedDistribution(E,4,5,1)
    N = 100000
    xi = T.sample(N)
    xi.sort()
    assert_equals( T.cdf(xi), linspace_midpoints(0,1,N), 'sample_cdf', abstol=1e-2 )

    ## Moments
    N = NormalDistribution(4,1)
    T = TranslatedDistribution(N,2,3,2)
    [mean,var,skew,kurt]=T.moments()
    assert_equals( mean, 10, 'moments1' )
    assert_equals( var, 9, 'moments2' )
    assert_equals( skew, 0, 'moments3' )
    assert_equals( kurt, 0, 'moments4' )

    ## Fix Moments
    # can test directly for the normal and uniform distributions
    N = NormalDistribution(2,5)
    T = TranslatedDistribution(N,0,1)
    dist = T.fix_moments(7,13)
    assert_equals( [dist.shift, dist.scale], [5, np.sqrt(13/25)], 'normal', abstol=0.0001 )

    U = UniformDistribution(22,88)
    T = TranslatedDistribution(U,0,1)
    dist=T.fix_moments(50, 3 )
    assert_equals( [dist.shift, dist.scale], [-5, 1/11], 'uniform' )

    # can test via the moments for the lognormal distribution
    LN = LogNormalDistribution(0,1)
    T = TranslatedDistribution(LN,0,1)
    dist = T.fix_moments(3.1, 2.4 )
    [mean, var, skew, kurt] = dist.moments()
    assert_equals( mean, 3.1, 'mean' )
    assert_equals( var, 2.4, 'var' )

    # change a second time
    dist = dist.fix_moments(7, 5)
    [mean, var, skew, kurt] = dist.moments()
    assert_equals( mean, 7., 'mean2' )
    assert_equals( var, 5., 'var2' )

    ## Fix Bounds
    # test for the uniform distribution
    U = UniformDistribution(2,3)
    T = TranslatedDistribution(U,3,0.5)
    dist = T.fix_bounds(2,4)
    assert_equals( dist.invcdf(0), 2., 'fix_bounds-uni_min' )
    assert_equals( dist.invcdf(1), 4., 'fix_bounds-uni_max' )

    # test with quantiles for the normal distribution
    N = NormalDistribution(2,3)
    T = TranslatedDistribution(N,0,1,2)
    dist = T.fix_bounds(2, 4, q0=0.001, q1=0.5)
    assert_equals( dist.invcdf(0.001), 2., 'fix_bounds-nor_min' )
    assert_equals( dist.invcdf(0.5), 4., 'fix_bounds-nor_max' )
    

################################################### NormalDistribution ##################################################

def unittest_NormalDistribution():
    ## Initialization
    N = NormalDistribution(2,3)
    assert_equals( N.mu, 2, 'Initialization default mu' )
    assert_equals( N.sigma, 3, 'Initialization default sigma' )
    assert_equals( N.__repr__(), 'N(2, 9.00)', '__repr__' )

    N = NormalDistribution(2)
    assert_equals( N.mu, 2, 'Initialization default mu' )
    assert_equals( N.sigma, 1, 'Initialization default sigma' )

    N = NormalDistribution()
    assert_equals( N.mu, 0, 'Initialization default mu' )
    assert_equals( N.sigma, 1, 'Initialization default sigma' )

    ## Mean & Var
    assert_equals( N.mean(), 0, 'mean' )
    assert_equals( N.var(), 1, 'var' )

    ## Moments
    N = NormalDistribution(1.2, 2.3)
    m_act = [1, 0, 0, 0, 0]
    m_act[1:] = N.moments()
    m_ex = compute_moments(N)
    assert_equals( m_act, m_ex, 'moments' )

    ## normal_cdf
    N = NormalDistribution(1,2)
    assert_equals( N.cdf(-math.inf), 0., 'cdf_minf' )
    assert_equals( N.cdf(math.inf), 1., 'cdf_inf' )
    assert_equals( N.cdf(N.mu), 1/2, 'cdf_median' )

    ## normal_pdf
    N = NormalDistribution(1,2)
    assert_equals( N.pdf(-math.inf), 0., 'pdf_minf' )
    assert_equals( N.pdf(math.inf), 0., 'pdf_inf' )

    ## normal_invcdf

    y = np.linspace(0, 1)
    x = np.linspace(-2, 3)

    N = NormalDistribution()
    assert_equals( N.cdf(N.invcdf(y)), y, 'cdf_invcdf_1', abstol=1e-7 )
    assert_equals( N.invcdf(N.cdf(x)), x, 'invcdf_cdf_1', abstol=1e-7 )
    assert_equals( [np.isnan(N.invcdf([-0.1, 1.1]))], [True, True], 'invcdf_nan1' )

    N = NormalDistribution(0.5)
    assert_equals( N.cdf(N.invcdf(y)), y, 'cdf_invcdf_2', abstol=1e-7 )
    assert_equals( N.invcdf(N.cdf(x)), x, 'invcdf_cdf_2', abstol=1e-7 )
    assert_equals( [np.isnan(N.invcdf([-0.1, 1.1]))], [True, True], 'invcdf_nan2' )

    N = NormalDistribution(0.7,1.5)
    assert_equals( N.cdf(N.invcdf(y)), y, 'cdf_invcdf_3', abstol=1e-7 )
    assert_equals( N.invcdf(N.cdf(x)), x, 'invcdf_cdf_3', abstol=1e-7 )
    assert_equals( [np.isnan(N.invcdf([-0.1, 1.1]))], [True, True], 'invcdf_nan3' )

    ## Sample
    #munit_control_rand('seed', 1234)
    N = NormalDistribution(0.7,1.5)
    Ni = 100000
    xi = N.sample(Ni)
    xi.sort()
    assert_equals( N.cdf(xi), linspace_midpoints(0,1,Ni), 'sample_cdf', abstol=1e-2 )


    ## normal_stdnor
    N = 50
    uni = np.linspace(0,1,N+2)
    uni = uni[1:-1]
    gam = np.sqrt(2)*sc.erfinv(2*uni-1)

    N = NormalDistribution(0.2,0.3)
    x = N.stdnor2base( gam )
    assert_equals( N.cdf(x), uni, 'normal', abstol=1e-7 )
    N = NormalDistribution(0,1)
    assert_equals( normal_stdnor(gam), N.stdnor2base(gam), 'normal_def12', abstol=1e-7 )
    assert_equals( normal_stdnor(gam, 0), N.stdnor2base(gam), 'normal_def2', abstol=1e-7 )

    ## translate
    N = NormalDistribution(2,3)
    tN = N.translate(2,3)
    [m, v, s, k] = tN.moments()
    assert_equals( m, 4, 'translated mean' )
    assert_equals( v, 81, 'translated var' )
    assert_equals( tN.pdf(0), 0.0401582, 'translated pdf', abstol=1e-7 )
    assert_equals( tN.pdf(math.inf), 0., 'translated pdf' )
    assert_equals( tN.cdf(0), 0.328360, 'translated cdf', abstol=0.0001 )
    assert_equals( tN.cdf(math.inf), 1., 'translated cdf' )
    assert_equals( tN.cdf(-math.inf), 0., 'translated cdf' )
    assert_equals( tN.invcdf(0), -math.inf, 'translated cdf' )

    ## Fix Moments
    # can test directly for the normal and uniform distributions
    N = NormalDistribution(2,5)
    dist = N.fix_moments(7,13)
    [m, v, s, k] = dist.moments()
    assert_equals( m, 7, 'mean fix_moments' )
    assert_equals( v, 13., 'var fix_moments' )

    ## Fix Bounds
    N = NormalDistribution(2,3)
    dist = N.fix_bounds( 2, 4, q0=0.001, q1= 0.5 )
    assert_equals( dist.invcdf(0.001), 2., 'fix_bounds-nor_min' )
    assert_equals( dist.invcdf(0.5), 4., 'fix_bounds-nor_max' )

    # assert_error(@()(fix_bounds(N, 2, 4, q0=0, q1=0.5)), 'sglib:', 'lower_bound_inf');
    # assert_error(@()(fix_bounds(N, 2, 4, q0=0, q1=1)), 'sglib:', 'upper_bound_inf');

    ## Orthogonal polynomials
    dist = NormalDistribution()
    polysys = dist.orth_polysys()
    N = 5
    assert_equals(compute_gramian(polysys, dist, N), np.diag([polysys.sqnorm(n) for n in range(N + 1)]), 'orth', abstol=1e-9)

    # dist = NormalDistribution(2,3);
    # assert_error(@()(dist.orth_polysys()), 'sglib:', 'no_standard_dist');

    ## Base dist stuff
    dist = NormalDistribution(2, 3)
    base = dist.get_base_dist()

    #assert_equals( base, NormalDistribution(0,1), 'base' )

    z = np.linspace(0,1)
    x1 = dist.invcdf(z)
    x2 = base.invcdf(z)
    assert_equals( dist.base2dist(x2), x1, 'base2dist', abstol=1e-8 )
    assert_equals( dist.dist2base(x1), x2, 'dist2base', abstol=1e-8 )
   
 
################################################### UniformDistribution ##################################################  
  
def unittest_UniformDistribution():
    ##Initialization
    U = UniformDistribution(2,4)
    assert_equals( U.a, 2, 'Initialization a' )
    assert_equals( U.b, 4, 'Initialization b' )
    assert_equals( U.__repr__(), 'U(2, 4)', '__repr__' )
    U = UniformDistribution(-1)
    assert_equals( U.a, -1, 'Initialization default a' )
    assert_equals( U.b, 1, 'Initialization default b' )
    U = UniformDistribution()
    assert_equals( U.a, 0, 'Initialization a' )
    assert_equals( U.b, 1, 'Initialization b' )
    
    ## Mean & Var
    assert_equals( U.mean(), 0.5, 'mean')
    assert_equals( U.var(), 0.08333333, 'var' )
    
    ## Moments
    m_act = [1, 0, 0, 0, 0]
    m_act[1:] = U.moments()
    m_ex = compute_moments(U)
    assert_equals( m_act, m_ex, 'moments' )
    
    ## uniform_cdf
    U = UniformDistribution(2,4)
    assert_equals( U.cdf(1.9), 0., 'cdf_smaller' )
    assert_equals( U.cdf(2.5), 1/4, 'cdf_inside' )
    assert_equals( U.cdf(5), 1., 'cdf_larger' )
    assert_equals( U.cdf((U.a+U.b)/2), 1/2, 'cdf_median' )
    
    ## uniform_pdf
    U = UniformDistribution(2,4)
    assert_equals( U.pdf(-math.inf), 0., 'pdf_minf' )
    assert_equals( U.pdf(3.5), 1/2, 'pdf_inside' )
    assert_equals( U.pdf(math.inf), 0., 'pdf_inf' )
    
    # pdf matches cdf
    x1 = np.linspace( -0.1,5,100 )
    x2 = (x1[:-1]+x1[1:])/2
    F = U.cdf(x1)
    F2 = pdf_integrate( U.pdf(x2), F, x1)
    assert_equals( F, F2, 'pdf_cdf_match', abstol=0.01 )
    
    ## uniform_invcdf
    y = np.linspace(0, 1)
    U = UniformDistribution()
    x = np.linspace(0, 1)
    assert_equals( U.cdf(U.invcdf(y)), y, 'cdf_invcdf_1' )
    assert_equals( U.invcdf(U.cdf(x)), x, 'invcdf_cdf_1' )
    assert_equals( [np.isnan(U.invcdf([-0.1, 1.1]))], [True, True], 'invcdf_nan1' )
    U = UniformDistribution(0.5)
    x = np.linspace(0.5, 1)
    assert_equals( U.cdf(U.invcdf(y)), y, 'cdf_invcdf_2' )
    assert_equals( U.invcdf(U.cdf(x)), x, 'invcdf_cdf_2' )
    assert_equals( [np.isnan(U.invcdf([-0.1, 1.1]))], [True, True], 'invcdf_nan2' )
    U = UniformDistribution(-2,3)
    x = np.linspace(-2, 3)
    assert_equals( U.cdf(U.invcdf(y)), y, 'cdf_invcdf_3' )
    assert_equals( U.invcdf(U.cdf(x)), x, 'invcdf_cdf_3' )
    assert_equals( [np.isnan(U.invcdf([-0.1, 1.1]))], [True, True], 'invcdf_nan3' )
    
    ## Sample
    #munit_control_rand('seed', 1234);
    U = UniformDistribution(-2,3)
    N = 100000
    xi = U.sample(N)
    xi.sort()
    assert_equals( U.cdf(xi), linspace_midpoints(0,1,N), 'sample_cdf', abstol=1e-2 )
                  
    ## uniform_stdnor
    N = 50
    uni = np.linspace(0,1,N+2)
    uni = uni[1:-1]
    gam = np.sqrt(2)*sc.erfinv(2*uni-1)
    U = UniformDistribution(0.2, 1.3)
    x = U.stdnor2base(gam)
    assert_equals( U.cdf(x), uni, 'uniform' )
    U = UniformDistribution(0,1)
    assert_equals( uniform_stdnor(gam), U.stdnor2base(gam), 'uniform_def12' )
    assert_equals( uniform_stdnor(gam, 0), U.stdnor2base(gam), 'uniform_def2' )
    
    ## translate
    U = UniformDistribution(2,3)
    #T=TranslatedDistribution(U,2,3);
    tU = U.translate(2,3)
    [m, v, s, k] = tU.moments()
    #[m2,v2]=T.moments();
    assert_equals( m, 4.5, 'translated mean' )
    assert_equals( v, 0.75, 'translated var' )
    assert_equals( tU.pdf(0),0.,'translated pdf' )
    assert_equals( tU.pdf(math.inf),0.,'translated pdf' )
    #T.pdf(0);
    assert_equals( tU.cdf(0),0.,'translated cdf',abstol=0.0001 )
    assert_equals( tU.cdf(math.inf),1.,'translated cdf' )
    assert_equals( tU.cdf(-math.inf),0.,'translated cdf' )
    assert_equals( tU.invcdf(0),3.,'translated cdf' )
    
    ## fix_moments
    U = UniformDistribution(22,88)
    dist = U.fix_moments( 50, 3 )
    [m, v, s, k] = dist.moments()
    assert_equals( m, 50., 'mean fix_moments' )
    assert_equals( v, 3., 'uvar fix_moments' )
    
    ## Fix Bounds
    U = UniformDistribution(2,3)
    dist = U.fix_bounds(2,4)
    assert_equals( dist.invcdf(0), 2., 'fix_bounds-uni_min' )
    assert_equals( dist.invcdf(1), 4., 'fix_bounds-uni_max' )
    
    ## Orthogonal polynomials
    dist = UniformDistribution(-1, 1)
    polysys = dist.orth_polysys()
    N = 5
    assert_equals( compute_gramian(polysys, dist, N), np.diag([polysys.sqnorm(n) for n in range(N + 1)]), 'orth' )
    dist = UniformDistribution(2,3)
    # assert_error(@()(dist.orth_polysys()), 'sglib:', 'no_standard_dist');
    
    ## Base dist stuff
    dist = UniformDistribution(2, 5)
    base = dist.get_base_dist()
    #assert_equals( base, UniformDistribution(-1,1), 'base' )
    z = np.linspace(0,1)
    x1 = dist.invcdf(z)
    x2 = base.invcdf(z)
    assert_equals( dist.base2dist(x2), x1, 'base2dist' )
    assert_equals( dist.dist2base(x1), x2, 'dist2base' )

    
################################################### LogNormalDistribution ##################################################  

def unittest_LogNormalDistribution():
    ## Initialization
    LN = LogNormalDistribution(2,3)
    assert_equals( LN.mu, 2, 'Initialization default a' )
    assert_equals( LN.sigma,3, 'Initialization default b' )
    assert_equals( LN.__repr__(), 'lnN(2, 9)', '__repr__' )

    LN = LogNormalDistribution(2)
    assert_equals( LN.mu, 2, 'Initialization default a' )
    assert_equals( LN.sigma, 1, 'Initialization default b' )

    LN = LogNormalDistribution()
    assert_equals( LN.mu, 0, 'Initialization default a' )
    assert_equals( LN.sigma, 1, 'Initialization default b' )

    ## Mean & Var
    assert_equals( LN.mean(), np.exp(1/2), 'mean' )
    assert_equals( LN.var(), np.exp(2)-np.exp(1), 'var' )

    ## Moments
    m_act = [1, 0, 0, 0, 0]
    m_act[1:] = LN.moments()
    m_ex = compute_moments(LN)
    assert_equals( m_act, m_ex, 'moments', abstol=1e-5 )

    ## Cdf
    LN = LogNormalDistribution(2,0.5)
    assert_equals( LN.cdf(-math.inf), 0., 'cdf_minf' )
    assert_equals( LN.cdf(-1e8), 0., 'cdf_negative' )
    assert_equals( LN.cdf(math.inf), 1., 'cdf_inf' )
    assert_equals( LN.cdf(np.exp(LN.mu)), 1/2, 'cdf_median' )

    ## Pdf
    assert_equals( LN.pdf(-math.inf), 0., 'pdf_minf' )
    assert_equals( LN.pdf(-1e8), 0., 'pdf_negative' )
    assert_equals( LN.pdf(math.inf), 0., 'pdf_inf' )

    ## Invcdf
    y = np.linspace(0, 1, 100)
    x = np.linspace(0, 10, 100)

    LN = LogNormalDistribution()
    assert_equals( LN.cdf(LN.invcdf(y)), y, 'cdf_invcdf_1' )
    assert_equals( LN.invcdf(LN.cdf(x)), x, 'invcdf_cdf_1' )
    assert_equals( [np.isnan(LN.invcdf([-0.1, 1.1]))], [True, True], 'invcdf_nan1' )

    LN = LogNormalDistribution(0,0.5)
    assert_equals( LN.cdf(LN.invcdf(y)), y, 'cdf_invcdf_2')
    assert_equals( LN.invcdf(LN.cdf(x)), x, 'invcdf_cdf_2')
    assert_equals( [np.isnan(LN.invcdf([-0.1, 1.1]))], [True, True], 'invcdf_nan2')

    LN = LogNormalDistribution(0.7,1.5)
    assert_equals( LN.cdf(LN.invcdf(y)), LN.cdf(LN.invcdf(y)),'cdf_invcdf_3')
    assert_equals( LN.invcdf(LN.cdf(x)), x, 'invcdf_cdf_3')
    assert_equals( [np.isnan(LN.invcdf([-0.1, 1.1]))], [True, True], 'invcdf_nan3')

    ## Sample
    #munit_control_rand('seed', 1234);
    LN = LogNormalDistribution(0.7,1.5)
    N = 100000
    xi = LN.sample(N)
    xi.sort()
    assert_equals( LN.cdf(xi), linspace_midpoints(0,1,N), 'sample_cdf', abstol=3e-3)

    ## Stdnor
    N = 50
    uni = np.linspace(0,1,N+2)
    uni = uni[1:-1]
    gam = np.sqrt(2)*sc.erfinv(2*uni-1)

    LN = LogNormalDistribution(0.2,0.3)
    x = LN.stdnor2base( gam )
    assert_equals( LN.cdf(x), uni, 'lognormal' )
    assert_equals( lognormal_stdnor(gam), lognormal_stdnor(gam, 0, 1), 'lognormal_def12' )
    assert_equals( lognormal_stdnor(gam, 0), lognormal_stdnor(gam, 0, 1), 'lognormal_def2' )

    ## fix_moments
    LN = LogNormalDistribution(2,3)
    dist = LN.fix_moments(4,14)
    [m, v, s, k] = dist.moments()
    assert_equals( m, 4., 'mean fix_moments' )
    assert_equals( v, 14., 'var fix_moments' )

    ## Fix Bounds
    LN = LogNormalDistribution(2,3)
    dist = LN.fix_bounds( 2, 4, q0=0.001, q1=0.5 )
    assert_equals( dist.invcdf(0.001), 2., 'fix_bounds-nor_min' )
    assert_equals( dist.invcdf(0.5), 4., 'fix_bounds-nor_max' )

    # ## Orthogonal polynomials
    # dist = LogNormalDistribution(0,1);
    # assert_error(@()(dist.orth_polysys()), 'sglib:', 'no_orth_polys');

    ## Base dist stuff
    dist = LogNormalDistribution(0.3, 1.2)
    base = dist.get_base_dist()

    #assert_equals(base, NormalDistribution(0,1), 'base');

    z = np.linspace(0,1,100)
    x1 = dist.invcdf(z)
    x2 = base.invcdf(z)
    assert_equals( dist.base2dist(x2), x1, 'base2dist' )
    assert_equals( dist.dist2base(x1), x2, 'dist2base' )
    

################################################### UniformDistribution ##################################################  

def unittest_BetaDistribution():
    ## Initialization
    B = BetaDistribution(2,3)
    assert_equals( B.__repr__(), 'Beta(2, 3)', '__repr__' )
    assert_equals( B.a, 2, 'Initialization a' )
    assert_equals( B.b, 3, 'Initialization b' )

    ## Mean & Var
    assert_equals( B.mean(), -0.2, 'mean' )
    assert_equals( B.var(), 0.16, 'var' )

    ## beta_cdf
    B = BetaDistribution(2,3)
    assert_equals( B.cdf(-math.inf), 0., 'cdf_minf' )
    assert_equals( B.cdf(-1-1e8), 0., 'cdf_zero' )
    assert_equals( B.cdf( 1+1e8), 1., 'cdf_zero' )
    assert_equals( B.cdf(math.inf), 1., 'cdf_inf' )
    
    B = BetaDistribution(2,2)
    assert_equals( B.cdf(0), 1/2, 'cdf_median' )
    
    B = BetaDistribution(3,3)
    assert_equals( B.cdf(0), 1/2, 'cdf_median' )
    
    B = BetaDistribution(1/3,1/3)
    assert_equals( B.cdf(0), 1/2, 'cdf_median' )

    ## beta_pdf
    B = BetaDistribution(2,3)
    assert_equals( B.pdf(-math.inf), 0., 'pdf_minf' )
    assert_equals( B.pdf(-1-1e8), 0., 'pdf_zero' )
    assert_equals( B.pdf(-1), 0., 'pdf_zero' )
    assert_equals( B.pdf(1), 0., 'pdf_zero' )
    assert_equals( B.pdf(1+1e8), 0., 'pdf_zero' )
    assert_equals( B.pdf(math.inf), 0., 'pdf_inf' )

    B = BetaDistribution(0.2,0.5)
    assert_equals( B.pdf(-1-1e-10), 0., 'pdf_zero' )
    assert_equals( B.pdf(1+1e-10), 0., 'pdf_zero' )

    ## beta_invcdf
    y = np.linspace(0, 1, 100)
    x = np.linspace(-1, 1, 100)

    B = BetaDistribution(2,3)
    assert_equals( B.cdf(B.invcdf(y)), y, 'cdf_invcdf_1' )
    assert_equals( B.invcdf(B.cdf(x)), x, 'invcdf_cdf_1' , abstol=1e-7)
    assert_equals( [np.isnan(B.invcdf([-0.1, 1.1]))], [True, True], 'invcdf_nan1' )

    B = BetaDistribution(0.5,0.5)
    assert_equals( B.cdf(B.invcdf(y)), y, 'cdf_invcdf_2' )
    assert_equals( B.invcdf(B.cdf(x)), x, 'invcdf_cdf_2' )
    assert_equals( [np.isnan(B.invcdf([-0.1, 1.1]))], [True, True], 'invcdf_nan2' )

    B = BetaDistribution(1,1)
    assert_equals( B.cdf(B.invcdf(y)), y, 'cdf_invcdf_3' )
    assert_equals( B.invcdf(B.cdf(x)), x, 'invcdf_cdf_3' )
    assert_equals( [np.isnan(B.invcdf([-0.1, 1.1]))], [True, True], 'invcdf_nan3' )

    ## Sample
    B = BetaDistribution(2,3)
    N = 100000
    xi = B.sample(N)
    xi.sort()
    assert_equals( B.cdf(xi), linspace_midpoints(0,1,N), 'sample_cdf', abstol=1e-2 )

    ## fix_moments
    B = BetaDistribution(2,3)
    dist = B.fix_moments(3,14)
    [m, v, s, k] = dist.moments()
    assert_equals( m, 3., 'mean fix_moments' )
    assert_equals( v, 14., 'var fix_moments' )

    ## Fix Bounds
    B = BetaDistribution(2,3)
    dist = B.fix_bounds(4,5)
    assert_equals( dist.invcdf(0), 4., 'fix_bounds-uni_min' )
    assert_equals( dist.invcdf(1), 5., 'fix_bounds-uni_max' )

    ## Base dist stuff
    dist = BetaDistribution(2, 3)
    base = dist.get_base_dist()

    # assert_equals(base, BetaDistribution(2,3), 'base_is_same');

    z = np.linspace(0,1,100)
    x1 = dist.invcdf(z)
    x2 = base.invcdf(z)
    assert_equals( dist.base2dist(x2), x1, 'base2dist' )
    assert_equals( dist.dist2base(x1), x2, 'dist2base' )

    ## Ortho polys
    dist = BetaDistribution(1.5, 0.5)
    polysys = dist.orth_polysys()
    N = 5
    #assert_equals(compute_gramian(polysys, dist, N), diag(polysys.sqnorm(0:N)), 'orth');
    

################################################### ExponentialDistribution ##################################################  
    
def unittest_ExponentialDistribution():
    ## Mean & Var
    E = ExponentialDistribution(1.5)
    assert_equals( E.__repr__(), 'Exp(1.5)', '__repr__' )
    assert_equals( E.mean(), 0.6666666, 'mean', abstol=0.0001 )
    assert_equals( E.var(), 0.44444444, 'var' )

    ## Moments
    m_act = [1, 0, 0, 0, 0]
    m_act[1:] = E.moments()
    m_ex = compute_moments(E)
    assert_equals( m_act, m_ex, 'moments' )

    ## exponential_cdf
    assert_equals( E.cdf(-math.inf), 0., 'cdf_minf' )
    assert_equals( E.cdf(-1e10), 0., 'cdf_negative' )
    assert_equals( E.cdf(math.inf), 1., 'cdf_inf' )
    assert_equals( E.cdf(np.log(2)/E.lambda_), 1/2, 'cdf_median' )

    ## exponential_pdf
    assert_equals( E.pdf(-math.inf), 0., 'pdf_minf' )
    assert_equals( E.pdf(-1e10), 0., 'pdf_negative' )
    assert_equals( E.pdf(math.inf), 0., 'pdf_inf' )

    ## exponential_invcdf
    y = np.linspace(0, 1, 100)
    E = ExponentialDistribution(2)
    assert_equals( E.cdf(E.invcdf(y)), y, 'cdf_invcdf_1' )
    assert_equals( E.invcdf(E.cdf(y)), y, 'invcdf_cdf_1' )
    assert_equals( [np.isnan(E.invcdf([-0.1, 1.1]))], [True, True], 'invcdf_nan1' )

    E = ExponentialDistribution(0.5)
    assert_equals( E.cdf(E.invcdf(y)), y, 'cdf_invcdf_1')
    assert_equals( E.invcdf(E.cdf(y)), y, 'invcdf_cdf_1')
    assert_equals( [np.isnan(E.invcdf([-0.1, 1.1]))], [True, True], 'invcdf_nan1')

    ## Sample
    E = ExponentialDistribution(0.5)
    N = 100000
    xi = E.sample(N)
    xi = np.sort(xi)
    assert_equals( E.cdf(xi), linspace_midpoints(0,1,N), 'sample_cdf', abstol=1e-2 )

    ## exponential_stdnor
    N = 50
    uni = np.linspace(0,1,N+2)
    uni = uni[1:-1]
    gam = np.sqrt(2)*sc.erfinv(2*uni-1)

    E = ExponentialDistribution(0.7)
    x = E.stdnor2base( gam )
    assert_equals( E.cdf(x), uni, 'exponential' )

    ## fix_moments
    E = ExponentialDistribution(4)
    dist = E.fix_moments(5,7)
    [m, v, s, k] = dist.moments()
    assert_equals( m,5.,'mean fix_moments' )
    assert_equals( v,7.,'var fix_moments' )

    ## Fix Bounds
    E = ExponentialDistribution(4)
    dist = E.fix_bounds(2,4,q0=0.001,q1=0.5)
    assert_equals( dist.invcdf(0.001), 2., 'fix_bounds-nor_min' )
    assert_equals( dist.invcdf(0.5), 4., 'fix_bounds-nor_max' )

    ## Orthogonal polynomials
    dist = ExponentialDistribution(1)
    polysys = dist.orth_polysys()
    N = 5
    # assert_equals(compute_gramian(polysys, dist, N), diag(polysys.sqnorm(0:N)), 'orth');

    # dist = ExponentialDistribution(1.4);
    # assert_error(@()(dist.orth_polysys()), 'sglib:', 'no_standard_dist');

    ## Base dist stuff
    dist = ExponentialDistribution(4.3)
    base = dist.get_base_dist()

    # assert_equals(base, ExponentialDistribution(1), 'exp(1)');

    z = np.linspace(0,1,100)
    x1 = dist.invcdf(z)
    x2 = base.invcdf(z)
    assert_equals( dist.base2dist(x2), x1, 'base2dist' )
    assert_equals( dist.dist2base(x1), x2, 'dist2base' )

    pass

    
if __name__ == "__main__":
    print('Unit testing TranslatedDistribution')
    unittest_TranslatedDistribution()
    print('-------------------------------')
    print('Unit testing NormalDistribution')
    unittest_NormalDistribution()
    print('-------------------------------')
    print('Unit testing UniformDistribution')
    unittest_UniformDistribution()
    print('-------------------------------')
    print('Unit testing LogNormalDistribution')
    unittest_LogNormalDistribution()
    print('-------------------------------')
    print('Unit testing BetaDistribution')
    unittest_BetaDistribution()
    print('-------------------------------')
    print('Unit testing ExponentialDistribution')
    unittest_ExponentialDistribution()
    print('-------------------------------')