from polysys import *

def syschar_to_polysys(syschar):
    poly_dict = {'H': HermitePolynomials,
                 'h': HermitePolynomials.normalized(),
                 'P': LegendrePolynomials,
                 'p': LegendrePolynomials.normalized(),
                 'T': ChebyshevTPolynomials,
                 't': ChebyshevTPolynomials.normalized(),
                 'U': ChebyshevUPolynomials,
                 'u': ChebyshevUPolynomials.normalized(),
                 'L': LaguerrePolynomials,
                 'l': LaguerrePolynomials.normalized(),
                 'M': Monomials}
    return poly_dict[syschar]