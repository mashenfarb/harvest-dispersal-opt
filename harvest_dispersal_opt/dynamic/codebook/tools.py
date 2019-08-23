from typing import Tuple

import numpy as np
import sympy as sym
from scipy.optimize import fsolve


# def get_equilibrium_nocontrols_stock(C11,C12,p1,r11,r22,K22,K11,c2,b):
#     def equations(p):
#         x10, x20 = p
#         return(x10*(C11/c2) - (p1/c2)*(x10**2) + r11*x10*(1-(x10/K11)) + b*(x20/K22) - b*(x10/K11) - x10,
#                x20*(C12/c2) - (p1/c2)*(x20**2) + r22*x20*(1-(x20/K22)) - b*(x20/K22) + b*(x10/K11) - x20
#     x10, x20 = fsolve(equations,(1,1))
#     return x10, x20


def get_init_eql_stock(C11: float, C12: float, p1: float, r11: float,
        r22: float, K22: float, K11: float, c2: float,
        b: float) -> Tuple[float]:
    x = sym.symbols('x', positive=True)
    y = sym.symbols('y', positive=True)

    f = sym.Eq(x*(C11/c2) -
               (p1/c2)*(x**2) +
               r11*x*(1-(x/K11)) +
               b*(y/K22) -
               b*(x/K11))  # - x)??
    g = sym.Eq(y*(C12/c2) -
               (p1/c2)*(y**2) +
               r22*y*(1-(y/K22)) -
               b*(y/K22) +
               b*(x/K11))  # - y)??

    eqlNDCStocks = sym.nonlinsolve([f, g], [x, y])
    eqlNDCStocks = [x for x in eqlNDCStocks if x[0] > 0 and x[1] > 0]
    if len(eqlNDCStocks) == 1:
        X0 = eqlNDCStocks[0]
    else:
        sys.exit(('Not exactly one positive, real answer '
                  'to equilibrium stock condition equations.'))
    return X0[0], X0[1]


def get_discount_factor(dis: float, time: np.array) -> np.array:
    return (1/(1+dis))**time
