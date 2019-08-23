'''
Note: In harvest steadystate, the two dispersal control control variables are separate.
      In harvest dynamic, the two dispersal control control variables are collapsed into one, equal term.
'''

from pprint import pprint
from typing import Tuple

import numpy as np
from scipy.optimize import minimize, NonlinearConstraint, BFGS

from harvest_dispersal_opt.steady_state.codebook.steadystate_objs import obj1
from harvest_dispersal_opt.steady_state.codebook.steadystate_cons import (
    const01_ineq_init, const01_eq, const01_eq_jac, get_Nonlinear_Constraints)


## Testing Parameters
## Economic
dis = 0.05  # Discount factor
P = 1       # price
C11 = .6    # cost coefficient c1 patch 1
C12 = .25   # cost coefficient c1 patch 2
c2 = .1     # quadratic cost coefficient
## Ecological
r11 = 1     # patch 1 growth rate
r22 = 1     # patch 2 growth rate
BB12 = .5   # dispersal to 1 from 2
BB21 = .5   # dispersal to 2 from 1
CB = .01    # dispersal control cost
K11 = 1     # carrying capacity
K22 = 1     # carrying capacity
umax = 2    # maximum dispersal control


def get_steadystate_dualcontrol(dis: float, r11: float, r22: float, K11: float,
        K22: float, BB12: float, BB21: float, P: float, C11: float, C12: float,
        c2: float, CB: float, umax: float) -> Tuple[float]:
    x01 = _get_steadystate_dualcontrol_init(dis, r11, r22, K11, K22, BB12,
                                            BB21, P, C11, C12, c2, CB, umax)

    cons = get_Nonlinear_Constraints(dis, umax, r11, r22, K11, K22, BB12, BB21,
                                     P, c2, C11, C12, CB)

    OptimizeResult01 = minimize(fun= obj1,
                                x0= np.append(x01,
                                              np.array([.1, .1, 0, 0,
                                                        0, 0, 0, 0]).T,
                                              axis = 0),
                                args= (r11, r22, K11, K22, BB12, BB21,
                                       P, c2, C11, C12, CB),
                                method='trust-constr',
                                jac='cs',
                                hess=BFGS(),
                                constraints= cons,
                                bounds= [(0, K11), (0, K22),
                                         (0, umax), (0, umax),
                                         (0, None), (0, None), (0, None),
                                         (0, None), (0, None), (0, None),
                                         (0, None), (0, None),],
                                options={'xtol':10e-12,
                                         'gtol':10e-12,
                                         'disp':True},
                                )
    X1T = OptimizeResult01.x[0]
    X2T = OptimizeResult01.x[1]
    return X1T, X2T


def _get_steadystate_dualcontrol_init(dis: float, r11: float, r22: float,
        K11: float, K22: float, BB12: float, BB21: float, P: float, C11: float,
        C12: float, c2: float, CB: float, umax: float) -> np.array:
    ## Used as initial guess for fully specified problem
    x01 = np.array([1, 1, .1, .1]).T
    cons = NonlinearConstraint(fun=lambda x: const01_ineq_init(x, r11, r22, K11,
                                                               K22, BB12, BB21),
                               lb=0,
                               ub=np.inf,
                               )
    OptimizeResult01_init = minimize(fun= obj1,
                                     x0= x01,
                                     args= (r11, r22, K11, K22, BB12, BB21,
                                            P, c2, C11, C12, CB),
                                     method='trust-constr',
                                     jac='cs',
                                     hess=BFGS(),
                                     constraints=cons,
                                     bounds= [(0, K11),
                                              (0, K22),
                                              (0, umax),
                                              (0, umax),
                                              ],
                                     tol=10e-14,
                                     )
    print(OptimizeResult01_init.status)
    print(OptimizeResult01_init.message)
    print(OptimizeResult01_init.fun)
    print(OptimizeResult01_init.method)
    print(*OptimizeResult01_init.x, sep='\n ')
    return OptimizeResult01_init.x


if __name__ == '__main__':
    X1T, X2T = get_steadystate_dualcontrol(dis, r11, r22, K11, K22, BB12,
                                           BB21, P, C11, C12, c2, CB, umax)
    print(X1T, X2T)
