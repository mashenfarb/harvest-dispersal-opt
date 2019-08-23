'''
Note: In harvest steadystate, the two dispersal control control variables are separate.
      In harvest dynamic, the two dispersal control control variables are collapsed into one, equal term.
'''
from typing import Tuple

import numpy as np
from scipy.optimize import minimize

from harvest_dispersal_opt.steady_state.codebook.steadystate_objs import obj2
from harvest_dispersal_opt.steady_state.codebook.steadystate_cons import (
    const02_ineq_init, const02_eq, const02_ineq)


## Testing Parameters
## Economic
dis = 0.05  # Discount factor
P = 1       # price
C11 = .5    # cost coefficient c1 patch 1
C12 = .25   # cost coefficient c1 patch 2
c2 = .1     # quadratic cost coefficient
## Ecological
r11 = 1     # patch 1 growth rate
r22 = 1     # patch 2 growth rate
BB12 = .5   # dispersal to 1 from 2
BB21 = .5   # dispersal to 2 from 1
CB = .025   # dispersal control cost
K11 = 1     # carrying capacity
K22 = 1     # carrying capacity
umax = 2    # maximum dispersal control


def get_steadystate_eqdispersal(dis: float, r11: float, r22: float, K11: float,
        K22: float, BB12: float, BB21: float, P: float, C11: float, C12: float,
        c2: float, CB: float, umax: float) -> Tuple[float]:
    x02_init = _get_steadystate_eqdispersal_init()

    OptimizeResult02 = minimize(fun= obj2,
                                x0= np.append(x02_init,
                                              np.array([1, 1, 0, 0, 0, 0]).T,
                                              axis=0
                                              ),
                                args=(r11, r22, K11, K22, BB12, BB21,
                                      P, c2, C11, C12, CB),
                                constraints=[
                                    {'type':'eq',
                                     'fun':const02_eq,
                                     'args':(dis, r11, r22, K11, K22,
                                             BB12, BB21, P, c2, C11,
                                             C12, CB)
                                    },
                                    {'type':'ineq',
                                    'fun':const02_ineq,
                                    'args':(umax, )
                                     }
                                ],
                                bounds=[(0, K11),
                                        (0, K22),
                                        (0, umax),
                                        (0, np.inf),
                                        (0, np.inf),
                                        (0, np.inf),
                                        (0, np.inf),
                                        (0, np.inf),
                                        (0, np.inf),
                                        ]
                                )
    X1T = OptimizeResult02.x[0]
    X2T = OptimizeResult02.x[1]
    print(OptimizeResult02.success)
    print(OptimizeResult02.message)
    return X1T, X2T


def _get_steadystate_eqdispersal_init():
    x02 = np.array([1, .5, .3]).T
    OptimizeResult02_init = minimize(fun= obj2,
                                     x0= x02,
                                     args= (r11, r22, K11, K22, BB12, BB21,
                                            P, c2, C11, C12, CB),
                                     constraints= [{'type':'ineq',
                                                    'fun':const02_ineq_init,
                                                    'args':(r11, r22, K11, K22,
                                                            BB12, BB21)},
                                                    ],
                                     bounds= [(0, K11),
                                              (0, K22),
                                              (0, umax),
                                              ]
                                     )
    return OptimizeResult02_init.x


if __name__ == '__main__':
    X1T, X2T = get_steadystate_eqdispersal(dis, r11, r22, K11, K22, BB12, BB21,
                                           P, C11, C12, c2, CB, umax)
    print(X1T, X2T)
