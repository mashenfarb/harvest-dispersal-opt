import numpy as np
from scipy.optimize import minimize

from harvest_dispersal_opt.steady_state.codebook.steadystate_objs import obj3
from harvest_dispersal_opt.steady_state.codebook.steadystate_cons import (
    const03_eq)


## Testing Parameters
## Economic
dis = 0.05   # Discount factor
P = 1        # price
C11 = .8     # cost coefficient
C12 = .1     # patch 2 is HIGH cost patch, patch 1 low cost
c2 = .1      # cost coefficient on second term
## Ecological
R = 1        # patch 1 growth rate
B = .5       # patch 1, 2 uncontrolled dispersal rate
K = 1        # carrying capacity
umax = 2     # maximum dispersal control


def get_steadystate_ndc(dis: float, R: float, K: float, B: float, P: float,
        C11: float, C12: float, c2: float):
    x03 = np.transpose(np.array([.9, .5]))

    OptimizeResult03 = minimize(fun= obj3,
                                x0= np.append(x03,
                                              np.array([1,1,.1,.1]).T,
                                              axis = 0), # last 2 elements changed from 0 0 (1850 successes)
                                args= (R,K,B,P,C11,C12,c2),
                                constraints= [{'type':'eq',
                                               'fun':const03_eq,
                                               'args':(dis,R,K,B,P,C11,C12,c2)},
                                               ],
                                bounds= [(0, K),
                                         (0, K),
                                         (0, np.inf),
                                         (0, np.inf),
                                         (0, np.inf),
                                         (0, np.inf),
                                         ]
                                )

    X1T = OptimizeResult03.x[0]
    X2T = OptimizeResult03.x[1]
    return X1T, X2T


if __name__ == '__main__':
    X1T, X2T = get_steadystate_ndc(dis, R, K, B, P, C11, C12, c2)
    print(X1T, X2T)
