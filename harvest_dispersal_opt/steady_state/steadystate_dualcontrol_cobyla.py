'''
Note: In harvest steadystate, the two dispersal control control variables are separate.
      In harvest dynamic, the two dispersal control control variables are collapsed into one, equal term.
'''

import numpy as np
from scipy.optimize import minimize, NonlinearConstraint, LinearConstraint, BFGS
from pprint import pprint
from steadystate_objs import obj1
from steadystate_cons import const01_ineq_init, const01_eq, const01_eq_negs, const01_eq_jac, const01_eq_negs_jac


def get_steadystate_dualcontrol_init(dis, r11, r22, K11, K22, BB12, BB21,
                                     P, C11, C12, c2, CB, umax):
    '''Used as initial guess for fully specified problem'''
    cons = [{'type': 'ineq',
            'fun': const01_ineq_init,
            'args': (r11, r22, K11, K22,
                    BB12, BB21)}]
    bounds= [(0, K11), (0, K22), (0, umax), (0, umax), ]
    for i in range(len(bounds)):
        lb, ub = bounds[i]
        l = {'type': 'ineq',
             'fun': lambda x: x[i] - lb}
        u = {'type': 'ineq',
             'fun': lambda x: ub - x[i]}
        cons.append(l)
        cons.append(u)

    x01 = np.array([1, 1, .1, .1]).T
    OptimizeResult01_init = minimize(fun= obj1,
                                     x0= x01,
                                     args= (r11, r22, K11, K22, BB12, BB21,
                                            P, c2, C11, C12, CB),
                                     method='COBYLA',
                                     constraints=cons,
                                     tol=10e-14,
                                     options={'disp':True,
                                              'catol':10e-12,
                                              'maxiter':100000}
                                     )
    print(OptimizeResult01_init.status)
    print(OptimizeResult01_init.message)
    print(OptimizeResult01_init.fun)
    print(*OptimizeResult01_init.x, sep='\n ')
    return OptimizeResult01_init.x


def get_steadystate_dualcontrol(dis, r11, r22, K11, K22, BB12, BB21,
                                P, C11, C12, c2, CB, umax):
    x01 = get_steadystate_dualcontrol_init(dis, r11, r22, K11, K22,
                                           BB12, BB21, P, C11, C12,
                                           c2, CB, umax)
    cons = [{'type':'ineq',
             'fun':const01_eq,
             'args':(dis, umax, r11, r22,
                      K11, K22, BB12, BB21,
                      P, c2, C11, C12, CB),
              'jac':const01_eq_jac,
              },
            {'type':'ineq',
             'fun':const01_eq_negs,
             'args':(dis, umax, r11, r22,
                     K11, K22, BB12, BB21,
                     P, c2, C11, C12, CB),
              'jac':const01_eq_negs_jac,
              }]
    bounds= [(0, K11), (0, K22),
             (0, umax), (0, umax),
             (0, np.inf), (0, np.inf), (0, np.inf),
             (0, np.inf), (0, np.inf), (0, np.inf),
             (0, np.inf), (0, np.inf),]
    for i in range(len(bounds)):
        lb, ub = bounds[i]
        l = {'type': 'ineq',
             'fun': lambda x: x[i] - lb}
        cons.append(l)
        if ub != np.inf:
            u = {'type': 'ineq',
                 'fun': lambda x: ub - x[i]}
            cons.append(u)
    OptimizeResult01 = minimize(fun= obj1,
                                x0= np.append(x01,
                                              np.array([.1, .1, 0, 0,
                                                        0, 0, 0, 0]).T,
                                              axis = 0),
                                args= (r11, r22, K11, K22, BB12, BB21,
                                       P, c2, C11, C12, CB),
                                method='COBYLA',
                                constraints= cons,
                                tol=10e-14,
                                options={'disp':True,
                                         'catol':10e-12,
                                         'maxiter':100000},
                                )
    X1T = OptimizeResult01.x[0]
    X2T = OptimizeResult01.x[1]
    print(OptimizeResult01.status)
    print(OptimizeResult01.message)
    print(OptimizeResult01.fun)
    print(*OptimizeResult01.x, sep='\n ')
    return X1T, X2T


if __name__ == '__main__':
    '''Testing Parameters'''
    '''Economic'''
    dis = 0.05  # Discount factor
    P = 1       # price
    C11 = .6    # cost coefficient c1 patch 1
    C12 = .25   # cost coefficient c1 patch 2
    c2 = .1     # quadratic cost coefficient
    '''Ecological'''
    r11 = 1     # patch 1 growth rate
    r22 = 1     # patch 2 growth rate
    BB12 = .5   # dispersal to 1 from 2
    BB21 = .5   # dispersal to 2 from 1
    CB = .1    # dispersal control cost
    K11 = 1     # carrying capacity
    K22 = 1     # carrying capacity
    umax = 2    # maximum dispersal control

    X1T, X2T = get_steadystate_dualcontrol(dis, r11, r22, K11, K22, BB12, BB21, P, C11, C12, c2, CB, umax)
