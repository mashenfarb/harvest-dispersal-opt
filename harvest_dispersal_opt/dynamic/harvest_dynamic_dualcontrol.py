'''
Note: In harvest steadystate, the two dispersal control control variables are separate.
      In harvest dynamic, the two dispersal control control variables are collapsed into one, equal term.
'''


import sys
import numpy as np
from opty import direct_collocation
import sympy as sym
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
from steadystate_dualcontrol_trustconstr import get_steadystate_dualcontrol
from plot_dual_control import plot_dual_control
from pprint import pprint


def get_init_equilibrium_stock(C11, C12, p1, r11, r22, K11, K22, c2, b):
    x = sym.symbols('x', positive=True)
    y = sym.symbols('y', positive=True)

    f = sym.Eq(x*(C11/c2) -
               (p1/c2)*(x**2) +
               r11*x*(1-(x/K11)) +
               b*(y/K22) -
               b*(x/K11))
    g = sym.Eq(y*(C12/c2) -
               (p1/c2)*(y**2) +
               r22*y*(1-(y/K22)) -
               b*(y/K22) +
               b*(x/K11))

    eqlNDCStocks = sym.nonlinsolve([f, g], [x, y])
    eqlNDCStocks = [x for x in eqlNDCStocks if x[0] > 0 and x[1] > 0]
    if len(eqlNDCStocks) == 1:
        X0 = eqlNDCStocks[0]
    else:
        sys.exit('Not exactly one positive, real answer to equilibrium stock condition equations.')
    return X0[0], X0[1]


def get_discount_factor(dis, time):
    discntFactr = (1/(1+dis))**time  # Define list of discount factors
    return discntFactr


def define_eom(x1, x2, u1, u2, u3, u4):
    eom = sym.Matrix([(r11*x1(t)*(1 - x1(t)/K11)
                       - u3(t)*BB21*(x1(t)/K11)
                       + u4(t)*BB12*(x2(t)/K22)
                       - u1(t) - x1(t).diff()),

                      (r22*x2(t)*(1 - x2(t)/K22)
                       + u3(t)*BB21*(x1(t)/K11)
                       - u4(t)*BB12*(x2(t)/K22)
                       - u2(t) - x2(t).diff())
                      ])
    return eom


def define_bounds(x1, x2, u1, u2, u3, u4):
    bounds = {x1(t): (0.0, K11),
              x2(t): (0.0, K22),
              u1(t): (0.0, float('inf')),
              u2(t): (0.0, float('inf')),
              u3(t): (0.0, umax),
              u4(t): (0.0, umax)}
    return bounds


def define_instance_constrains(x1, x2, X10, X20, X1T, X2T):
    instance_constraints = (x1(0.0) - X10,
                            x2(0.0) - X20,
                            x1(duration) - X1T,
                            x2(duration) - X2T,
                            )
    return instance_constraints


def obj_ndc(free):
    x1 = free[0:numNodes]
    x2 = free[numNodes:2*numNodes]
    u1 = free[2*numNodes:3*numNodes]
    u2 = free[3*numNodes:4*numNodes]
    u3 = free[4*numNodes:5*numNodes]
    u4 = free[5*numNodes:6*numNodes]
    obj = (discntFactr*(u1*(p1 - (C11/x1) - u1*(c2/(x1)**2)) +
                       u2*(p1 - (C12/x2) - u2*(c2/(x2)**2))
                       - CB*(1 - u3)**2 - CB*(1 - u4)**2))
    return - intervalValue * np.sum(obj)


def obj_grad_ndc(free):
    x1 = free[0:numNodes]
    x2 = free[numNodes:2*numNodes]
    u1 = free[2*numNodes:3*numNodes]
    u2 = free[3*numNodes:4*numNodes]
    u3 = free[4*numNodes:5*numNodes]
    u4 = free[5*numNodes:6*numNodes]

    grad = np.zeros_like(free)
    grad[0:numNodes] = discntFactr*u1*(C11/x1**2 + 2*c2*u1/x1**3)  # gradient for x1
    grad[numNodes:2*numNodes] = discntFactr*u2*(C12/x2**2 + 2*c2*u2/x2**3)  # gradient for x2
    grad[2*numNodes:3*numNodes] = discntFactr*(-C11/x1 - 2*c2*u1/x1**2 + p1)  # gradient for u1
    grad[3*numNodes:4*numNodes] = discntFactr*(-C12/x2 - 2*c2*u2/x2**2 + p1)  # gradient for u2
    grad[4*numNodes:5*numNodes] = -CB*discntFactr*(2*u3 - 2)  # gradient for u3
    grad[5*numNodes:6*numNodes] = -CB*discntFactr*(2*u4 - 2)  # gradient for u4
    return - intervalValue * grad


def define_initial_guess(num_free, numNodes, duration, time):
    initial_guess = np.array([0]*(num_free))
    initial_guess[0:numNodes] = time/duration  # Initial guess for x1
    initial_guess[numNodes:2*numNodes] = time/duration  # Initial guess for x2
    initial_guess[2*numNodes:3*numNodes] = .1575*time/duration  # Initial guess for u1
    initial_guess[3*numNodes:4*numNodes] = .1575*time/duration  # Initial guess for u2
    initial_guess[4*numNodes:5*numNodes] = .21575*time/duration  # Initial guess for u3
    initial_guess[5*numNodes:6*numNodes] = .21575*time/duration  # Initial guess for u4
    return initial_guess


if __name__ == '__main__':
    '''Note: Parameters must be global variables for obj and obj_grad'''
    '''Economic Parameters'''
    dis = 0.05  # Discount factor
    p1 = 1       # price
    C11 = .6    # .25/.55 np.arange(.1,.801,.01) # cost coefficient - short loop for testing.  formerly c11
    C12 = .25   # .1/.8 # patch 2 is HIGH cost patch, patch 1 low cost  # NOTE: This value is from the second definition of this variable
    c2 = .1     # cost coefficient on second term
    '''Ecological Parameters'''
    r11 = 1     # patch 1 growth rate
    r22 = 1     # patch 2 growth rates
    b = .5      # patch 1, 2 uncontrolled dispersal rate
    BB12 = .5   # dispersal bw patches.  Dispersal from patch 2
    BB21 = .5   # dispersal bw patches.  Dispersal from patch 1
    CB = .1    # dispersal control cost
    K11 = 1     # carrying capacity of patch 1
    K22 = 1     # carrying capacity of patch 2
    umax = 2    # maximum dispersal control
    '''Time Paramters'''
    duration = 35  # total time
    numNodes = 500   # Num collocation nodes during run
    time = np.linspace(0, duration, numNodes)
    intervalValue = duration/(numNodes - 1)  # time interval between collocation nodes
    discntFactr = get_discount_factor(dis, time)

    '''Steady State Open Access and Optimal'''
    X10, X20 = get_init_equilibrium_stock(C11, C12, p1, r11, r22, K11, K22, c2, b)
    X1T, X2T = get_steadystate_dualcontrol(dis, r11, r22, K11, K22, BB12, BB21, p1, C11, C12, c2, CB, umax)
    print('Start and End Points for Stock 1:')
    print(str(X10) + " ---> " + str(X1T))
    print('--------------------------------------------')
    print('Start and End Points for Stock 2:')
    print(str(X20) + " ---> " + str(X2T))


    '''Paramterize Dynamic Path Optimization'''
    t = sym.symbols('t')
    x1, x2, u1, u2, u3, u4 = sym.symbols('x1, x2, u1, u2, u3, u4', cls=sym.Function)
    eom = define_eom(x1, x2, u1, u2, u3, u4)
    state_symbols = (x1(t), x2(t))
    specified_symbols = (u1(t), u2(t), u3(t), u4(t))
    instanceCons = define_instance_constrains(x1, x2, X10, X20, X1T, X2T)
    bounds = define_bounds(x1, x2, u1, u2, u3, u4)

    '''Solve for Optimal Dynamic Path'''
    prob = direct_collocation.Problem(obj=obj_ndc,
                                      obj_grad=obj_grad_ndc,
                                      equations_of_motion=eom,
                                      state_symbols=state_symbols,
                                      num_collocation_nodes=numNodes,
                                      node_time_interval=intervalValue,
                                      instance_constraints=instanceCons,
                                      time_symbol=t,
                                      bounds=bounds)

    '''Initial Guess of Optimal Solution'''
    initial_guess = define_initial_guess(prob.num_free,
                                         numNodes,
                                         duration,
                                         time)
    solution, info = prob.solve(initial_guess)

    '''Plot Solution'''
    plot_dual_control(solution, numNodes, K11, K22, BB21, BB12)
    # prob.plot_trajectories(solution)
    # prob.plot_constraint_violations(solution)
    # prob.plot_objective_value()
    # plt.show()
