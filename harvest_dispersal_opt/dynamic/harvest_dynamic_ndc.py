import sys

import numpy as np
from opty import direct_collocation
import sympy as sym
import matplotlib.pyplot as plt

from steadystate_ndc import get_steadystate_ndc


def get_discount_factor(dis, time):
    discntFactr = (1/(1+dis))**time  # Define list of discount factors
    return discntFactr


def define_eom(x1, x2, u1, u2):
    eom = sym.Matrix([(r11*x1(t)*(1 - x1(t)/K11) +
                      1*b*(x2(t)/K22) - 1*b*(x1(t)/K11) - u1(t) - x1(t).diff()),
                      (r22*x2(t)*(1 - x2(t)/K22) -
                      1*b*(x2(t)/K22) + 1*b*(x1(t)/K11) - u2(t) - x2(t).diff())
                      ])
    return eom


def define_bounds(x1, x2, u1, u2):
    bounds = {x1(t): (0.0, K11),
              x2(t): (0.0, K22),
              u1(t): (0.0, float('inf')),
              u2(t): (0.0, float('inf')), }
    return bounds


def define_instance_constrains(x1, x2, X10, X20, X1T, X2T):
    instance_constraints = (x1(0.0) - X10,
                            x2(0.0) - X20,
                            x1(duration) - X1T,  # .816094,# (ss_ndc: 0.821732893841439),(opty: .816094)
                            x2(duration) - X2T,  # .6446575 # (ss_ndc: 0.6513857890339054),(opty: .6446575)
                            )
    return instance_constraints


def obj_ndc(free):
    x1 = free[0:num_nodes]
    x2 = free[num_nodes:2*num_nodes]
    u1 = free[2*num_nodes:3*num_nodes]
    u2 = free[3*num_nodes:4*num_nodes]
    obj_1 = (discnt_factr*(u1*(p1-(C11/x1)-(c2/(x1**2))*u1) +
                           u2*(p1-(C12/x2)-(c2/(x2**2))*u2))
             )
    return - interval_value * np.sum(obj_1)


def obj_grad_ndc(free):
    x1 = free[0:num_nodes]
    x2 = free[num_nodes:2*num_nodes]
    u1 = free[2*num_nodes:3*num_nodes]
    u2 = free[3*num_nodes:4*num_nodes]

    grad = np.zeros_like(free)
    grad[0:num_nodes] = discnt_factr*u1*(C11/x1**2 + 2*c2*u1/x1**3)  # gradient for x1
    grad[num_nodes:2*num_nodes] = discnt_factr*u2*(2*c2*u2/x2**3 + C12/x2**2)  # gradient for x2
    grad[2*num_nodes:3*num_nodes] = discnt_factr*(-C11/x1 - 2*c2*u1/x1**2 + p1)  # gradient for u1
    grad[3*num_nodes:4*num_nodes] = discnt_factr*(-2*c2*u2/x2**2 - C12/x2 + p1)  # gradient for u2
    return - interval_value * grad


def define_initial_guess(num_free, num_nodes, duration, time):
    initial_guess = np.array([0]*(num_free))
    initial_guess[0:num_nodes] = time/duration  # Initial guess for x1
    initial_guess[num_nodes:2*num_nodes] = time/duration  # Initial guess for x2
    initial_guess[2*num_nodes:3*num_nodes] = .1575*time/duration  # Initial guess for u1
    initial_guess[3*num_nodes:4*num_nodes] = .1575*time/duration  # Initial guess for u2
    return initial_guess


if __name__ == '__main__':
    '''Economic Parameters'''
    dis = 0.05  # discount factor #set to 0 to turn off discounting
    p1 = 1      # price
    c2 = .1     # harvest control cost coefficient on second term
    umax = 2    # maximum dispersal control
    C11 = 0.6   # patch 1 cost coefficient (previously 0.8)(?)
    C12 = 0.25  # patch 2 cost coefficient (previously 0.1)(?)
    '''Ecological Parameters'''
    K11 = 1  # patch 1 carrying capacity
    K22 = 1  # patch 2 carrying capacity
    r11 = 1  # patch 1 growth rate
    r22 = 1  # patch 2 growth rate
    b = .5   # uncontrolled dispersal
    '''Time Paramters'''
    duration = 35  # total time
    num_nodes = 500   # Num collocation nodes during run
    time = np.linspace(0, duration, num_nodes)
    interval_value = duration/(num_nodes - 1)  # time interval between collocation nodes
    discnt_factr = get_discount_factor(dis, time)

    '''Steady State Open Access and Optimal'''
    X10, X20 = get_init_equilibrium_stock(C11, C12, p1, r11, r22, K22, K11, c2, b)
    X1T, X2T = get_steadystate_ndc(dis, r11, K11, b, p1, C11, C12, c2)
    print(str(X10) + " ---> " + str(X1T))
    print(str(X20) + " ---> " + str(X2T))

    '''Paramterize Dynamic Path Optimization'''
    t = sym.symbols('t')
    x1, x2, u1, u2 = sym.symbols('x1, x2, u1, u2', cls=sym.Function)
    eom = define_eom(x1, x2, u1, u2)
    state_symbols = (x1(t), x2(t))
    specified_symbols = (u1(t), u2(t))
    instanceCons = define_instance_constrains(x1, x2, X10, X20, X1T, X2T)
    bounds = define_bounds(x1, x2, u1, u2)

    '''Solve for Optimal Dynamic Path'''
    prob = direct_collocation.Problem(obj=obj_ndc,
                                      obj_grad=obj_grad_ndc,
                                      equations_of_motion=eom,
                                      state_symbols=state_symbols,
                                      num_collocation_nodes=num_nodes,
                                      node_time_interval=interval_value,
                                      instance_constraints=instanceCons,
                                      time_symbol=t,
                                      bounds=bounds)

    '''Initial Guess of Optimal Solution'''
    initial_guess = define_initial_guess(prob.num_free,
                                         num_nodes,
                                         duration,
                                         time)
    solution_sb_ndc, info = prob.solve(initial_guess)

    '''Plot Solution'''
    prob.plot_trajectories(solution_sb_ndc)
    prob.plot_constraint_violations(solution_sb_ndc)
    prob.plot_objective_value()
    plt.show()
