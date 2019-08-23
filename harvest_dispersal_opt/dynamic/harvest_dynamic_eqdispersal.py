'''
Note: In harvest steadystate, the two dispersal control control variables are separate.
      In harvest dynamic, the two dispersal control control variables are collapsed into one, equal term.
'''

import sys

import numpy as np
import sympy as sym
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
from opty import direct_collocation

from harvest_dispersal_opt.dynamic.codebook.tools import (
    get_init_eql_stock, get_discount_factor
)
from harvest_dispersal_opt.steady_state.steadystate_eqdispersal import (
    get_steadystate_eqdispersal,
)


## Note: Parameters must be global variables for obj and obj_grad
## Economic Parameters
dis = 0.05  # Discount factor
p1 = 1      # price
C11 = .6    # cost coefficient
C12 = .25   # patch 2 is HIGH cost patch, patch 1 low cost
c2 = .1     # cost coefficient on second term
## Ecological Parameters
r11 = 1     # patch 1 growth rate
r22 = 1     # patch 2 growth rates
b = .5      # patch 1, 2 uncontrolled dispersal rate
BB12 = .5   # dispersal bw patches
BB21 = .5   # dispersal bw patches
CB = .1     # dispersal control cost
K11 = 1     # carrying capacity of patch 1
K22 = 1     # carrying capacity of patch 2
umax = 2    # maximum dispersal control
## Time Paramters
duration = 35  # total time
num_nodes = 500   # Num collocation nodes during run
time = np.linspace(0, duration, num_nodes)
interval_val = duration/(num_nodes - 1)  # interval between collocation nodes
dis_factor = get_discount_factor(dis, time)

def main() -> None:
    ## Steady State Open Access and Optimal
    X10, X20 = get_init_eql_stock(C11, C12, p1, r11, r22, K11, K22, c2, b)
    X1T, X2T = get_steadystate_eqdispersal(dis, r11, r22, K11, K22, BB12, BB21,
                                           p1, C11, C12, c2, CB, umax)
    print(str(X10)+"--->"+str(X1T))
    print(str(X20)+"--->"+str(X2T))

    solution_sb_sdc = dynamic_opt(X10, X20, X1T, X2T)

    ## Plot Solution
    prob.plot_trajectories(solution_sb_sdc)
    prob.plot_constraint_violations(solution_sb_sdc)
    prob.plot_objective_value()
    plt.show()


def dynamic_opt(X10: float, X20: float, X1T: float, X2T: float) -> np.array:
    ## Global symbols for time, state vars, and control vars
    t = sym.symbols('t')
    x1, x2, u1, u2, u3 = sym.symbols('x1, x2, u1, u2, u3', cls=sym.Function)

    eom = define_eom(t, x1, x2, u1, u2, u3)
    state_symbols = (x1(t), x2(t))
    specified_symbols = (u1(t), u2(t), u3(t))
    instance_cons = define_instance_constrains(x1, x2, X10, X20, X1T, X2T)
    bounds = define_bounds(t, x1, x2, u1, u2, u3)

    ## Solve for Optimal Dynamic Path
    prob = direct_collocation.Problem(obj=obj_ndc,
                                      obj_grad=obj_grad_ndc,
                                      equations_of_motion=eom,
                                      state_symbols=state_symbols,
                                      num_collocation_nodes=num_nodes,
                                      node_time_interval=interval_val,
                                      instance_constraints=instance_cons,
                                      time_symbol=t,
                                      bounds=bounds)

    ## Initial Guess of Optimal Solution
    initial_guess = define_initial_guess(prob.num_free, num_nodes,
                                         duration, time)
    solution_sb_sdc, info = prob.solve(initial_guess)
    return solution_sb_sdc


def define_eom(t, x1, x2, u1, u2, u3):
    eom = sym.Matrix([(r11*x1(t)*(1 - x1(t)/K11) +
                      u3(t)*b*(x2(t)/K22) - u3(t)*b*(x1(t)/K11) - u1(t) -
                      x1(t).diff()),
                      (r22*x2(t)*(1 - x2(t)/K22) -
                      u3(t)*b*(x2(t)/K22) + u3(t)*b*(x1(t)/K11) - u2(t) -
                      x2(t).diff())
                      ])
    return eom


def define_bounds(t, x1, x2, u1, u2, u3):
    bounds = {x1(t): (0.0, K11),
              x2(t): (0.0, K22),
              u1(t): (0.0, float('inf')),
              u2(t): (0.0, float('inf')),
              u3(t): (0.0, umax,)}
    return bounds


def define_instance_constrains(x1, x2, X10, X20, X1T, X2T):
    instance_constraints = (x1(0.0) - X10,
                            x2(0.0) - X20,
                            x1(duration) - X1T,
                            x2(duration) - X2T,
                            )
    return instance_constraints


def obj_ndc(free):
    x1 = free[0:num_nodes]
    x2 = free[num_nodes:2*num_nodes]
    u1 = free[2*num_nodes:3*num_nodes]
    u2 = free[3*num_nodes:4*num_nodes]
    u3 = free[4*num_nodes:5*num_nodes]
    obj_1 = (dis_factor*(u1*(p1-(C11/x1)-(c2/(x1**2))*u1) +
                          u2*(p1-(C12/x2)-(c2/(x2**2))*u2) -
                          CB*(1-u3)**2 - CB*(1-u3)**2
                          )
             )
    return - interval_val * np.sum(obj_1)


def obj_grad_ndc(free):
    x1 = free[0:num_nodes]
    x2 = free[num_nodes:2*num_nodes]
    u1 = free[2*num_nodes:3*num_nodes]
    u2 = free[3*num_nodes:4*num_nodes]
    u3 = free[4*num_nodes:5*num_nodes]

    grad = np.zeros_like(free)
    grad[0:num_nodes] = dis_factor*u1*(C11/x1**2 + 2*c2*u1/x1**3)  # gradient for x1
    grad[num_nodes:2*num_nodes] = dis_factor*u2*(C12/x2**2 + 2*c2*u2/x2**3)  # gradient for x2
    grad[2*num_nodes:3*num_nodes] = dis_factor*(-C11/x1 - 2*c2*u1/x1**2 + p1)  # gradient for u1
    grad[3*num_nodes:4*num_nodes] = dis_factor*(-C12/x2 - 2*c2*u2/x2**2 + p1)  # gradient for u2
    grad[4*num_nodes:5*num_nodes] = -2*CB*dis_factor*(2*u3 - 2)  # gradiewnt for u3
    return - interval_val * grad


def define_initial_guess(num_free, num_nodes, duration, time):
    initial_guess = np.array([0]*(num_free))
    initial_guess[0:num_nodes] = time/duration  # Initial guess for x1
    initial_guess[num_nodes:2*num_nodes] = time/duration  # Initial guess for x2
    initial_guess[2*num_nodes:3*num_nodes] = .1575*time/duration  # Initial guess for u1
    initial_guess[3*num_nodes:4*num_nodes] = .1575*time/duration  # Initial guess for u2
    initial_guess[4*num_nodes:5*num_nodes] = .21575*time/duration  # Initial guess for u3
    return initial_guess


if __name__ == '__main__':
    main()
