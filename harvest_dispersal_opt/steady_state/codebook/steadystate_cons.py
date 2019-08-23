import numpy as np
from scipy.optimize import NonlinearConstraint

def const01_ineq_init(x, r11, r22, K11, K22, BB12, BB21):
    H1 = (r11*x[0]*(1-(x[0]/K11)) + BB12*x[2]*x[1]/K22 - BB21*x[3]*x[0]/K11)
    H2 = (r22*x[1]*(1-(x[1]/K22)) + BB21*x[3]*x[0]/K11 - BB12*x[2]*x[1]/K22)
    return np.array([H1, H2]).T


# def const01_ineq_init(x, r11, r22, K11, K22, BB12, BB21):
#     H1 = (r11*x[0]*(1-(x[0]/K11)) + BB12*x[2]*x[1]/K22 - BB21*x[3]*x[0]/K11)
#     negH1 = -1*H1
#     H2 = (r22*x[1]*(1-(x[1]/K22)) + BB21*x[3]*x[0]/K11 - BB12*x[2]*x[1]/K22)
#     negH2 = -1*H2
#     return np.array([negH1, negH2]).T


def const01_eq(x, dis, umax, r11, r22, K11, K22, BB12, BB21, P, c2, C11, C12, CB):
    '''For cobyla'''
    lam1 = P - C11/x[0] - (2*c2*x[4]/(x[0]**2)) + x[6]
    lam2 = P - C12/x[1] - (2*c2*x[5]/(x[1]**2)) + x[7]
    f1p = r11 - (2*r11*x[0]/K11)
    f2p = r22 - (2*r22*x[1]/K22)
    c1p = -x[4]*(C11/(x[0]**2)) - (2*c2*(x[4]**2)/(x[0]**3))
    c2p = -x[5]*(C12/(x[1]**2)) - (2*c2*(x[5]**2)/(x[1]**3))

    ceq01 = 2*CB*(1-x[2]) + BB12*(x[1]/K22)*(lam1-lam2) + x[8] - x[10]
    ceq02 = 2*CB*(1-x[3]) + BB21*(x[0]/K11)*(lam2-lam1) + x[9] - x[11]
    ceq03 = lam1*(dis - f1p + (BB21*x[3]/K22)) + c1p - lam2*(BB21*x[3]/K22)
    ceq04 = lam2*(dis - f2p + (BB12*x[2]/K11)) + c2p - lam1*(BB12*x[2]/K11)
    ceq05 = x[4] - (r11*x[0]*(1-(x[0]/K11)) + BB12*(x[2]*x[1]/K22) - BB21*(x[3]*x[0]/K11))
    ceq06 = x[5] - (r22*x[1]*(1-(x[1]/K22)) - BB12*(x[2]*x[1]/K22) + BB21*(x[3]*x[0]/K11))
    ceq07 = x[4]*x[6]
    ceq08 = x[5]*x[7]
    ceq09 = x[2]*x[8]
    ceq10 = x[3]*x[9]
    ceq11 = (umax-x[2])*x[10]
    ceq12 = (umax-x[3])*x[11]
    return np.array([ceq01, ceq02, ceq03, ceq04, ceq05, ceq06, ceq07, ceq08, ceq09, ceq10, ceq11, ceq12,]).T


def const01_eq_negs(x, dis, umax, r11, r22, K11, K22, BB12, BB21, P, c2, C11, C12, CB):
    '''For cobyla'''
    lam1 = P - C11/x[0] - (2*c2*x[4]/(x[0]**2)) + x[6]
    lam2 = P - C12/x[1] - (2*c2*x[5]/(x[1]**2)) + x[7]
    f1p = r11 - (2*r11*x[0]/K11)
    f2p = r22 - (2*r22*x[1]/K22)
    c1p = -x[4]*(C11/(x[0]**2)) - (2*c2*(x[4]**2)/(x[0]**3))
    c2p = -x[5]*(C12/(x[1]**2)) - (2*c2*(x[5]**2)/(x[1]**3))

    ceq01 = 2*CB*(1-x[2]) + BB12*(x[1]/K22)*(lam1-lam2) + x[8] - x[10]
    ceq02 = 2*CB*(1-x[3]) + BB21*(x[0]/K11)*(lam2-lam1) + x[9] - x[11]
    ceq03 = lam1*(dis - f1p + (BB21*x[3]/K22)) + c1p - lam2*(BB21*x[3]/K22)
    ceq04 = lam2*(dis - f2p + (BB12*x[2]/K11)) + c2p - lam1*(BB12*x[2]/K11)
    ceq05 = x[4] - (r11*x[0]*(1-(x[0]/K11)) + BB12*(x[2]*x[1]/K22) - BB21*(x[3]*x[0]/K11))
    ceq06 = x[5] - (r22*x[1]*(1-(x[1]/K22)) - BB12*(x[2]*x[1]/K22) + BB21*(x[3]*x[0]/K11))
    ceq07 = x[4]*x[6]
    ceq08 = x[5]*x[7]
    ceq09 = x[2]*x[8]
    ceq10 = x[3]*x[9]
    ceq11 = (umax-x[2])*x[10]
    ceq12 = (umax-x[3])*x[11]
    return np.array([-ceq01, -ceq02, -ceq03, -ceq04, -ceq05, -ceq06, -ceq07, -ceq08, -ceq09, -ceq10, -ceq11, -ceq12,]).T


def const01_eq_jac(x, dis, umax, r11, r22, K11, K22, BB12, BB21, P, c2, C11, C12, CB):
    '''For cobyla'''
    jac = np.array([[BB12*x[1]*(C11/x[0]**2 + 4*c2*x[4]/x[0]**3)/K22, BB12*x[1]*(-C12/x[1]**2 - 4*c2*x[5]/x[1]**3)/K22 + BB12*(-C11/x[0] + C12/x[1] + 2*c2*x[5]/x[1]**2 - 2*c2*x[4]/x[0]**2 + x[6] - x[7])/K22, -2*CB, 0, -2*BB12*c2*x[1]/(K22*x[0]**2), 2*BB12*c2/(K22*x[1]), BB12*x[1]/K22, -BB12*x[1]/K22, 1, 0, -1, 0],
    [BB21*x[0]*(-C11/x[0]**2 - 4*c2*x[4]/x[0]**3)/K11 + BB21*(C11/x[0] - C12/x[1] - 2*c2*x[5]/x[1]**2 + 2*c2*x[4]/x[0]**2 - x[6] + x[7])/K11, BB21*x[0]*(C12/x[1]**2 + 4*c2*x[5]/x[1]**3)/K11, 0, -2*CB, 2*BB21*c2/(K11*x[0]), -2*BB21*c2*x[0]/(K11*x[1]**2), -BB21*x[0]/K11, BB21*x[0]/K11, 0, 1, 0, -1],
    [2*C11*x[4]/x[0]**3 + 6*c2*x[4]**2/x[0]**4 + (C11/x[0]**2 + 4*c2*x[4]/x[0]**3)*(BB21*x[3]/K22 + dis - r11 + 2*r11*x[0]/K11) + 2*r11*(-C11/x[0] + P - 2*c2*x[4]/x[0]**2 + x[6])/K11, -BB21*x[3]*(C12/x[1]**2 + 4*c2*x[5]/x[1]**3)/K22, 0, BB21*(-C11/x[0] + P - 2*c2*x[4]/x[0]**2 + x[6])/K22 - BB21*(-C12/x[1] + P - 2*c2*x[5]/x[1]**2 + x[7])/K22, -C11/x[0]**2 - 2*c2*(BB21*x[3]/K22 + dis - r11 + 2*r11*x[0]/K11)/x[0]**2 - 4*c2*x[4]/x[0]**3, 2*BB21*c2*x[3]/(K22*x[1]**2), BB21*x[3]/K22 + dis - r11 + 2*r11*x[0]/K11, -BB21*x[3]/K22, 0, 0, 0, 0],
    [-BB12*x[2]*(C11/x[0]**2 + 4*c2*x[4]/x[0]**3)/K11, 2*C12*x[5]/x[1]**3 + 6*c2*x[5]**2/x[1]**4 + (C12/x[1]**2 + 4*c2*x[5]/x[1]**3)*(BB12*x[2]/K11 + dis - r22 + 2*r22*x[1]/K22) + 2*r22*(-C12/x[1] + P - 2*c2*x[5]/x[1]**2 + x[7])/K22, -BB12*(-C11/x[0] + P - 2*c2*x[4]/x[0]**2 + x[6])/K11 + BB12*(-C12/x[1] + P - 2*c2*x[5]/x[1]**2 + x[7])/K11, 0, 2*BB12*c2*x[2]/(K11*x[0]**2), -C12/x[1]**2 - 2*c2*(BB12*x[2]/K11 + dis - r22 + 2*r22*x[1]/K22)/x[1]**2 - 4*c2*x[5]/x[1]**3, -BB12*x[2]/K11, BB12*x[2]/K11 + dis - r22 + 2*r22*x[1]/K22, 0, 0, 0, 0],
    [BB21*x[3]/K11 - r11*(1 - x[0]/K11) + r11*x[0]/K11, -BB12*x[2]/K22, -BB12*x[1]/K22, BB21*x[0]/K11, 1, 0, 0, 0, 0, 0, 0, 0],
    [-BB21*x[3]/K11, BB12*x[2]/K22 - r22*(1 - x[1]/K22) + r22*x[1]/K22, BB12*x[1]/K22, -BB21*x[0]/K11, 0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, x[6], 0, x[4], 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, x[7], 0, x[5], 0, 0, 0, 0],
    [0, 0, x[8], 0, 0, 0, 0, 0, x[2], 0, 0, 0],
    [0, 0, 0, x[9], 0, 0, 0, 0, 0, x[3], 0, 0],
    [0, 0, -x[10], 0, 0, 0, 0, 0, 0, 0, umax - x[2], 0],
    [0, 0, 0, -x[11], 0, 0, 0, 0, 0, 0, 0, umax - x[3]]]).T
    return jac


def const01_eq_negs_jac(x, dis, umax, r11, r22, K11, K22, BB12, BB21, P, c2, C11, C12, CB):
    '''For cobyla'''
    jac = np.array([[BB12*x[1]*(C11/x[0]**2 + 4*c2*x[4]/x[0]**3)/K22, BB12*x[1]*(-C12/x[1]**2 - 4*c2*x[5]/x[1]**3)/K22 + BB12*(-C11/x[0] + C12/x[1] + 2*c2*x[5]/x[1]**2 - 2*c2*x[4]/x[0]**2 + x[6] - x[7])/K22, -2*CB, 0, -2*BB12*c2*x[1]/(K22*x[0]**2), 2*BB12*c2/(K22*x[1]), BB12*x[1]/K22, -BB12*x[1]/K22, 1, 0, -1, 0],
    [BB21*x[0]*(-C11/x[0]**2 - 4*c2*x[4]/x[0]**3)/K11 + BB21*(C11/x[0] - C12/x[1] - 2*c2*x[5]/x[1]**2 + 2*c2*x[4]/x[0]**2 - x[6] + x[7])/K11, BB21*x[0]*(C12/x[1]**2 + 4*c2*x[5]/x[1]**3)/K11, 0, -2*CB, 2*BB21*c2/(K11*x[0]), -2*BB21*c2*x[0]/(K11*x[1]**2), -BB21*x[0]/K11, BB21*x[0]/K11, 0, 1, 0, -1],
    [2*C11*x[4]/x[0]**3 + 6*c2*x[4]**2/x[0]**4 + (C11/x[0]**2 + 4*c2*x[4]/x[0]**3)*(BB21*x[3]/K22 + dis - r11 + 2*r11*x[0]/K11) + 2*r11*(-C11/x[0] + P - 2*c2*x[4]/x[0]**2 + x[6])/K11, -BB21*x[3]*(C12/x[1]**2 + 4*c2*x[5]/x[1]**3)/K22, 0, BB21*(-C11/x[0] + P - 2*c2*x[4]/x[0]**2 + x[6])/K22 - BB21*(-C12/x[1] + P - 2*c2*x[5]/x[1]**2 + x[7])/K22, -C11/x[0]**2 - 2*c2*(BB21*x[3]/K22 + dis - r11 + 2*r11*x[0]/K11)/x[0]**2 - 4*c2*x[4]/x[0]**3, 2*BB21*c2*x[3]/(K22*x[1]**2), BB21*x[3]/K22 + dis - r11 + 2*r11*x[0]/K11, -BB21*x[3]/K22, 0, 0, 0, 0],
    [-BB12*x[2]*(C11/x[0]**2 + 4*c2*x[4]/x[0]**3)/K11, 2*C12*x[5]/x[1]**3 + 6*c2*x[5]**2/x[1]**4 + (C12/x[1]**2 + 4*c2*x[5]/x[1]**3)*(BB12*x[2]/K11 + dis - r22 + 2*r22*x[1]/K22) + 2*r22*(-C12/x[1] + P - 2*c2*x[5]/x[1]**2 + x[7])/K22, -BB12*(-C11/x[0] + P - 2*c2*x[4]/x[0]**2 + x[6])/K11 + BB12*(-C12/x[1] + P - 2*c2*x[5]/x[1]**2 + x[7])/K11, 0, 2*BB12*c2*x[2]/(K11*x[0]**2), -C12/x[1]**2 - 2*c2*(BB12*x[2]/K11 + dis - r22 + 2*r22*x[1]/K22)/x[1]**2 - 4*c2*x[5]/x[1]**3, -BB12*x[2]/K11, BB12*x[2]/K11 + dis - r22 + 2*r22*x[1]/K22, 0, 0, 0, 0],
    [BB21*x[3]/K11 - r11*(1 - x[0]/K11) + r11*x[0]/K11, -BB12*x[2]/K22, -BB12*x[1]/K22, BB21*x[0]/K11, 1, 0, 0, 0, 0, 0, 0, 0],
    [-BB21*x[3]/K11, BB12*x[2]/K22 - r22*(1 - x[1]/K22) + r22*x[1]/K22, BB12*x[1]/K22, -BB21*x[0]/K11, 0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, x[6], 0, x[4], 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, x[7], 0, x[5], 0, 0, 0, 0],
    [0, 0, x[8], 0, 0, 0, 0, 0, x[2], 0, 0, 0],
    [0, 0, 0, x[9], 0, 0, 0, 0, 0, x[3], 0, 0],
    [0, 0, -x[10], 0, 0, 0, 0, 0, 0, 0, umax - x[2], 0],
    [0, 0, 0, -x[11], 0, 0, 0, 0, 0, 0, 0, umax - x[3]]]).T
    negjac = -jac
    return negjac

# for trust constr
def get_Nonlinear_Constraints(dis, umax, r11, r22, K11, K22, BB12, BB21,
                             P, c2, C11, C12, CB):
    '''For trust-constr'''
    def const01_eq_nonlin(x, dis, umax, r11, r22, K11, K22, BB12, BB21, P, c2, C11, C12, CB):
        lam1 = P - C11/x[0] - (2*c2*x[4]/(x[0]**2)) + x[6]
        lam2 = P - C12/x[1] - (2*c2*x[5]/(x[1]**2)) + x[7]
        f1p = r11 - (2*r11*x[0]/K11)
        f2p = r22 - (2*r22*x[1]/K22)
        c1p = -x[4]*(C11/(x[0]**2)) - (2*c2*(x[4]**2)/(x[0]**3))
        c2p = -x[5]*(C12/(x[1]**2)) - (2*c2*(x[5]**2)/(x[1]**3))

        ceq01 = 2*CB*(1-x[2]) + BB12*(x[1]/K22)*(lam1-lam2) + x[8] - x[10]
        ceq02 = 2*CB*(1-x[3]) + BB21*(x[0]/K11)*(lam2-lam1) + x[9] - x[11]
        ceq03 = lam1*(dis - f1p + (BB21*x[3]/K22)) + c1p - lam2*(BB21*x[3]/K22)
        ceq04 = lam2*(dis - f2p + (BB12*x[2]/K11)) + c2p - lam1*(BB12*x[2]/K11)
        ceq05 = x[4] - (r11*x[0]*(1-(x[0]/K11)) + BB12*(x[2]*x[1]/K22) - BB21*(x[3]*x[0]/K11))
        ceq06 = x[5] - (r22*x[1]*(1-(x[1]/K22)) - BB12*(x[2]*x[1]/K22) + BB21*(x[3]*x[0]/K11))
        ceq07 = x[4]*x[6]
        ceq08 = x[5]*x[7]
        ceq09 = x[2]*x[8]
        ceq10 = x[3]*x[9]
        ceq11 = (umax-x[2])*x[10]
        ceq12 = (umax-x[3])*x[11]
        cons = np.array([ceq01, ceq02, ceq03, ceq04, ceq05, ceq06, ceq07, ceq08, ceq09, ceq10, ceq11, ceq12,]).T
        cons = np.append(cons,-cons)
        return cons
    nl_cons = NonlinearConstraint(fun=lambda x: const01_eq_nonlin(x, dis, umax,
                                                                  r11, r22,
                                                                  K11, K22,
                                                                  BB12, BB21,
                                                                  P, c2,
                                                                  C11, C12, CB),
                               lb=0,
                               ub=np.inf,
                               # jac=lambda x: const01_eq_jac(x, dis, umax, r11,
                               #                              r22, K11, K22, BB12,
                               #                              BB21, P, c2, C11,
                               #                              C12, CB),
                               # hess=BFGS()
                               )
    return nl_cons


def const02_ineq_init(x, r11, r22, K11, K22, BB12, BB21):
    H1 = (r11*x[0]*(1-(x[0]/K11)) + BB12*(x[2]*x[1]/K22 - x[2]*x[0]/K11))
    negH1 = -1*H1
    H2 = (r22*x[1]*(1-(x[1]/K22)) + BB21*(x[2]*x[0]/K11 - x[2]*x[1]/K22))
    negH2 = -1*H2
    return np.array([negH1, negH2]).T


def const02_eq(x, dis, r11, r22, K11, K22, BB12, BB21, P, c2, C11, C12, CB):
    lam1 = (P - (C11/x[0]) - (2*c2*x[3]/(x[0]**2)) + x[5])
    lam2 = (P - (C12/x[1]) - (2*c2*x[4]/(x[1]**2)) + x[6])
    f1p = r11 - (2*r11*x[0]/K11)
    f2p = r22 - (2*r22*x[1]/K22)
    c1p = -x[3]*(C11/(x[0]**2)) - (2*c2*(x[3]**2)/(x[0]**3))
    c2p = -x[4]*(C12/(x[1]**2)) - (2*c2*(x[4]**2)/(x[1]**3))

    const2_eq1 = 4*CB*(1-x[2]) + ((BB21*x[0]/K11) - (BB12*x[1]/K22))*(lam2-lam1) + x[7] - x[8]
    const2_eq2 = lam1*(dis - f1p + (BB21*x[2]/K11)) + c1p - lam2*(BB21*x[2]/K22)
    const2_eq3 = lam2*(dis - f2p + (BB12*x[2]/K22)) + c2p - lam1*(BB12*x[2]/K11)
    const2_eq4 = x[3] - ((r11*x[0]*(1-(x[0]/K11)) + BB12*((x[2]*x[1]/K22)) - BB21*((x[2]*x[0]/K11))))
    const2_eq5 = x[4] - ((r22*x[1]*(1-(x[1]/K22)) + BB21*((x[2]*x[0]/K11)) - BB12*((x[2]*x[1]/K22))))
    return np.array([const2_eq1, const2_eq2, const2_eq3, const2_eq4, const2_eq5,]).T


def const02_ineq(x, umax):
    const2_eq6 = x[3]*x[5]
    const2_eq7 = x[4]*x[6]
    const2_eq8 = x[2]*x[7]
    const2_eq9 = (umax-x[2])*x[8]
    return np.array([const2_eq6, const2_eq7, const2_eq8, const2_eq9,]).T

def const02_ineq(x, umax):
    const2_eq6 = x[3]*x[5]
    const2_eq7 = x[4]*x[6]
    const2_eq8 = x[2]*x[7]
    const2_eq9 = (umax-x[2])*x[8]
    return np.array([const2_eq6, const2_eq7, const2_eq8, const2_eq9,]).T


def const03_eq(x, dis, R, K, B, P, C11, C12, c2):
    lam1 = P - C11/x[0] - 2*c2*x[2]/x[0]**2 + x[4]
    lam2 = P - C12/x[1] - 2*c2*x[3]/x[1]**2 + x[5]
    f1p = R - 2*R*x[0]/K
    f2p = R - 2*R*x[1]/K

    ceq01 = 100*(lam1*(dis - f1p + B/K) - x[2]*(C11/x[0]**2 + 2*c2*x[2]/x[0]**3) - lam2*(B/K))
    ceq02 = 100*(lam2*(dis - f2p + B/K) - x[3]*(C12/x[1]**2 + 2*c2*x[3]/x[1]**3) - lam1*(B/K))
    ceq03 = 100*(x[2] - (R*x[0]*(1-(x[0]/K)) + B*x[1]/K - B*x[0]/K))
    ceq04 = 100*(x[3] - (R*x[1]*(1-(x[1]/K)) + B*x[0]/K - B*x[1]/K))
    ceq05 = 100*(x[2]*x[4]) # harvest 1 positive
    ceq06 = 100*(x[3]*x[5]) # harvest 2 positive
    return np.array([ceq01, ceq02, ceq03, ceq04, ceq05, ceq06,]).T

def const03_ineq(x):
    ceq05 = 100*(x[2]*x[4]) # harvest 1 positive
    ceq06 = 100*(x[3]*x[5]) # harvest 2 positive
    return np.array([ceq05, ceq06,]).T
    # NOTE: Needs mu because
