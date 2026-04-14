import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import multiprocessing
import time
import colorsys

######################################################################
#                           Single Thread RK45
######################################################################

class RK45Solver:
    def __init__(self):
        self.A = [ 0, 2/9, 1/3, 3/4, 1, 5/6 ]
        self.B = [
            [0],
            [2/9],
            [1/12, 1/4],
            [69/128, -243/128, 135/64],
            [-17/12, 27/4, -27/5, 16/15],
            [65/432, -5/16, 13/16, 4/27, 5/144],
        ]
        self.c = [ 1/9, 0, 9/20, 16/45, 1/12 ]
        self.c_hat = [ 47/450, 0, 12/25, 32/225, 1/30, 6/25 ]

    def step(self, dxdt, x, t, h, eps=1e-6):
        """
        NOTE:
        not enough ops in here to warrant parallelizing them? need to check if a reduction on 6 terms is faster than doing 6 on single thread
        all k's depend on previous, cannot parallelize k1, k2, etc
        y and z independent, but could parallelize?
        the norm calculation is a good parallel candidate if n=len(x) is large
        """
        # calc the k's
        k1 = h * dxdt(x,                                                                  t + self.A[0]*h)
        k2 = h * dxdt(x + self.B[1][0]*k1,                                                     t + self.A[1]*h)
        k3 = h * dxdt(x + self.B[2][0]*k1 + self.B[2][1]*k2,                                        t + self.A[2]*h)
        k4 = h * dxdt(x + self.B[3][0]*k1 + self.B[3][1]*k2 + self.B[3][2]*k3,                           t + self.A[3]*h)
        k5 = h * dxdt(x + self.B[4][0]*k1 + self.B[4][1]*k2 + self.B[4][2]*k3 + self.B[4][3]*k4,              t + self.A[4]*h)
        k6 = h * dxdt(x + self.B[5][0]*k1 + self.B[5][1]*k2 + self.B[5][2]*k3 + self.B[5][3]*k4 + self.B[5][4]*k5, t + self.A[5]*h)

        # updates
        y = x + self.c[0]*k1 + self.c[1]*k2 + self.c[2]*k3 + self.c[3]*k4 + self.c[4]*k5
        z = x + self.c_hat[0]*k1 + self.c_hat[1]*k2 + self.c_hat[2]*k3 + self.c_hat[3]*k4 + self.c_hat[4]*k5 + self.c_hat[5]*k6

        if np.linalg.norm(y-z) < eps:
            return z, t+h, h
        else:
            return self.step(dxdt, x, t, h=h/2)

    def solve(self, dxdt, x0, t0, t_end, h0=.01):
        x_output = [x0]
        t_output = [t0]
        x = x0
        t = t0
        h = h0

        while t<t_end:
            h_prev = h
            x, t, h = self.step(dxdt, x, t, h)
            if h == h_prev:
                h *= 2
            """
            NOTE:
            should the output be sent to cpu immediately? or send it all at the end?
            """
            x_output.append([xi.item() for xi in x])
            t_output.append(t)
        return x_output, t_output


if __name__ == "__main__":

    ######################################################################
    #   user defines the ODE, params, initial conditions, and time bounds
    ######################################################################

    """
    NOTE:
    define this as a GPU kernel
    ig user would have to write custom function for gpu parallelization
    x, p - arrays of set size
    t - float
    can parallelize each xi on different thread, write to Xj    (i ~ state_no,    j ~ ode_no)
    if many additions are done, could parallelize by having each non-addition operation done independently, then reduction for each xi

    pass both x and p arrays by reference, they are read only
    """
    def f(x, p, t):
        x1, x2 = x
        x1dot = x2
        x2dot = p[0]*(1-x1**2)*x2 - x1

        return np.array([x1dot, x2dot])

    """
    NOTE:
    user defined x0 and params, negligible - just do on cpu
    """
    n_odes = 4
    params_arr = np.random.rand(n_odes,1) * .2
    x0_arr = np.random.rand(n_odes,2) * n_odes
    t0 = 0
    t_end = 10
    s = x0_arr.shape[0]

    ######################################################################
    #                       Run the solver
    ######################################################################

    """
    NOTE:
    if adaptive stepping, x_output and t_output are undetermined length at compile time - how to deal with output data?
    if fixed timestep, x_output and t_output can be preallocated
    need to know how gpus send data back to cpu
    """
    x_rk45 = list()
    t_rk45 = list()
    rk45solver = RK45Solver()
    for i in range(s):
        dxdt = lambda x,t : f(x, params_arr[i], t)
        x_output, t_output = rk45solver.solve(dxdt, x0_arr[i,:], t0, t_end)
        x_rk45.append(x_output)
        t_rk45.append(t_output)

    x_rk45 = [np.array(arr) for arr in x_rk45]
