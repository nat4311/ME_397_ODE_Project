"""
reference
https://maths.cnam.fr/IMG/pdf/RungeKuttaFehlbergProof.pdf
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import multiprocessing
import time
import colorsys

"""######################################################################
                            Single Thread RK45
######################################################################"""

class RK45Solver:

    def __init__(self):
        """
        Butcher table: see formula 1(Fehlberg) at
        https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta%E2%80%93Fehlberg_method
        """
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
        returns x_next, t_next, h_next
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
        """
        return x_output, t_output, h_output

        x_output: 2D list of state values, x[i] = [x0[i], x1[i], x2[i], ...]
        t_output: 1D list of time values, t[i] = time at x[i]
        h_output: 1D list of timestep values used at each iteration
        """
        x_output = [x0]
        t_output = [t0]
        h_output = [h0]
        x = x0
        t = t0
        h = h0

        while t<t_end:
            h_prev = h
            x, t, h = self.step(dxdt, x, t, h)
            if h == h_prev:
                h *= 2
            x_output.append([xi.item() for xi in x])
            t_output.append(t)
            h_output.append(h)
        return x_output, t_output, h_output


if __name__ == "__main__":

    """######################################################################
        user defines the ODE, params, initial conditions, and time bounds
    ######################################################################"""

    def f(x, p, t):
        x1, x2 = x
        x1dot = x2
        x2dot = p[0]*(1-x1**2)*x2 - x1

        return np.array([x1dot, x2dot])

    n_odes = 4
    params_arr = np.random.rand(n_odes,1) * .2

    x0_arr = np.random.rand(n_odes,2) * n_odes

    t0 = 0
    t_end = 10

    """######################################################################
                            Validate user input
    ######################################################################"""

    s = x0_arr.shape[0]
    assert params_arr.shape[0] == s
    try:
        xdot = f(x0_arr[0], params_arr[0], 0)
        if xdot.shape[0] != x0_arr.shape[1]:
            raise Exception("xdot has wrong shape")
    except Exception as e:
        print(e)
        raise

    """######################################################################
                            Single Thread Built in Solver
    ######################################################################"""


    timestamp = time.time()

    t_odeint = np.linspace(t0,t_end,10000)
    solutions_odeint = list()
    for i in range(s):
        dxdt = lambda x,t : f(x, params_arr[i], t)
        solution = odeint(dxdt, x0_arr[i,:], t_odeint)
        solutions_odeint.append(solution)

    print(f"odeint total time: {round(1000*(time.time()-timestamp), 2)} ms")

    """######################################################################
                            Run the solver
    ######################################################################"""

    timestamp = time.time()

    x_rk45 = list()
    t_rk45 = list()
    h_rk45 = list()
    rk45solver = RK45Solver()
    for i in range(s):
        dxdt = lambda x,t : f(x, params_arr[i], t)
        x_output, t_output, h_output = rk45solver.solve(dxdt, x0_arr[i,:], t0, t_end)
        x_rk45.append(x_output)
        t_rk45.append(t_output)
        h_rk45.append(h_output)

    x_rk45 = [np.array(arr) for arr in x_rk45]
    print(f"rk45 total time: {round(1000*(time.time()-timestamp), 2)} ms")

    """######################################################################
                            Evalution and Plots
    ######################################################################"""

    default_saturation = 1
    default_value = 1
    hues = [h.item() for h in np.linspace(0, .8, s)]
    colors = [colorsys.hsv_to_rgb(h, default_saturation, default_value) for h in hues]

    plt.figure()

    ## plot x1, x2
    for i in range(s):
        plt.plot(
            solutions_odeint[i][:, 0],
            solutions_odeint[i][:, 1],
            color = colors[i],
            linestyle = ":",
            label=f"st: {i = }")

        plt.plot(
            x_rk45[i][:, 0],
            x_rk45[i][:, 1],
            color = colors[i],
            linestyle = "-",
            alpha = .5,
            label=f"mt: {i = }")

    # ## plot rk45 vs odeint
    # i = 1
    # x_index = 1
    # plt.plot(t_odeint, solutions_odeint[i][:,x_index], linestyle = ":", label = "odeint")
    # plt.plot(t_rk45[i], x_rk45[i][:,x_index], label = "rk45", linestyle="-", alpha = .3)
    # plt.legend()

    # ## plot h
    # plt.plot(t_rk45[0], h_rk45[0], label = "rk45", linestyle="-", alpha = .3)

    plt.show()

    """

    """
