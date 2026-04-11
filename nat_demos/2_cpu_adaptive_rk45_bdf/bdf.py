"""
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import multiprocessing
import time
import colorsys

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
                            Single Thread RK45
######################################################################"""

"""
Butcher table: see formula 1(Fehlberg) at
https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta%E2%80%93Fehlberg_method
"""
A = [ 0, 2/9, 1/3, 3/4, 1, 5/6 ]
B = [
    [0],
    [2/9],
    [1/12, 1/4],
    [69/128, -243/128, 135/64],
    [-17/12, 27/4, -27/5, 16/15],
    [65/432, -5/16, 13/16, 4/27, 5/144],
]
c = [ 1/9, 0, 9/20, 16/45, 1/12 ]
c_hat = [ 47/450, 0, 12/25, 32/225, 1/30, 6/25 ]

def rk45_step(self, dxdt, x, t, h, eps=1e-5):
    """
    returns x_next, t_next
    """
    # calc the k's
    k1 = h * dxdt(x,                                                                  t + A[0]*h)
    k2 = h * dxdt(x + B[1][0]*k1,                                                     t + A[1]*h)
    k3 = h * dxdt(x + B[2][0]*k1 + B[2][1]*k2,                                        t + A[2]*h)
    k4 = h * dxdt(x + B[3][0]*k1 + B[3][1]*k2 + B[3][2]*k3,                           t + A[3]*h)
    k5 = h * dxdt(x + B[4][0]*k1 + B[4][1]*k2 + B[4][2]*k3 + B[4][3]*k4,              t + A[4]*h)
    k6 = h * dxdt(x + B[5][0]*k1 + B[5][1]*k2 + B[5][2]*k3 + B[5][3]*k4 + B[5][4]*k5, t + A[5]*h)

    # updates
    y = x + c[0]*k1 + c[1]*k2 + c[2]*k3 + c[3]*k4 + c[4]*k5
    z = x + c_hat[0]*k1 + c_hat[1]*k2 + c_hat[2]*k3 + c_hat[3]*k4 + c_hat[4]*k5 + c_hat[5]*k6

    if np.linalg.norm(y-z) < eps:
        return z, t+h, h
    else:
        return self.rk45_step(dxdt, x, t, h=h/2)

def rk45_solve(self, dxdt, x0, t0, t_end, h0=.01):
    x_output = [x0]
    t_output = [t0]
    h_output = [h0]
    x = x0
    t = t0
    h = h0

    while t<t_end:
        h_prev = h
        x, t, h = self.rk45_step(dxdt, x, t, h)
        if h_prev == h:
            h *= 2
        x_output.append([xi.item() for xi in x])
        t_output.append(t)
        h_output.append(h)
    return x_output, t_output, h_output

timestamp = time.time()

x_rk45 = list()
t_rk45 = list()
h_rk45 = list()
for i in range(s):
    dxdt = lambda x,t : f(x, params_arr[i], t)
    x_output, t_output, h_output = rk45_solve(dxdt, x0_arr[i,:], t0, t_end)
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

if False:
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
else:
    # plt.plot(t_odeint, solutions_odeint[0][:,0], linestyle = ":", label = "odeint")
    plt.plot(t_rk45[0], h_rk45[0], label = "rk45", linestyle="-", alpha = .3)
    plt.legend()

plt.show()
