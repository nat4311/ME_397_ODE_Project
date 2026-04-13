"""
references
https://www.scipedia.com/wd/images/e/ed/Draft_Content_631961461p3348.pdf
https://en.wikipedia.org/wiki/Backward_differentiation_formula
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import multiprocessing
import time
import colorsys
from copy import deepcopy
print("====================================")

"""######################################################################
    user defines the ODE, Jacobian (optional), params, initial conditions, and time bounds
######################################################################"""

def f(x, params, t):
    x1, x2 = x
    x1dot = x2
    x2dot = params[0]*(1-x1**2)*x2 - x1

    return np.array([x1dot, x2dot])

n_odes = 1
params_arr = np.random.rand(n_odes,1) * .2

x0_arr = np.random.rand(n_odes,2) * n_odes

t0 = 0
t_end = 1

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

t_odeint = np.linspace(t0,t_end,int(t_end/.01) + 1)
solutions_odeint = list()
for i in range(s):
    dxdt = lambda x,t : f(x, params_arr[i], t)
    solution = odeint(dxdt, x0_arr[i,:], t_odeint)
    solutions_odeint.append(solution)

print(f"odeint total time: {round(1000*(time.time()-timestamp), 2)} ms")

"""######################################################################
                            Single Thread BDF
######################################################################"""

def numerical_jacobian(g, q, t, n, eps=1e-6):
    Jg = np.zeros((n,n))
    for i in range(n):
        dq = np.zeros(n)
        dq[i] = eps
        col = (g(q+dq, t) - g(q, t))/eps
        Jg[:,i] = col
    return Jg


def next_step_size(hprev, xn, xnm1, xnm2, xnm3, Atol=1e-10, Rtol=1e-5, F=.8, Fmin=0, Fmax=2.414):
    """
    LTE and sc
    see https://www.scipedia.com/wd/images/e/ed/Draft_Content_631961461p3348.pdf equation (1)
    uses local truncation error approximation with BDF3 method

    according to LLM:
    Atol: absolute error floor
    Rtol: relative accuracy

    F, Fmin, Fmax
    see https://www.scipedia.com/wd/images/e/ed/Draft_Content_631961461p3348.pdf equation (5)
    """

    LTE_norm = np.linalg.norm(1/3*xn - xnm1 + xnm2 - 1/3*xnm3)
    if LTE_norm < 1e-10:
        return hprev * Fmax
    sc = Atol + max(np.linalg.norm(xn), np.linalg.norm(xnm1))*Rtol
    err = LTE_norm/sc
    
    return hprev * min(Fmax, max(Fmin, (1/err)**(1/3)*F))

def BDF1_step(g, xcurr, t, h, n, Jg=None, newtonMaxIters=20, newtonTolerance=1e-6):
    """
    see reference section 2.1
    g and Jg are dxdt and jacobian of dxdt
    """

    f = lambda q: q - h*g(q, t) - xcurr
    I = np.eye(n)
    if Jg is None:
        dfinv = lambda q: np.linalg.inv(I - h*numerical_jacobian(g, q, t, n)) # todo: add delta to make sure invertible?
    else:
        dfinv = lambda q: np.linalg.inv(I - h*Jg(q, t)) # todo: add delta to make sure invertible?

    q = xcurr
    for i in range(newtonMaxIters):
        residual = f(q)
        if np.linalg.norm(residual) < newtonTolerance:
            break
        if i == newtonMaxIters-1:
            print("reached max iters")
        dq = -dfinv(q)@residual
        q += dq

    assert type(q) == np.ndarray
    return q

def BDF2_step(g, xcurr, xprev, t, h, hprev, n, Jg=None, newtonMaxIters=20, newtonTolerance=1e-10):
    """
    g and Jg are dxdt and jacobian of dxdt
    """

    wn = h/hprev

    # f
    a = (1+2*wn)/(1+wn)
    b = -(1+wn)**2/(1+wn)*xcurr + (wn**2)/(1+wn)*xprev
    f = lambda q: a*q - h*g(q, t) + b

    # df
    c = a*np.eye(n)
    if Jg is None:
        dfinv = lambda q: np.linalg.inv(c - h*numerical_jacobian(g, q, t, n)) # todo: add delta to make sure invertible?
    else:
        dfinv = lambda q: np.linalg.inv(c - h*Jg(q, t)) # todo: add delta to make sure invertible?

    # first guess
    q = xcurr

    # newton
    for i in range(newtonMaxIters):
        residual = f(q)
        if np.linalg.norm(residual) < newtonTolerance:
            break
        if i == newtonMaxIters-1:
            print("reached max iters")
        dq = -dfinv(q)@residual
        q += dq

    assert type(q) == np.ndarray
    return q

def BDF2_solve(g, x0:np.array, t0:float, t_end:float, Jg=None, h0:float=.01):
    """
    INPUT:
        g: ODE to be solved as a function(x,t)
        x0: initial condition as 1D np.array
        t0: start time as float
        t_end: end time as float
        Jg: Optional - Jacobian of the ODE function(x,t). if None provided use numerical method
        h0: initial step size
    ---------------------------------------------------------------------------
    OUTPUT:
        returns x_output, t_output, h_output
        x_output: nxm np.array of state values, x[i,:] = np.array([x0i, x1i, x2i, ...])
        t_output: 1D list of time values, t[i] = time at x[i]
        h_output: 1D list of timestep values used at each iteration
    """

    x_output = [x0]
    t_output = [t0]
    h_output = []
    xcurr = x0
    t = t0
    h = h0
    n = x0.shape[0]

    # compute first step using BDF1
    xcurr = BDF1_step(g, xcurr, t, h, n, Jg)
    t += h
    x_output.append(deepcopy(xcurr))
    t_output.append(deepcopy(t))
    h_output.append(deepcopy(h))

    while t<t_end:
        xprev = x_output[-2] #todo first value
        hprev = h_output[-1]

        # update x
        xcurr = BDF2_step(g, xcurr, xprev, t, h, hprev, n, Jg)

        # update t
        t += h

        # next step size
        if len(x_output) >= 3:
            h = next_step_size(hprev, xcurr, x_output[-1], x_output[-2], x_output[-3])

        # save data
        x_output.append(deepcopy(xcurr))
        t_output.append(deepcopy(t))
        h_output.append(deepcopy(h))

    x_output = np.array(x_output)

    return x_output, t_output, h_output

"""######################################################################
                        Run rk45 solver
######################################################################"""
from rk45 import RK45Solver

timestamp = time.time()

rk45solver = RK45Solver()
x_rk45 = list()
t_rk45 = list()
h_rk45 = list()
for i in range(s):
    dxdt = lambda x,t : f(x, params_arr[i], t)
    x_output, t_output, h_output = rk45solver.solve(dxdt, x0_arr[i,:], t0, t_end)
    x_rk45.append(x_output)
    t_rk45.append(t_output)
    h_rk45.append(h_output)

x_rk45 = [np.array(arr) for arr in x_rk45]
print(f"rk45 total time: {round(1000*(time.time()-timestamp), 2)} ms")

"""######################################################################
                        Run the solver
######################################################################"""

timestamp = time.time()

x_bdf = list()
t_bdf = list()
h_bdf = list()
for i in range(s):
    dxdt = lambda x,t : f(x, params_arr[i], t)
    x_output, t_output, h_output = BDF2_solve(dxdt, x0_arr[i,:], t0, t_end)
    x_bdf.append(x_output)
    t_bdf.append(t_output)
    h_bdf.append(h_output)

print(f"BDF2 total time: {round(1000*(time.time()-timestamp), 2)} ms")

"""######################################################################
                        Evalution and Plots
######################################################################"""


default_saturation = 1
default_value = 1
hues = [h.item() for h in np.linspace(0, .8, s)]
colors = [colorsys.hsv_to_rgb(h, default_saturation, default_value) for h in hues]

plt.figure()

################## plot x1 vs x2
# for i in range(s):
#     plt.plot(
#         solutions_odeint[i][:, 0],
#         solutions_odeint[i][:, 1],
#         color = colors[i],
#         linestyle = ":",
#         label=f"st: {i = }")
#
#     plt.plot(
#         x_bdf[i][:, 0],
#         x_bdf[i][:, 1],
#         color = colors[i],
#         linestyle = "-",
#         alpha = .5,
#         label=f"mt: {i = }")

#################### plot single state x vs t

paramindex = 0
stateno = 0
plt.plot(t_odeint, solutions_odeint[paramindex][:,stateno], linestyle = "-", label = "odeint")
plt.plot(t_rk45[paramindex], x_rk45[paramindex][:,stateno], linestyle = ":", label = "rk45")
plt.plot(t_bdf[paramindex], x_bdf[paramindex][:,stateno], linestyle = "-.", label = "bdf")
plt.legend()

###################### plot step size
# plt.plot(t_bdf[0][:-1], h_bdf[0], label = "bdf", linestyle="-", alpha = .3)

# plt.show()
