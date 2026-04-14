"""
how to deal with output? adaptive means we can't preallocate memory since we don't know the size

jacobian computation (for newtons method inside BDF_step)
matrix inversion     (invert the jacobian for newton's method)
mat*vec              (to compute the next q_guess in newton's method)
dxdt computation     (to compute residual for newton's method. this is defined by user, less important?)

vector norm calcs    (these appear in multiple places)
vector sums          (if n=O(10), should be vectorized without any additional work from us? need to check size of vectorized add for gpu. these appear in multiple places)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import multiprocessing
import time
import colorsys
from copy import deepcopy

######################################################################
#                           Single Thread BDF
######################################################################

"""
NOTE:
definitely write this as a gpu kernel
*** HIGH VALUE OPERATION ***
"""
def numerical_jacobian(g, q, t, n, eps=1e-6):
    Jg = np.zeros((n,n))
    for i in range(n):
        dq = np.zeros(n)
        dq[i] = eps
        col = (g(q+dq, t) - g(q, t))/eps
        Jg[:,i] = col
    return Jg

"""
NOTE:
can maybe parallelize the norm calcs and vector additions
"""
def next_step_size(hprev, xn, xnm1, xnm2, xnm3, Atol=1e-10, Rtol=1e-5, F=.8, Fmin=0, Fmax=2.414):
    LTE_norm = np.linalg.norm(1/3*xn - xnm1 + xnm2 - 1/3*xnm3)
    if LTE_norm < 1e-10:
        return hprev * Fmax
    sc = Atol + max(np.linalg.norm(xn), np.linalg.norm(xnm1))*Rtol
    err = LTE_norm/sc
    
    return hprev * min(Fmax, max(Fmin, (1/err)**(1/3)*F))


def BDF1_step(g, xcurr, t, h, n, Jg=None, newtonMaxIters=20, newtonTolerance=1e-6):
    f = lambda q: q - h*g(q, t) - xcurr
    I = np.eye(n)
    """
    NOTE:
    the jacobian numerical computation and inversion should be parallelized on GPU
    *** HIGH VALUE OPERATION ***
    """
    if Jg is None:
        dfinv = lambda q: np.linalg.inv(I - h*numerical_jacobian(g, q, t, n)) # todo: add delta to make sure invertible?
    else:
        dfinv = lambda q: np.linalg.inv(I - h*Jg(q, t)) # todo: add delta to make sure invertible?

    q = deepcopy(xcurr)
    for i in range(newtonMaxIters):
        """
        NOTE:
        the residual calculation involves vector addition and computing dxdt (g here)
        this can probably be parallelized
        there is also the norm computation
        """
        residual = f(q)
        if np.linalg.norm(residual) < newtonTolerance:
            break
        if i == newtonMaxIters-1:
            print("WARNING: BDF1_step reached max iters")
        """
        NOTE:
        mat*vec
        *** HIGH VALUE OPERATION ***
        """
        dq = -dfinv(q)@residual
        q += dq

    assert type(q) == np.ndarray
    return q


def BDF2_step(g, xcurr, xprev, t, h, hprev, n, Jg=None, newtonMaxIters=20, newtonTolerance=1e-10):
    wn = h/hprev

    # f
    a = (1+2*wn)/(1+wn)
    b = -(1+wn)**2/(1+wn)*xcurr + (wn**2)/(1+wn)*xprev
    f = lambda q: a*q - h*g(q, t) + b

    # df
    """
    NOTE:
    the jacobian numerical computation and inversion should be parallelized on GPU
    *** HIGH VALUE OPERATION ***
    """
    c = a*np.eye(n)
    if Jg is None:
        dfinv = lambda q: np.linalg.inv(c - h*numerical_jacobian(g, q, t, n)) # todo: add delta to make sure invertible?
    else:
        dfinv = lambda q: np.linalg.inv(c - h*Jg(q, t)) # todo: add delta to make sure invertible?

    # first guess
    q = deepcopy(xcurr)

    # newton
    for i in range(newtonMaxIters):
        residual = f(q)
        if np.linalg.norm(residual) < newtonTolerance:
            break
        if i == newtonMaxIters-1:
            print("WARNING: BDF2_step reached max iters")
        """
        NOTE:
        mat*vec
        *** HIGH VALUE OPERATION ***
        """
        dq = -dfinv(q)@residual
        q += dq

    assert type(q) == np.ndarray
    return q


def BDF2_solve(g, x0:np.array, t0:float, t_end:float, Jg=None, h0:float=.01):
    """
    NOTE:
    g is dxdt
    Jg is the jacobian - if none provided, use the numerical jacobian function
    """
    x_output = [deepcopy(x0)]
    t_output = [deepcopy(t0)]
    xcurr = x0
    t = t0
    h = h0
    """
    NOTE:
    biros said n will typically be O(10)
    so not too many states, but 10x10 jacobian w newton's method should have parallelization opportunities
    """
    n = x0.shape[0]

    # compute first step using BDF1
    xcurr = BDF1_step(g, xcurr, t, h, n, Jg)
    t += h
    x_output.append(deepcopy(xcurr))
    t_output.append(deepcopy(t))
    hprev = h

    while t<t_end:
        xprev = deepcopy(x_output[-2])

        # update x
        xcurr = BDF2_step(g, xcurr, xprev, t, h, hprev, n, Jg)

        # update t
        t += h

        # next step size
        hprev = h
        if len(x_output) >= 3:
            h = next_step_size(hprev, xcurr, deepcopy(x_output[-1]), deepcopy(x_output[-2]), deepcopy(x_output[-3]))

        # save data
        x_output.append(deepcopy(xcurr))
        t_output.append(deepcopy(t))

    x_output = np.array(x_output)

    return x_output, t_output


if __name__ == "__main__":

    ######################################################################
    #   user defines the ODE, Jacobian (optional), params, initial conditions, and time bounds
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
    def dxdt(x, params, t):
        x1, x2 = x
        x1dot = x2
        x2dot = params[0]*(1-x1**2)*x2 - x1

        return np.array([x1dot, x2dot])

    """
    these are negligible, perform on CPU
    """
    n_odes = 4
    params_arr = np.random.rand(n_odes,1) * .2
    x0_arr = np.random.rand(n_odes,2) * n_odes
    t0 = 0
    t_end = 10
    s = x0_arr.shape[0]

    ######################################################################
    #                       Run the BDF solver
    ######################################################################

    """
    NOTE:
    if adaptive stepping, x_output and t_output are undetermined length at compile time - how to deal with output data?
    if fixed timestep, x_output and t_output can be preallocated
    need to know how gpus send data back to cpu
    """
    x_bdf = list()
    t_bdf = list()
    for i in range(s):
        g = lambda x,t : dxdt(x, params_arr[i], t)
        x_output, t_output = BDF2_solve(g, x0_arr[i,:], t0, t_end)
        x_bdf.append(x_output)
        t_bdf.append(t_output)
