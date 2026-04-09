"""
sequential
adaptive and fixed
read about GPU
"""

import time
from math import sin, cos
import numpy as np
import matplotlib.pyplot as plt

m = 1
b = 0
k = 1

def F(t):
    # return sin(t)
    return 0

def f(x, t):
    x1 = x[0]
    x2 = x[1]

    x1dot = x2
    x2dot = -b/m*x2 - k/m*x1 + F(t)/m

    return np.array([x1dot, x2dot])

def df(x,t):
    return np.array([
        [0, 1],
        [-k/m, -b/m]
    ])

t_end = 10
dt = .001
N = int(t_end/dt+1)
t = np.linspace(0,t_end, N)
x0 = np.array([1,0])
implicit_euler_tolerance = 1e-6
implicit_euler_max_iters = 20

x_explicit_euler = np.zeros((N, 2))
x_explicit_euler[0,:] = x0
x_implicit_euler = np.zeros((N, 2))
x_implicit_euler[0,:] = x0

for i in range(N-1):
    # explicit euler
    x_explicit_euler[i+1] = x_explicit_euler[i] + dt*f(x_explicit_euler[i],t[i])

    # implicit euler / BDF1
    x_prev = x_implicit_euler[i,:].copy()
    x_guess = x_explicit_euler[i+1].copy()
    for _ in range(implicit_euler_max_iters):
        res = x_guess - x_prev - dt*f(x_guess, t[i+1])
        if np.linalg.norm(res) < implicit_euler_tolerance:
            break
        J = np.eye(2) - dt*df(x_guess, t[i+1])
        x_guess -= np.linalg.solve(J, res)
    x_implicit_euler[i+1,:] = x_guess

# rk45
x_rk45 = np.zeros((N, 2))
x_rk45[0,:] = x0
dt_rk45 = .01
rk45_tol = .00001
t_rk45 = np.zeros(N)
for i in range(N-1):
    # calc the stuff
    k1 = dt_rk45 * f(x_rk45[i], t_rk45[i])
    k2 = dt_rk45 * f(x_rk45[i] + 1/4*k1, t_rk45[i] + 1/4*dt_rk45)
    k3 = dt_rk45 * f(x_rk45[i] + 3/32*k1 + 9/32*k2, t_rk45[i] + 3/8*dt_rk45)
    k4 = dt_rk45 * f(x_rk45[i] + 1932/2197*k1 - 7200/2197*k2 + 7296/2197*k3, t_rk45[i] + 12/13*dt_rk45)
    k5 = dt_rk45 * f(x_rk45[i] + 439/216*k1 - 8*k2 + 3680/513*k3 - 845/4104*k4, t_rk45[i] + dt_rk45)
    k6 = dt_rk45 * f(x_rk45[i] - 8/27*k1 + 2*k2 - 3544/2565*k3 + 1859/4104*k4 - 11/40*k5, t_rk45[i] + 1/2*dt_rk45)

    y = x_rk45[i] + 25/216*k1 + 1408/2565*k3 + 2197/4101*k4 - 1/5*k5
    z = x_rk45[i] + 16/135*k1 + 6656/12825*k3 + 28561/56430*k4 - 9/50*k5 + 2/55*k6

    # updates
    t_rk45[i+1] = t_rk45[i] + dt_rk45
    x_rk45[i+1] = z

    s = (rk45_tol/(2*np.linalg.norm(z-y)))**.25
    dt_rk45 *= s
if t_rk45[-1] > t_end:
    rk45_end_idx = np.where(t_rk45>t_end)[0][0]
else:
    rk45_end_idx = N

# plots
print(f"euler methods took {N-1} iters")
print(f"rk45 took {rk45_end_idx-1} iters")

plt.figure()
plt.plot(t, x_explicit_euler[:,0], label="x, explicit")
plt.plot(t, x_implicit_euler[:,0], label="x, implicit")
plt.plot(t_rk45[:rk45_end_idx], x_rk45[:rk45_end_idx,0], label="x, rk45")
# plt.plot(t, x_explicit_euler[:,1], label="v, explicit")
# plt.plot(t, x_implicit_euler[:,1], label="v, implicit")
# plt.plot(t_rk45[:rk45_end_idx], x_rk45[:rk45_end_idx,1], label="v, rk45")
plt.legend()
plt.show()
