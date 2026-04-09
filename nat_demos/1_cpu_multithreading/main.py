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

n_odes = 20
params_arr = np.random.rand(n_odes,1) * .2

x0_arr = np.random.rand(n_odes,2) * 20

t = np.linspace(0,30,10000000)

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
                        Single Thread Solver
######################################################################"""

t0 = time.time()

solutions_single_thread = list()
for i in range(s):
    dxdt = lambda x,t : f(x, params_arr[i], t)
    solution = odeint(dxdt, x0_arr[i], t)
    solutions_single_thread.append(solution)

t1 = time.time()
print(f"      single thread total time: {round(1000*(t1-t0), 2)} ms")
print(f"single thread avg time per ode: {round(1000*(t1-t0)/s, 2)} ms")

"""######################################################################
                        Multithread Solver
######################################################################"""

t0 = time.time()

q = multiprocessing.Queue()
processes = []

for i in range(s):
    def single_solver(i, q):
        dxdt = lambda x,t : f(x, params_arr[i], t)
        solution = odeint(dxdt, x0_arr[i], t)
        q.put((i, solution))
    process = multiprocessing.Process(target=single_solver, args=(i, q))
    processes.append(process)
    process.start()

solutions_multithread = [q.get() for _ in range(s)]
for i in range(s):
    processes[i].join()

solutions_multithread_sorted = [None for _ in range(s)]
for i, solution in solutions_multithread:
    # print(i)
    solutions_multithread_sorted[i] = solution

t1 = time.time()

print(f"           parallel total time: {round(1000*(t1-t0), 2)} ms")

"""######################################################################
                        Evalution and Plots
######################################################################"""

# default_saturation = 1
# default_value = 1
# hues = [h.item() for h in np.linspace(0, .8, s)]
# colors = [colorsys.hsv_to_rgb(h, default_saturation, default_value) for h in hues]
#
# plt.figure()
# for i in range(s):
#     plt.plot(
#         solutions_single_thread[i][:, 0],
#         solutions_single_thread[i][:, 1],
#         color = colors[i],
#         linestyle = ":",
#         label=f"st: {i = }")
#
#     plt.plot(
#         solutions_multithread_sorted[i][:, 0],
#         solutions_multithread_sorted[i][:, 1],
#         color = colors[i],
#         linestyle = "-",
#         alpha = .5,
#         label=f"mt: {i = }")
#
# plt.show()
