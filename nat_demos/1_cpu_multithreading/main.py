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
    if p[0] < .16:
        time.sleep(.001)
    x1, x2 = x

    x1dot = x2
    x2dot = p[0]*(1-x1**2)*x2 - x1

    return [x1dot, x2dot]

params_arr = np.array([
    [.13],
    [.15],
    [.17],
    [.2],
])

x0_arr = np.array([
    [20,0],
    [18,0],
    [17,0],
    [15,0],
])

t = np.linspace(0,30,1000)

"""######################################################################
                        Validate user input
######################################################################"""

assert x0_arr.shape[0] == params_arr.shape[0]
s = x0_arr.shape[0]
try:
    f(x0_arr[0], params_arr[0], 0)
except Exception as e:
    print(e)

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
print(f"single thread avg time per ode: {round(1000*(t1-t0)/2, 2)} ms")

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

for i in range(s):
    processes[i].join()

solutions_multithread = [q.get() for _ in range(s)]
solutions_multithread_sorted = [None for _ in range(s)]
for i, solution in solutions_multithread:
    print(i)
    solutions_multithread_sorted[i] = solution

t1 = time.time()

print(f"           parallel total time: {round(1000*(t1-t0), 2)} ms")

"""######################################################################
                        Evalution and Plots
######################################################################"""
saturation = 1
value = 1
hues = [h.item() for h in np.linspace(0, .8, s)]
colors = [colorsys.hsv_to_rgb(h, saturation, value) for h in hues]

plt.figure()
for i in range(s):
    plt.plot(
        solutions_single_thread[i][:, 0],
        solutions_single_thread[i][:, 1],
        color = colors[i],
        linestyle = ":",
        label=f"st: {i = }")

    plt.plot(
        solutions_multithread_sorted[i][:, 0],
        solutions_multithread_sorted[i][:, 1],
        color = colors[i],
        linestyle = "-",
        alpha = .5,
        label=f"mt: {i = }")

plt.legend()
plt.show()
