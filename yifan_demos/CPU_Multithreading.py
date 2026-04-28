#!/usr/bin/env python3
import numpy as np
import time
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
import os

# -----------------------------
# RK45 (same as before, compact)
# -----------------------------
def rk45(f, t0, tf, x0,
         rtol=1e-6, atol=1e-9,
         h0=1e-3):

    t, x, h = t0, x0.copy(), h0
    c = [0,1/5,3/10,4/5,8/9,1,1]
    a = [
        [], [1/5],
        [3/40,9/40],
        [44/45,-56/15,32/9],
        [19372/6561,-25360/2187,64448/6561,-212/729],
        [9017/3168,-355/33,46732/5247,49/176,-5103/18656],
        [35/384,0,500/1113,125/192,-2187/6784,11/84]
    ]

    b5 = np.array([35/384,0,500/1113,125/192,-2187/6784,11/84,0])
    b4 = np.array([5179/57600,0,7571/16695,393/640,-92097/339200,187/2100,1/40])

    while t < tf:
        if t + h > tf:
            h = tf - t

        k = []
        for i in range(7):
            xi = x.copy()
            for j in range(i):
                xi += h * a[i][j] * k[j]
            k.append(f(t + c[i]*h, xi))

        k = np.array(k)

        x5 = x + h * np.sum(b5[:,None]*k, axis=0)
        x4 = x + h * np.sum(b4[:,None]*k, axis=0)

        err = np.linalg.norm(x5 - x4, np.inf)
        tol = atol + rtol * max(np.linalg.norm(x,np.inf), np.linalg.norm(x5,np.inf))

        if err <= tol:
            t += h
            x = x5

        h = 0.9*h*(tol/(err+1e-16))**0.2

    return x


# -----------------------------
# ODE
# -----------------------------
def f(t, x):
    return -2.0 * x


# -----------------------------
# Worker
# -----------------------------
def solve_one(x0):
    return rk45(f, 0.0, 5.0, np.array([x0]))


# -----------------------------
# Benchmark
# -----------------------------
def benchmark():

    cpu_count = os.cpu_count()
    print(f"CPU cores: {cpu_count}")

    sizes = [10, 50, 100, 500, 1000]

    serial_times = []
    parallel_times = []

    for N in sizes:
        x0s = np.linspace(1, 10, N)

        # --- Serial ---
        start = time.time()
        results_serial = [solve_one(x0) for x0 in x0s]
        t_serial = time.time() - start

        # --- Parallel ---
        start = time.time()
        with ProcessPoolExecutor(max_workers=cpu_count) as ex:
            results_parallel = list(ex.map(solve_one, x0s))
        t_parallel = time.time() - start

        serial_times.append(t_serial)
        parallel_times.append(t_parallel)

        print(f"N={N:5d} | Serial: {t_serial:.4f}s | Parallel: {t_parallel:.4f}s")

    return sizes, serial_times, parallel_times


# -----------------------------
# Run benchmark
# -----------------------------
sizes, serial, parallel = benchmark()

# -----------------------------
# Plot (BIG FONT, REPORT READY)
# -----------------------------
plt.figure(figsize=(10, 6))

plt.plot(sizes, serial, marker='o', linewidth=2, label='Serial')
plt.plot(sizes, parallel, marker='s', linewidth=2, label='Parallel (CPU)')

plt.xlabel('Number of Trajectories', fontsize=18)
plt.ylabel('Execution Time (s)', fontsize=18)
plt.title('RK45 Parallel Scaling (Multiprocessing)', fontsize=20)

plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

plt.legend(fontsize=14)
plt.grid(True)

plt.tight_layout()
plt.show()