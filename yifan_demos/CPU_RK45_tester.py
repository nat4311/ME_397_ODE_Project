#!/bin/env python3
import numpy as np


def rk45(f, t0, tf, x0,
         rtol=1e-6,
         atol=1e-9,
         h0=1e-3,
         h_min=1e-12,
         h_max=1.0):
    """
    Adaptive RK45 (Dormand–Prince) solver
    CPU implementation
    """

    # Dormand–Prince coefficients
    c = np.array([0,
                  1/5,
                  3/10,
                  4/5,
                  8/9,
                  1,
                  1])

    a = [
        [],
        [1/5],
        [3/40, 9/40],
        [44/45, -56/15, 32/9],
        [19372/6561, -25360/2187, 64448/6561, -212/729],
        [9017/3168, -355/33, 46732/5247,
         49/176, -5103/18656],
        [35/384, 0, 500/1113, 125/192,
         -2187/6784, 11/84]
    ]

    # 5th order weights
    b5 = np.array([
        35/384,
        0,
        500/1113,
        125/192,
        -2187/6784,
        11/84,
        0
    ])

    # 4th order weights
    b4 = np.array([
        5179/57600,
        0,
        7571/16695,
        393/640,
        -92097/339200,
        187/2100,
        1/40
    ])

    t = t0
    x = x0.copy()
    h = h0

    t_values = [t]
    x_values = [x.copy()]

    while t < tf:

        if t + h > tf:
            h = tf - t

        k = []

        # Compute stages
        for i in range(7):
            if i == 0:
                xi = x
            else:
                xi = x.copy()
                for j in range(i):
                    xi += h * a[i][j] * k[j]

            ti = t + c[i] * h
            k.append(f(ti, xi))

        k = np.array(k)

        # 5th order solution
        x5 = x + h * np.sum(b5[:, None] * k, axis=0)

        # 4th order solution
        x4 = x + h * np.sum(b4[:, None] * k, axis=0)

        # Error estimate
        error = np.linalg.norm(x5 - x4, ord=np.inf)

        # Tolerance scaling
        tol = atol + rtol * max(
            np.linalg.norm(x, ord=np.inf),
            np.linalg.norm(x5, ord=np.inf)
        )

        if error <= tol:
            # Accept step
            t += h
            x = x5

            t_values.append(t)
            x_values.append(x.copy())

        # Step size control (safety factor 0.9)
        if error == 0:
            h_new = h_max
        else:
            h_new = 0.9 * h * (tol / error) ** 0.2

        h = min(max(h_new, h_min), h_max)

        if h < h_min:
            raise RuntimeError("Step size underflow")

    return np.array(t_values), np.array(x_values)

def f(t, x):
    return -2.0 * x

t, x = rk45(
    f,
    t0=0.0,
    tf=5.0,
    x0=np.array([1.0])
)

print(x[-1])  # Should be close to exp(-10)

import matplotlib.pyplot as plt
plt.plot(t, x[:, 0], label='RK45')
plt.legend()
plt.xlabel('Time')
plt.ylabel('x(t)')
plt.title('RK45 Solver for Stiff ODE')
plt.show()