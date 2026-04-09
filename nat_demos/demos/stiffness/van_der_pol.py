"""
x'' - u(1-x^2)x' + x = 0

u >= 0 controls nonlinearity and damping

small u close to 0
    behaves like a weakly nonlinear oscillator with
    smooth limit cycles

large u >> 1
    enters a stiff regime with rapid "jumps" and slow
    "creep" phases, resembling relaxation oscillations

Why Stiff?
Large μ amplifies the negative damping for |x| < 1,
causing fast exponential growth, while positive damping
for |x| > 1 triggers quick decay. This creates disparate
eigenvalues in the Jacobian—one large (fast dynamics) and
one small (slow manifold)—demanding implicit solvers like
BDF methods for stability. Piecewise linear variants retain
positive stiffness but introduce bifurcations
(e.g., Hopf, saddle-node) tied to damping transitions.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

u = .13
def xdot(x, t):
    x1, x2 = x

    x1dot = x2
    x2dot = u*(1-x1**2)*x2 - x1

    return [x1dot, x2dot]

x0 = [20, 0]
t = np.linspace(0,30,1000)
sol = odeint(xdot, x0, t)

plt.figure()
plt.plot(sol[:, 0], sol[:, 1])
plt.show()
