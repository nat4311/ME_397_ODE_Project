"""
reference
https://maths.cnam.fr/IMG/pdf/RungeKuttaFehlbergProof.pdf
"""

import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import multiprocessing
import time
import colorsys
import subprocess
from copy import deepcopy
from helper import write_global_params

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

    def step(self, dxdt, x, t, h):
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
        z = x + self.c_hat[0]*k1 + self.c_hat[1]*k2 + self.c_hat[2]*k3 + self.c_hat[3]*k4 + self.c_hat[4]*k5 + self.c_hat[5]*k6

        return z, t+h, h

    def solve(self, dxdt, x0, t0, t_end, h):
        """
        return x_output, t_output, h_output

        x_output: 2D list of state values, x[i] = [x0[i], x1[i], x2[i], ...]
        t_output: 1D list of time values, t[i] = time at x[i]
        h_output: 1D list of timestep values used at each iteration
        """
        x_output = [x0]
        t_output = [t0]
        x = x0
        t = t0

        while t<t_end:
            h_prev = h
            x, t, h = self.step(dxdt, x, t, h)
            x_output.append([xi.item() for xi in x])
            t_output.append(t)
        return x_output, t_output

if __name__ == "__main__":

    """######################################################################
        user defines the ODE, params, initial conditions, and time bounds
    ######################################################################"""

    def f(x, p):
        """
        coupled van der pol oscillator with 10 states
        p[0:4] determine the stiffness μ
        p[5:9] determine the coupling
        Non-stiff: μ=1
        Moderately stiff: μ=5-20
        Very stiff: μ=50-200
        """
        xdot = np.zeros(10)

        xdot[0] = x[1] + p[5]*x[9]
        xdot[1] = p[0]*(1-x[0]**2)*x[1] - x[0]

        xdot[2] = x[3] + p[6]*x[1]
        xdot[3] = p[1]*(1-x[2]**2)*x[3] - x[2]

        xdot[4] = x[5] + p[7]*x[3]
        xdot[5] = p[2]*(1-x[4]**2)*x[5] - x[4]

        xdot[6] = x[7] + p[8]*x[5]
        xdot[7] = p[3]*(1-x[6]**2)*x[7] - x[6]

        xdot[8] = x[9] + p[9]*x[7]
        xdot[9] = p[4]*(1-x[8]**2)*x[9] - x[8]

        return xdot
    n = 10 # number of states - coupled to ODE def
    m = 10 # number of params - coupled to ODE def

    ###########################################################
    #                SET THESE TO TEST PERFORMANCE
    ###########################################################
    write_to_csv_flag = False
    run_odeint_comparison = False
    show_plots = False
    blockSize = 128
    s = 128*64*16 # number of odes to solve
    t0 = 0 # start time
    t_end = 10 # end time
    n_samples_per_ode = 100
    ###########################################################

    # initial conditions
    x0_arr = np.ones((s,n))

    # timestep values
    dt_arr = np.ones(s)*.001
    # for i in range(int(s/2)):
    #     dt_arr[2*i] *= .5

    # sampling to avoid running out of memory
    sample_period_arr = []
    for dt in dt_arr:
        total_samples = (t_end-t0)/dt.item()
        sample_period = total_samples / n_samples_per_ode
        sample_period_arr.append(sample_period)

    # more ode params - don't really matter
    mu_arr = np.logspace(-1,1,s)
    coupling_coeffs = 10*np.ones(5)
    p_arr = np.zeros((s,10))
    for ode_index in range(s):
        p_arr[ode_index,:5] = np.array([mu_arr[ode_index].item() for _ in range(5)])
        p_arr[ode_index,5:] = deepcopy(coupling_coeffs)

    # write global_params.csv
    try:
        os.remove("global_params.csv")
    except:
        pass
    write_global_params("blockSize", blockSize)
    write_global_params("n", n)
    write_global_params("m", m)
    write_global_params("s", s)
    write_global_params("t0", t0)
    write_global_params("t_end", t_end)
    write_global_params("dt_arr", [dt.item() for dt in dt_arr])
    write_global_params("sample_period_arr", sample_period_arr)
    for i in range(x0_arr.shape[0]):
        write_global_params(f"x0_arr_i{i}", [x.item() for x in x0_arr[i,:]])
    for i in range(p_arr.shape[0]):
        write_global_params(f"p_arr_i{i}", [p.item() for p in p_arr[i,:]])


    odeint_time = None
    cuda_time = None

    raise

    """######################################################################
                            Run the CUDA solver
    ######################################################################"""

    subprocess.run([
        "nvcc", "-ccbin", "gcc-12", "-std=c++11",
        "-Xcompiler", "-fPIC", "-lstdc++",
        "rk45_fixed.cu", "-o", "rk45_fixed.out", "-lm"
    ], check=True)

    if write_to_csv_flag or show_plots:
        timestamp = time.time()
        subprocess.run(["./rk45_fixed.out", "-wc"], check=True)
        cuda_time = time.time() - timestamp
        print(f"rk45_fixed CUDA kernel time + writing to csv total time: {round(1000*cuda_time, 2)} ms")

        # read results from file
        with open ("rk45_fixed_output.csv") as file:
            data = file.readlines()
        data = [[float(x) for x in line.split(",")[:-1]] for line in data]
        x_cuda = []
        t_cuda = []
        for i in range(s):
            t_cuda.append(data[(n+1)*i])
            x_cuda.append(np.array(data[(n+1)*i+1:(n+1)*i+1+n]).T)
    else:
        timestamp = time.time()
        subprocess.run(["./rk45_fixed.out"], check=True)
        cuda_time = time.time() - timestamp
        print(f"rk45_fixed CUDA kernel time: {round(1000*cuda_time, 2)} ms")

    if odeint_time is not None:
        ratio = odeint_time / cuda_time
        print(f"cuda code was {round(ratio, 1)}x faster")

    """######################################################################
                            Built in Solver
    ######################################################################"""

    if run_odeint_comparison or show_plots:
        timestamp = time.time()
        solutions_odeint = list()
        for i in range(s):
            t_odeint = np.linspace(t0,t_end,n_samples_per_ode+1)
            dxdt = lambda x,t : f(x, p_arr[i,:])
            solution = odeint(dxdt, x0_arr[i,:], t_odeint)
            solutions_odeint.append(solution)
        odeint_time = time.time() - timestamp
        print(f"scipy.integrate odeint total time: {round(1000*odeint_time, 2)} ms")

    """######################################################################
                            Evalution and Plots
    ######################################################################"""

    if show_plots:
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
                label=f"{round(p_arr[i,0],2)}",
            )

            plt.plot(
                x_cuda[i][:, 0],
                x_cuda[i][:, 1],
                color = colors[i],
                linestyle = "-",
                alpha = .5,
            )

        # plt.legend()
        plt.show()
