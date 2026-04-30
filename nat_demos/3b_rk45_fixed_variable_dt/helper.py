def write_global_params(name, data):
    with open("global_params.csv", "a") as f:
        f.writelines([name, ","])
        if type(data) is list:
            f.writelines([f"{d}," for d in data])
        else:
            f.writelines([f"{data},"])
        f.writelines("\n")

if __name__ == "__main__":
    import os
    import subprocess
    from copy import deepcopy
    import numpy as np

    n = 10 # number of states - coupled to ODE def
    m = 10 # number of params - coupled to ODE def

    s = 2 # number of odes to solve
    h = .1 # step size - todo: make different per ode
    t0 = 0 # start time
    t_end = 1 # end time

    # initial conditions
    x0_arr = np.array([[2 for state_index in range(n)] for ode_index in range(s)])
    # mu_arr = np.logspace(-1,2.3,s) # .1 to 200
    mu_arr = np.logspace(0,1,s) # remove
    coupling_coeffs = 0*np.array([0.01, 0.01, 0.01, 0.01, 0.01])
    p_arr = np.zeros((s,10))
    for state_index in range(s):
        p_arr[state_index,:5] = np.array([mu_arr[state_index].item() for _ in range(5)])
        p_arr[state_index,5:] = deepcopy(coupling_coeffs)


    try:
        os.remove("global_params.csv")
    except:
        pass
    write_global_params("n", n)
    write_global_params("m", m)
    write_global_params("s", s)
    write_global_params("h", h)
    write_global_params("t0", t0)
    write_global_params("t_end", t_end)
    for i in range(x0_arr.shape[0]):
        write_global_params(f"x0_arr_i{i}", [x.item() for x in x0_arr[i,:]])
    for i in range(p_arr.shape[0]):
        write_global_params(f"p_arr_i{i}", [p.item() for p in p_arr[i,:]])
    subprocess.run(["cat", "global_params.csv"])
