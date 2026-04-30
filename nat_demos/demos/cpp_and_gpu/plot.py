import subprocess
import matplotlib.pyplot as plt

subprocess.run(["g++", "single_thread_BDF2.cpp"])
subprocess.run(["./a.out"])

with open("output.csv") as f:
    data = f.readlines()

t_arr = [float(x) for x in data[0].split(",")[1:]]
x1_arr = [float(x) for x in data[1].split(",")[1:]]
x2_arr = [float(x) for x in data[2].split(",")[1:]]

plt.plot(x1_arr, x2_arr)
plt.show()
