# 3_rk45_fixed/

CUDA implementation of rk45 with fixed timestep across all ODEs

to compile: 
    nvcc -ccbin gcc-12 -std=c++11 -Xcompiler -fPIC -lstdc++ rk45_fixed.cu -o rk45_fixed.out -lm

to run:
    ./rk45_fixed.out

# 3b_rk45_fixed_variable_dt/

CUDA implementation of rk45 with variable timestep across all ODEs

to compile: 
    nvcc -ccbin gcc-12 -std=c++11 -Xcompiler -fPIC -lstdc++ rk45_fixed.cu -o rk45_fixed.out -lm
to run:
    ./rk45_fixed.out
