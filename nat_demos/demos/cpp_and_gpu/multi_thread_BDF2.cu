/************************************************************************
# to compile:
nvcc -ccbin gcc-12 multi_thread_BDF2.cu -o gpu_app.out

# to run:
./gpu_app.out
************************************************************************/

#include <cuda_runtime.h>
#include <stdio.h>

#define NEWTON_MAX_ITERS 5
#define NEWTON_TOLERANCE 1e-10
#define NUMERICAL_JACOBIAN_EPS 1e-6

int main() {
    return EXIT_SUCCESS;
}
