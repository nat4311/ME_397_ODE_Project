/************************************************************************
nvcc -ccbin gcc-12 rk45_fixed.cu -o rk45_fixed; ./rk45_fixed
************************************************************************/

#include <cuda_runtime.h>
#include <stdio.h>

/************************************************************************
                    user defines ODE
************************************************************************/

struct Vec2{
    double x, y;
    __host__ __device__ Vec2(double x, double y) : x(x), y(y) {}
    __host__ __device__ Vec2 operator+(const Vec2& other){ return {x+other.x, y+other.y}; }
    __host__ __device__ Vec2 operator*(double s){ return {s*x, s*y}; }
    __host__ __device__ double operator()(int i) const{ return (i==0)? x : y; } // getter
    __host__ __device__ double& operator()(int i){ return (i==0)? x : y; } // setter
};

const double h = .01;
const double t0 = 0.0;
const double t_end = 10.0;
const int n_timesteps = int((t_end-t0)/h);

// arrays are flattened for transfer
const int s = 4;
const int n = 2; // x is nxs
const int m = 1; // p is mxs
const double x0_arr[n][s] = {
    .1,.2,
    .1,.2,
    .1,.2,
    .1,.2,
};
const double p_arr[m][s] = {
    .1,
    .2,
    .3,
    .4
};

/************************************************************************
                        SOLVER KERNEL                   
************************************************************************/

// Butcher Table Values
const double A1 = 0.0;
const double A2 = 2.0/9.0;
const double A3 = 1.0/3.0;
const double A4 = 3.0/4.0;
const double A5 = 1.0;
const double A6 = 5.0/6.0;
const double B21 =    2.0/9.0;
const double B31 =   1.0/12.0; const double B32 =      1.0/4.0;
const double B41 = 69.0/128.0; const double B42 = -243.0/128.0; const double B43 = 135.0/64.0;
const double B51 = -17.0/12.0; const double B52 =     27.0/4.0; const double B53 =  -27.0/5.0; const double B54 = 16.0/15.0;
const double B61 = 65.0/432.0; const double B62 =    -5.0/16.0; const double B63 =  13.0/16.0; const double B64 =  4.0/27.0; const double B65 = 5.0/144.0;
const double C1 = 47.0/450.0;
const double C2 = 0.0;
const double C3 = 12.0/25.0;
const double C4 = 32.0/225.0;
const double C5 = 1.0/30.0;
const double C6 = 6.0/25.0;

// #define write_x(x_output_GPU, x, tid, i) x_output_GPU[tid*n*n_timesteps+n*i]=x.x; x_output_GPU[tid*n*n_timesteps+n*i+1]=x.y;
// #define write_t(t_output_GPU, t, tid, i) t_output_GPU[tid*n_timesteps+i] = t;

__device__ void write_x(double* x_output_GPU, Vec2* x, int tid, int i) {
    x_output_GPU[tid*n*n_timesteps+n*i]=x->x;
    x_output_GPU[tid*n*n_timesteps+n*i+1]=x->y;
}
__device__ void write_t(double* t_output_GPU, double t, int tid, int i) {
    t_output_GPU[tid*n_timesteps+i] = t;
}

__global__ void rk45_fixed(double* x_output_GPU, double* t_output_GPU) {
    // get tid - todo: figure out optimal thread/block sizes
    int tid = threadIdx.x + blockDim.x * blockIdx.x; // range: [0,s-1]

    // unflatten values - todo: make this more general (currently depends on user ODE)
    Vec2 x(x0_arr[tid][0], x0_arr[tid][1]);
    double p = p_arr[tid][0];
    double t = t0;
    double k1, k2, k3, k4, k5, k6;

    // write initial data
    write_x(x_output_GPU, &x, tid, 0);
    write_t(t_output_GPU, t, tid, 0);

    for (int i=1; i<=n_timesteps; i++) {
        // step
        k1 = h;

        // write data
        write_x(x_output_GPU, &x, tid, i);
        write_t(t_output_GPU, t, tid, i);
    }

}

int main() {

    // init memory pointers
    double* x_output_GPU = nullptr;
    double* t_output_GPU = nullptr;
    double* x_output = nullptr;
    double* t_output = nullptr;
    double* x0_arr_GPU = nullptr;
    double* p_arr_GPU = nullptr;

    // allocate memory on cpu and gpu
    cudaMalloc(&x_output_GPU, n*s * (n_timesteps+1) * sizeof(double));
    cudaMallocHost(&x_output, n*s * (n_timesteps+1) * sizeof(double));
    cudaMalloc(&t_output_GPU, s * (n_timesteps+1) * sizeof(double));
    cudaMallocHost(&t_output, s * (n_timesteps+1) * sizeof(double));
    cudaMalloc(&x0_arr_GPU, n*s * sizeof(double));
    cudaMalloc(&p_arr_GPU, m*s * sizeof(double));

    // run rk45 on GPU
    rk45_fixed<<<1,s>>>(x_output_GPU, t_output_GPU);
    cudaDeviceSynchronize();

    // copy results back to CPU
    cudaMemcpy(x_output, x_output_GPU, n*s * n_timesteps * sizeof(double), cudaMemcpyDefault);
    cudaMemcpy(t_output, t_output_GPU, n_timesteps * sizeof(double), cudaMemcpyDefault);

    // free the memory
    cudaFreeHost(x_output);
    cudaFreeHost(t_output);
    cudaFree(x_output_GPU);
    cudaFree(t_output_GPU);

    return EXIT_SUCCESS;
}
