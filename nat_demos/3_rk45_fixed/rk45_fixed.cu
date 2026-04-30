/************************************************************************
compilation commands:
nvcc -ccbin gcc-12 -std=c++11 -Xcompiler -fPIC -lstdc++ rk45_fixed.cu -o rk45_fixed.out -lm; ./rk45_fixed.out
nvcc -ccbin gcc-12 -std=c++11 -Xcompiler -fPIC -lstdc++ rk45_fixed.cu -o rk45_fixed.out -lm; ./rk45_fixed.out; cat rk45_fixed_output.csv

NOTE:
The Vec operator overloads do not perform vectorized ops on the GPU.
A kernel function would have to be made to do this.

************************************************************************/

#include <cmath>
#include <cassert>
#include <cuda_runtime.h>
#include <stdio.h>
#include <fstream>
#include "helper.h"

#define print(x) std::cout << #x << " = " << (x) << std::endl

/************************************************************************
                        Section: ODE FUNCTION
************************************************************************/

struct Vec10{
    double data[10];

    __host__ __device__ Vec10() {}
    __host__ __device__ Vec10(double* input_data) {
        for (int i=0; i<10; i++) {
            this->data[i] = input_data[i];
        }
    }

    // addition
    __host__ __device__ Vec10 operator+(const Vec10& b){
        Vec10 c;
        for (int i=0; i<10; i++) {
            c[i] = data[i] + b[i];
        }
        return c;
    }

    // scalar multiplication
    __host__ __device__ Vec10 operator*(double s) {
        Vec10 c;
        for (int i=0; i<10; i++) {
            c[i] = data[i] * s;
        }
        return c;
    }
    __host__ __device__ Vec10& operator*=(double s) {
        for (int i=0; i<10; i++) {
            data[i] *= s;
        }
        return *this;
    }

    // getter
    __host__ __device__ double operator[](int i) const{
        return data[i];
    }
    // setter
    __host__ __device__ double& operator[](int i){
        return data[i];
    }
};
__host__ __device__ Vec10 operator*(double s, const Vec10& v) { 
    Vec10 w;
    for (int i=0; i<10; i++) {
        w[i] = s*v[i];
    }
    return w;
}

const int n = 10;
const int m = 10;
__device__ Vec10 dxdt(Vec10 x, Vec10 p) {
    Vec10 xdot;

    xdot[0] = x[1] + p[5]*x[9];
    xdot[1] = p[0]*(1-x[0]*x[0])*x[1] - x[0];
    xdot[2] = x[3] + p[6]*x[1];
    xdot[3] = p[1]*(1-x[2]*x[2])*x[3] - x[2];
    xdot[4] = x[5] + p[7]*x[3];
    xdot[5] = p[2]*(1-x[4]*x[4])*x[5] - x[4];
    xdot[6] = x[7] + p[8]*x[5];
    xdot[7] = p[3]*(1-x[6]*x[6])*x[7] - x[6];
    xdot[8] = x[9] + p[9]*x[7];
    xdot[9] = p[4]*(1-x[8]*x[8])*x[9] - x[8];

    return xdot;
}

/************************************************************************
                        Section: GlobalParams
************************************************************************/

// load global_params.csv
struct GlobalParams {
    std::string filename;
    double h; // Timestep
    double t0; // Start time
    double t_end; // End time
    int s; // # of odes
    int n; // # of states
    int m; // # of params
    double* x0_arr; // initial state values - x is nx1, x0_arr is nxs
    double* p_arr; // p values - p is mx1, p_arr is mxs
    int n_timesteps;
    bool loaded_data = false;
    int blockSize;
    int gridSize;

    GlobalParams(std::string filename) : filename(filename){
        load_from_csv(filename);
    }

    void load_from_csv(std::string filename) {
        loaded_data = false;
        // open the file
        std::ifstream read_csv_file(filename);
        if (!read_csv_file) {
            std::cout << "Error: Unable to open global_params.csv" << std::endl;
            return;
        }

        // initialize check bools
        bool blockSize_check = false;
        bool h_check = false;
        bool t0_check = false;
        bool t_end_check = false;
        bool s_check = false;
        bool n_check = false;
        bool m_check = false;
        int x0_arr_check = 0;
        int p_arr_check = 0;

        // read the lines
        std::string line;
        while (std::getline(read_csv_file, line)) {
            DataLine dl(line);
            if (dl.name == "h") {
                h = dl.data[0];
                h_check = true;
            }
            else if (dl.name == "t0") {
                t0 = dl.data[0];
                t0_check = true;
            }
            else if (dl.name == "t_end") {
                t_end = dl.data[0];
                t_end_check = true;
            }
            else if (dl.name == "s") {
                s = dl.data[0];
                s_check = true;
            }
            else if (dl.name == "n") {
                n = dl.data[0];
                n_check = true;
            }
            else if (dl.name == "m") {
                m = dl.data[0];
                m_check = true;
            }
            else if (dl.name == "blockSize") {
                blockSize = dl.data[0];
                blockSize_check = true;
            }
            else if (dl.name.substr(0,6) == "x0_arr") {
                if (!(n_check && s_check)) {
                    std::cout << "n and s must be read before x0_arr\n";
                    return;
                }
                if (x0_arr_check == 0) {
                    // cudaMallocHost(&x0_arr, n*s*sizeof(double)); // todo: how to check this
                    if (cudaMallocHost(&x0_arr, n*s*sizeof(double)) != cudaSuccess) { // this leads to segfault
                        std::cout << "failed to malloc for x0_arr\n";
                        return;
                    }
                }

                int ode_index = std::stod(dl.name.substr(8, dl.name.size()-1));
                int block_start = ode_index*n;
                for (int i=0; i<n; i++) {
                    x0_arr[block_start + i] = dl.data[i];
                }
                x0_arr_check++;
            }
            else if (dl.name.substr(0,5) == "p_arr") {
                if (!(m_check && s_check)) {
                    std::cout << "m and s must be read before p_arr\n";
                    return;
                }
                if (p_arr_check == 0) {
                    // cudaMallocHost(&p_arr, m*s*sizeof(double));
                    if (cudaMallocHost(&p_arr, m*s*sizeof(double)) != cudaSuccess) {
                        std::cout << "failed to malloc for p_arr\n";
                        return;
                    }
                }

                int ode_index = std::stod(dl.name.substr(7, dl.name.size()-1));
                int block_start = ode_index*m;
                for (int i=0; i<n; i++) {
                    p_arr[block_start + i] = dl.data[i];
                }
                p_arr_check++;
            }
        }

        gridSize = int(ceil(double(s)/double(blockSize)));

        if (!blockSize_check) {
            std::cout << "kernel blockSize not read from global_params.csv\n";
            return;
        }
        if (!h_check) {
            std::cout << "h not read from global_params.csv\n";
            return;
        }
        if (!t0_check) {
            std::cout << "t0 not read from global_params.csv\n";
            return;
        }
        if (!t_end_check) {
            std::cout << "t_end not read from global_params.csv\n";
            return;
        }
        if (!s_check){
            std::cout << "s not read from global_params.csv\n";
            return;
        }
        if (!n_check) {
            std::cout << "n not read from global_params.csv\n";
            return;
        }
        if (!m_check) {
            std::cout << "m not read from global_params.csv\n";
            return;
        }
        if (x0_arr_check != s){
            std::cout << "x0_arr not fully read from global_params.csv\n";
            std::cout << "check = " << x0_arr_check << "\n";
            return;
        }
        if (p_arr_check != s){
            std::cout << "p_arr not fully read from global_params.csv\n";
            std::cout << "check = " << p_arr_check << "\n";
            return;
        }
        n_timesteps = int((t_end-t0)/h) + 1;

        loaded_data = true;
        std::cout << "global_params.csv parsed successfully\n";
    }
};

void print_GlobalParams(GlobalParams* gp, bool print_arrays = false) {
    std::cout << "==========================================\nGlobalParams: \n";
    std::cout << "h = " << gp->h << std::endl; 
    std::cout << "t0 = " << gp->t0 << std::endl; 
    std::cout << "t_end = " << gp->t_end << std::endl; 
    std::cout << "s = " << gp->s << std::endl; 
    std::cout << "n = " << gp->n << std::endl; 
    std::cout << "m = " << gp->m << std::endl; 
    std::cout << "n_timesteps = " << gp->n_timesteps << std::endl; 
    std::cout << "loaded_data = " << gp->loaded_data << std::endl; 

    if (print_arrays) {
        // x0_arr
        std::cout << "x0_arr = {\n";
        for (int ode_index=0; ode_index<gp->s; ode_index++) {
            std::cout << "    ";
            int block_start = ode_index*gp->n;
            for (int i=0; i<gp->n; i++) {
                std::cout << gp->x0_arr[block_start + i] << ", ";
            }
            std::cout << "\n";
        }
        std::cout << "}\n";

        // p_arr
        std::cout << "p_arr = {\n";
        for (int ode_index=0; ode_index<gp->s; ode_index++) {
            std::cout << "    ";
            int block_start = ode_index*gp->m;
            for (int i=0; i<gp->m; i++) {
                std::cout << gp->p_arr[block_start + i] << ", ";
            }
            std::cout << "\n";
        }
        std::cout << "}\n";
    }
}

// this looks like:
// t0 - for x0i
// t1 - for x1i
// t2 - for x2i
// ...
// x00
// x01
// x02
// ...
// x10
// x11
// x12
// ...
// x20
// x21
// x22
// ...
void write_data_csv(std::string filename, double* tsol, double* xsol, GlobalParams* gp) {
    std::ofstream file(filename);
    for (int ode_index=0; ode_index<gp->s; ode_index++) {
        // write time data
        int t_block_start = ode_index * gp->n_timesteps;
        for (int i=0; i<gp->n_timesteps; i++) {
            file << tsol[t_block_start+i] << ",";
        }
        file << "\n";

        // write x data
        int x_block_start = n * ode_index * gp->n_timesteps;
        for (int state_index=0; state_index<n; state_index++) {
            for (int i=0; i<gp->n_timesteps; i++) {
                file << xsol[x_block_start + n*i + state_index] << ",";
            }
            file << "\n";
        }
    }
}

/************************************************************************
                        Section: SOLVER KERNEL                   
************************************************************************/

// Butcher table values
constexpr double B21 =      2.0/9.0;
constexpr double B31 =     1.0/12.0;
constexpr double B32 =      1.0/4.0;
constexpr double B41 =   69.0/128.0;
constexpr double B42 = -243.0/128.0;
constexpr double B43 =   135.0/64.0;
constexpr double B51 =   -17.0/12.0;
constexpr double B52 =     27.0/4.0;
constexpr double B53 =    -27.0/5.0;
constexpr double B54 =    16.0/15.0;
constexpr double B61 =   65.0/432.0;
constexpr double B62 =    -5.0/16.0;
constexpr double B63 =    13.0/16.0;
constexpr double B64 =     4.0/27.0;
constexpr double B65 =    5.0/144.0;
constexpr double C1 = 47.0/450.0;
constexpr double C2 =        0.0;
constexpr double C3 =  12.0/25.0;
constexpr double C4 = 32.0/225.0;
constexpr double C5 =   1.0/30.0;
constexpr double C6 =   6.0/25.0;

// write to the GPU 
__device__ void write_x(double* xsol_GPU, Vec10 x, int ode_index, int i, int n_timesteps) {
    for (int state_index=0; state_index<n; state_index++) {
        xsol_GPU[ode_index*n*n_timesteps + n*i + state_index] = x[state_index];
    }
}
__device__ void write_t(double* tsol_GPU, double t, int ode_index, int i, int n_timesteps) {
    tsol_GPU[ode_index*n_timesteps + i] = t;
}

__global__ void rk45_fixed(double* xsol_GPU, double* tsol_GPU, double* x0_arr_GPU, double* p_arr_GPU, double t0, int n_timesteps, double h, int s) {
    int ode_index = threadIdx.x + blockDim.x * blockIdx.x;
    if (ode_index >= s) {
        return;
    }
    Vec10 x(x0_arr_GPU+ode_index*n);
    Vec10 p(p_arr_GPU + ode_index*m);
    double t = t0;
    Vec10 k1, k2, k3, k4, k5, k6;

    // write initial data
    write_x(xsol_GPU, x, ode_index, 0, n_timesteps);
    write_t(tsol_GPU, t, ode_index, 0, n_timesteps);

    for (int i=1; i<n_timesteps; i++) {
        // step
        k1 = h * dxdt(x, p);
        k2 = h * dxdt(x + B21*k1, p);
        k3 = h * dxdt(x + B31*k1 + B32*k2, p);
        k4 = h * dxdt(x + B41*k1 + B42*k2 + B43*k3, p);
        k5 = h * dxdt(x + B51*k1 + B52*k2 + B53*k3 + B54*k4, p);
        k6 = h * dxdt(x + B61*k1 + B62*k2 + B63*k3 + B64*k4 + B65*k5, p);
        x = x + C1*k1 + C2*k2 + C3*k3 + C4*k4 + C5*k5 + C6*k6;
        t += h;

        // write data
        write_x(xsol_GPU, x, ode_index, i, n_timesteps);
        write_t(tsol_GPU, t, ode_index, i, n_timesteps);
    }
}

/************************************************************************
                        Section: MAIN
************************************************************************/

int main(int argc, char* argv[]) {

    bool write_to_csv_flag = false;
    if (argc > 1) {
        if (std::string(argv[1]) == "-wc") {
            write_to_csv_flag = true;
        }
    }

    // read global_params
    GlobalParams gp("global_params.csv");
    if (!gp.loaded_data) {
        std::cout << "unable to load global params\n";
        print_GlobalParams(&gp);
        return EXIT_FAILURE;
    }
    assert (n==gp.n);
    assert (m==gp.m);

    // init memory pointers
    double* xsol_GPU = nullptr;
    double* tsol_GPU = nullptr;
    double* xsol = nullptr;
    double* tsol = nullptr;
    double* x0_arr_GPU = nullptr;
    double* p_arr_GPU = nullptr;

    // allocate memory on cpu and gpu
    const size_t xsol_size = gp.s * n*gp.n_timesteps * sizeof(double);
    const size_t tsol_size = gp.s * gp.n_timesteps * sizeof(double);
    const size_t x0_arr_size = n*gp.s * sizeof(double);
    const size_t p_arr_size = m*gp.s * sizeof(double);
    cudaMallocHost(&xsol, xsol_size);
    cudaMallocHost(&tsol, tsol_size);
    if (cudaMalloc(&xsol_GPU, xsol_size) != cudaSuccess) {
        std::cout << "failed to malloc for xsol_GPU\n";
        return EXIT_FAILURE;
    }
    if (cudaMalloc(&tsol_GPU, tsol_size) != cudaSuccess) {
        std::cout << "failed to malloc for tsol_GPU\n";
        return EXIT_FAILURE;
    }
    if (cudaMalloc(&x0_arr_GPU, x0_arr_size) != cudaSuccess) {
        std::cout << "failed to malloc for x0_arr_GPU\n";
        return EXIT_FAILURE;
    }
    if (cudaMalloc(&p_arr_GPU, p_arr_size) != cudaSuccess) {
        std::cout << "failed to malloc for p_arr_GPU\n";
        return EXIT_FAILURE;
    }

    // copy x0_arr, p_arr, gp to GPU
    cudaMemcpy(x0_arr_GPU, gp.x0_arr, x0_arr_size, cudaMemcpyHostToDevice);
    cudaMemcpy(p_arr_GPU, gp.p_arr, p_arr_size, cudaMemcpyHostToDevice);

    // run rk45 on GPU
    std::cout << "running rk45_fixed...\n";
    std::cout << "launching kernel of size <<<" << gp.gridSize << "," << gp.blockSize << ">>>" << std::endl;
    rk45_fixed<<<gp.gridSize, gp.blockSize>>>(xsol_GPU, tsol_GPU, x0_arr_GPU, p_arr_GPU, gp.t0, gp.n_timesteps, gp.h, gp.s);
    cudaDeviceSynchronize();

    // copy results back to CPU
    std::cout << "copying results back to CPU...\n";
    cudaMemcpy(xsol, xsol_GPU, n*gp.s * gp.n_timesteps * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(tsol, tsol_GPU, gp.s * gp.n_timesteps * sizeof(double), cudaMemcpyDeviceToHost);

    if (write_to_csv_flag) {
        std::cout << "writing csv...\n";
        write_data_csv("rk45_fixed_output.csv", tsol, xsol, &gp);

        // free the memory
        cudaFreeHost(xsol);
        cudaFreeHost(tsol);
        cudaFree(xsol_GPU);
        cudaFree(tsol_GPU);
    }

    std::cout << "rk45_fixed.cu success\n";
    return 0;
}
