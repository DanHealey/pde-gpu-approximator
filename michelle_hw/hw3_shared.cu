#include <stdio.h>
#include <math.h>
#include <chrono>
#include <algorithm>
#include <iostream>

#define USE_HIP
#ifdef USE_HIP
#include <hip/hip_runtime.h>
#define cudaGetDeviceCount      hipGetDeviceCount
#define cudaSetDevice           hipSetDevice
#define cudaDeviceSynchronize   hipDeviceSynchronize
#define cudaMalloc              hipMalloc 
#define cudaHostMalloc          hipHostMalloc
#define cudaFree                hipFree
#define cudaFreeHost            hipFreeHost
#define cudaMemcpy              hipMemcpy
#define cudaMemset              hipMemset
#define cudaMemcpyAsync         hipMemcpyAsync
#define cudaMemcpyHostToDevice  hipMemcpyHostToDevice
#define cudaMemcpyDeviceToDevice hipMemcpyDeviceToDevice
#define cudaMemcpyDeviceToHost  hipMemcpyDeviceToHost
#define cudaStreamSynchronize   hipStreamSynchronize
#define cudaStreamCreate        hipStreamCreate
#define cudaStreamDestroy       hipStreamDestroy
#define cudaStream_t            hipStream_t
#define cudaError_t             hipError_t
#define cudaEvent_t             hipEvent_t
#define cudaEventCreate         hipEventCreate
#define cudaEventRecord         hipEventRecord
#define cudaEventElapsedTime    hipEventElapsedTime
#else
#include <cuda_runtime.h>
#endif

#define NUM_TRIALS 5

/*
Task: Write an iterative solver to find the value of u(x,y) using the
shared memory model and offloading to GPU

Question:
Solve 2D differential equation on domain [0,1] x [0,1]; 2nd order diff eq: ∂2u/∂x2 + ∂2u/∂y2 = f(x,y)
Finite difference scheme:
    ∂2u/∂x2 = (u[i-1,j]-2*u[i,j]+u[i+1,j] ) / (Dx*Dx)
    ∂2u/∂y2 = (u[i,j-1]-2*u[i,j]+u[i,j+1] ) / (Dy*Dy)

Solve for solution given: f(x,y) = - 2*(2p)*(2p) sin(2px) * cos(2py) (actual: U = sin(2px) * cos(2py))
Algorithm: (dx=dy=h=1/(N-1))
    f(x,y) = (u[i-1,j]-2*u[i,j]+u[i+1,j] + u[i,j-1]-2*u[i,j]+u[i,j+1]) / h^2
    u[i,j] = (sum of 4 neighbors - f(x,y)*h^2) / 4
*/

double pi = 4 * atan(1.0);
double actual(double x, double y) {
    return sin(2 * pi * x) * cos(2 * pi * y);
}
double f(double x, double y) {
    return -2 * (2 * pi) * (2 * pi) * sin(2 * pi * x) * cos(2 * pi * y);
}

/**
 * @brief Set the boundary conditions on edges for a 2D grid of size NxN, using actual solution function.
 */
void generate_boundaries(double* mat, size_t N, double h, double x0, double x1, double y0, double y1) {
    for (size_t i = 0; i < N; i++) {
        mat[0 * N + i] = actual(x0, y0 + i * h);
        mat[(N - 1) * N + i] = actual(x1, y0 + i * h);
        mat[i * N + 0] = actual(x0 + i * h, y0);
        mat[i * N + (N - 1)] = actual(x0 + i * h, y1);
    }
}

/**
 * @brief CUDA kernel to update U and compute local maximum difference.
 */
__global__
void update_U(double* U_d, double* U_old_d, double* f_U_d, int N, double h, double* max_diff_per_block_d) {
    // Global indices (start from 1)
    int i = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int j = blockIdx.x * blockDim.x + threadIdx.x + 1;

    // Index in shared memory
    int tid = threadIdx.y * blockDim.x + threadIdx.x;

    // Shared memory for differences
    extern __shared__ double s_diff[];

    double diff = 0.0;

    if (i < N - 1 && j < N - 1) {
        double U_new = (U_old_d[(i + 1) * N + j] + U_old_d[(i - 1) * N + j] + U_old_d[i * N + (j + 1)] + U_old_d[i * N + (j - 1)] - f_U_d[i * N + j] * h * h) / 4.0;
        diff = fabs(U_new - U_old_d[i * N + j]);
        U_d[i * N + j] = U_new;
    }
    s_diff[tid] = diff;
    __syncthreads();

    // Reduction to find max difference
    int blockSize = blockDim.x * blockDim.y;
    for (int stride = blockSize / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            s_diff[tid] = fmax(s_diff[tid], s_diff[tid + stride]);
        }
        __syncthreads();
    }

    // Write max difference from block to global memory
    if (tid == 0) {
        max_diff_per_block_d[blockIdx.y * gridDim.x + blockIdx.x] = s_diff[0];
    }
}

/**
 * @brief Solves the equation using second order finite difference on the GPU.
 */
double second_finite_difference(size_t N, double error_threshold = 1e-6) {
    const double x0 = 0;
    const double x1 = 1;
    const double y0 = 0;
    const double y1 = 1;
    const double h = (x1 - x0) / (N - 1);

    double* U_actual = new double[N * N];
    double* f_U = new double[N * N];
    double* U = new double[N * N];
    double* U_old = new double[N * N];

    // Define grid and block dimensions
    int block_size = 16;
    dim3 blockDim(block_size, block_size);
    dim3 gridDim((N - 2 + blockDim.x - 1) / blockDim.x, (N - 2 + blockDim.y - 1) / blockDim.y);
    int num_blocks = gridDim.x * gridDim.y;

    double *max_diff_per_block_h = new double[num_blocks];

    // Allocate device memory
    double *U_d, *U_old_d, *f_U_d;
    cudaMalloc(&U_d, N * N * sizeof(double));
    cudaMalloc(&U_old_d, N * N * sizeof(double));
    cudaMalloc(&f_U_d, N * N * sizeof(double));

    // Allocate memory for maximum differences per block
    double *max_diff_per_block_d;
    cudaMalloc(&max_diff_per_block_d, num_blocks * sizeof(double));

    double total_time = 0;
    double copy_time = 0;
    size_t c = 0;

    size_t shared_mem_size = blockDim.x * blockDim.y * sizeof(double);

    for (int t = 0; t < NUM_TRIALS; t++) {

        // Initialize U_actual and f_U
        for (size_t i = 0; i < N; i++) {
            for (size_t j = 0; j < N; j++) {
                double x = x0 + i * h;
                double y = y0 + j * h;
                U_actual[i * N + j] = actual(x, y);
                f_U[i * N + j] = f(x, y);
                U_old[i * N + j] = 0.0;
                U[i * N + j] = 0.0;
            }
        }

        // Initialize U_old and U with boundary conditions
        generate_boundaries(U_old, N, h, x0, x1, y0, y1);
        generate_boundaries(U, N, h, x0, x1, y0, y1);

        // Copy f_U to device
        cudaMemcpy(f_U_d, f_U, N * N * sizeof(double), cudaMemcpyHostToDevice);

        // Copy U_old to U_old_d
        cudaMemcpy(U_old_d, U_old, N * N * sizeof(double), cudaMemcpyHostToDevice);

        // Initialize U_d with U_old_d
        cudaMemcpy(U_d, U_old_d, N * N * sizeof(double), cudaMemcpyDeviceToDevice);

        double error = 1.0;

        // Start timer
        auto start = std::chrono::high_resolution_clock::now();

        while (error > error_threshold) {
            // Set max_diff_per_block_d to zero
            // cudaMemset(max_diff_per_block_d, 0, num_blocks * sizeof(double));

            // Launch the kernel
            update_U<<<gridDim, blockDim, shared_mem_size>>>(U_d, U_old_d, f_U_d, N, h, max_diff_per_block_d);

            #ifdef TIMED
            cudaDeviceSynchronize();
            #endif

            // Copy max_diff_per_block_d to host
            auto copy_start = std::chrono::high_resolution_clock::now();
            cudaMemcpy(max_diff_per_block_h, max_diff_per_block_d, num_blocks * sizeof(double), cudaMemcpyDeviceToHost);
            auto copy_end = std::chrono::high_resolution_clock::now();

            copy_time += std::chrono::duration_cast<std::chrono::nanoseconds>(copy_end - copy_start).count();

            // Reduce to find maximum error
            double local_error = 0.0;
            for (int i = 0; i < num_blocks; i++) {
                local_error = std::max(local_error, max_diff_per_block_h[i]);
            }
            error = local_error;

            // Swap U_d and U_old_d pointers
            std::swap(U_d, U_old_d);

            c++;
        }

        auto end = std::chrono::high_resolution_clock::now();

        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
        total_time += duration.count() * 1e-9;
        //std::cout << duration.count() << " " << c << std::endl;

    }

    // convert to seconds
    copy_time = copy_time * 1e-9;

    double num_bytes = ((6 + 2) * (N - 2) * (N - 2)) * 8 * (c / NUM_TRIALS);
    double num_flop = (1 + 7) * (N - 2) * (N - 2) * (c / NUM_TRIALS);

    double time = total_time / NUM_TRIALS;
    double no_copy = (total_time - copy_time) / NUM_TRIALS;

    printf("Mean time taken: %lf ms\n", total_time * 1e3 / NUM_TRIALS);
    printf("Mean iterations: %d\n", (int) ((double) c / NUM_TRIALS));
    printf("Mean time per iteration: %lf (µs)\n", total_time * 1e6 / c );
    printf("%lf GFLOP/s, %lf GB/s\n", num_flop * 1e-9 / time, num_bytes * 1e-9 / time);
    #ifdef TIMED
    printf("Mean time copying between host and device per iteration: %lf (µs)\n", copy_time * 1e6 / c);
    printf("Mean time per iteration without copying: %lf (µs)\n", (total_time - copy_time) * 1e6 / c);
    printf("(without copying) %lf GFLOP/s, %lf GB/s\n", num_flop * 1e-9 / no_copy, num_bytes * 1e-9 / no_copy);
    #endif
    
    // Copy U_old_d (latest U values) to U
    cudaMemcpy(U, U_old_d, N * N * sizeof(double), cudaMemcpyDeviceToHost);

    // Compute true error (MSE) between U and U_actual
    double true_error = 0;
    for (size_t i = 1; i < N - 1; i++) {
        for (size_t j = 1; j < N - 1; j++) {
            true_error += pow(U_actual[i * N + j] - U[i * N + j], 2);
        }
    }
    true_error = true_error / ((N - 2) * (N - 2));

    // Free device memory
    cudaFree(U_d);
    cudaFree(U_old_d);
    cudaFree(f_U_d);
    cudaFree(max_diff_per_block_d);

    // Free host memory
    delete[] U_actual;
    delete[] f_U;
    delete[] U;
    delete[] U_old;
    delete[] max_diff_per_block_h;

    return true_error;
}

int main(int argc, char **argv) {
    size_t Ns[] = {10, 100, 200, 500, 700, 1000};
    double error;

    for (auto N : Ns) {
        std::cout << "N: " << N << std::endl;
        error = second_finite_difference(N, 1e-6);
        std::cout << "Mean squared error: " << error << std::endl;
        std::cout << std::endl;
    }

    return 0;
}
