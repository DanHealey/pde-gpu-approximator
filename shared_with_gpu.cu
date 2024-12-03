#include <hip/hip_runtime.h>
#include <stdio.h>
#include <cmath>
#include <iostream>
#include <algorithm>
#include <sys/time.h>

__device__ double pi = 3.14159265358979323846;
__device__ constexpr double n1 = 2;
__device__ constexpr double m1 = 2;
__device__ constexpr double k1 = 2;

long long time_diff(struct timeval start, struct timeval end) {
    return (end.tv_sec - start.tv_sec) * 1000000LL + (end.tv_usec - start.tv_usec);
}

__device__ double exact_phi(double x, double y, double z) {
    return sin(n1 * pi * x) * cos(m1 * pi * y) * sin(k1 * pi * z);
}

__device__ double f(double x, double y, double z) {
    return -(k1 * k1 + m1 * m1 + n1 * n1) * (pi * pi) * exact_phi(x, y, z);
}

__global__ void initialize(double* phi, double* phi_old, double* phi_actual, double* f_phi, size_t N, double h) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    size_t j = blockIdx.y * blockDim.y + threadIdx.y;
    size_t k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i >= 0 && j>=0 && k>=0 && i < N && j < N && k < N) {
        double x = i * h;
        double y = j * h;
        double z = k * h;
        size_t idx = i + j * N + k * N * N;

        phi[idx] = 0.0;
        phi_old[idx] = 0.0;
        phi_actual[idx] = exact_phi(x, y, z);
        f_phi[idx] = f(x, y, z);

        if (i == 0 || i == N - 1 || j == 0 || j == N - 1 || k == 0 || k == N - 1) {
            phi[idx] = exact_phi(x, y, z);
            phi_old[idx] = phi[idx];
        }
    }
}

__global__ void update_phi(double* phi, double* phi_old, double* f_phi, size_t N, double h) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i > 0 && i < N - 1 && j > 0 && j < N - 1 && k > 0 && k < N - 1) {
        size_t idx = i + j * N + k * N * N;
        phi[idx] = (
            phi_old[(i - 1) + j * N + k * N * N] +
            phi_old[(i + 1) + j * N + k * N * N] +
            phi_old[i + (j - 1) * N + k * N * N] +
            phi_old[i + (j + 1) * N + k * N * N] +
            phi_old[i + j * N + (k - 1) * N * N] +
            phi_old[i + j * N + (k + 1) * N * N] -
            f_phi[idx] * (h * h)
        ) / 6.0;
    }
}

__global__ void compute_error_and_convergence(double* phi, double* phi_old, double* phi_actual, size_t N, double* error, double *conv) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    size_t j = blockIdx.y * blockDim.y + threadIdx.y;
    size_t k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i > 0 && i < N - 1 && j > 0 && j < N - 1 && k > 0 && k < N - 1) {
        size_t idx = i + j * N + k * N * N;

        // Calculate error
        double diff = phi[idx] - phi_actual[idx];
        atomicAdd(error, diff * diff);

        // Calculate convergence
        diff = phi[idx] - phi_old[idx];
        atomicAdd(conv, diff * diff);
    }
}

hipError_t GPU_ERROR;

void finite_difference() {

    // METRICS
    long long average_time_per_iteration;
    long long min_time_per_iteration;
    long long max_time_per_iteration;
    long long total_time;
    long long avg_time_per_iteration;

    struct timeval start_all, end_all, start_iter, end_iter;

    gettimeofday(&start_all, NULL);

    const size_t N = 10;
    const double h = 1.0 / (N - 1);
    const double tol = 1e-6;

    size_t size = N * N * N * sizeof(double);
    double *d_phi, *d_phi_old, *d_phi_actual, *d_f_phi, *d_error, *d_conv;

    double phi_actual[N * N * N]; // phi(x, y, z)
    double f_phi[N * N * N]; // f(x, y, z)
    double phi[N * N * N]; // intermediate "new" phi(x, y, z)
    double phi_old[N * N * N]; // intermediate "old" phi(x, y, z)

    GPU_ERROR = hipMalloc(&d_phi, size);
    GPU_ERROR = hipMalloc(&d_phi_old, size);
    GPU_ERROR = hipMalloc(&d_phi_actual, size);
    GPU_ERROR = hipMalloc(&d_f_phi, size);
    GPU_ERROR = hipMalloc(&d_error, sizeof(double));
    GPU_ERROR = hipMalloc(&d_conv, sizeof(double));

    dim3 threadsPerBlock(8, 8, 8);
    dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (N + threadsPerBlock.y - 1) / threadsPerBlock.y,
                   (N + threadsPerBlock.z - 1) / threadsPerBlock.z);

    initialize<<<numBlocks, threadsPerBlock>>>(d_phi, d_phi_old, d_phi_actual, d_f_phi, N, h);

    double error = INFINITY;
    double conv = INFINITY;
    int iter = 1;
    do {

        gettimeofday(&start_iter, NULL);

        update_phi<<<numBlocks, threadsPerBlock>>>(d_phi, d_phi_old, d_f_phi, N, h);
        GPU_ERROR = hipDeviceSynchronize();

        error = 0.0;
        conv = 0.0;
        GPU_ERROR = hipMemset(d_error, 0, sizeof(double));
        GPU_ERROR = hipMemset(d_conv, 0, sizeof(double));
        compute_error_and_convergence<<<numBlocks, threadsPerBlock>>>(d_phi, d_phi_old, d_phi_actual, N, d_error, d_conv);
        GPU_ERROR = hipDeviceSynchronize();
        GPU_ERROR = hipMemcpy(d_phi_old, d_phi, size, hipMemcpyDeviceToDevice);
        GPU_ERROR = hipMemcpy(&error, d_error, sizeof(double), hipMemcpyDeviceToHost);
        GPU_ERROR = hipMemcpy(&conv, d_conv, sizeof(double), hipMemcpyDeviceToHost);

        gettimeofday(&end_iter, NULL);

        long long time_per_iteration = time_diff(start_iter, end_iter);

        if (iter == 1) {
            min_time_per_iteration = time_per_iteration;
            max_time_per_iteration = time_per_iteration;
        } else {
            min_time_per_iteration = std::min(min_time_per_iteration, time_per_iteration);
            max_time_per_iteration = std::max(max_time_per_iteration, time_per_iteration);
        }

        total_time += time_per_iteration;
        avg_time_per_iteration = total_time / iter;

        printf("Iteration %d: Max Time: %lld us, Min Time: %lld us, Avg Time: %lld us\n", 
            iter, 
            max_time_per_iteration, 
            min_time_per_iteration, 
            avg_time_per_iteration
        );
        std::cout << "Error: " << error << std::endl;
        std::cout << "Convergence: " << conv << std::endl;

        iter++;
    } while (conv > tol);

    GPU_ERROR = hipMemcpy(phi, d_phi, size, hipMemcpyDeviceToHost);

    GPU_ERROR = hipFree(d_phi);
    GPU_ERROR = hipFree(d_phi_old);
    GPU_ERROR = hipFree(d_phi_actual);
    GPU_ERROR = hipFree(d_f_phi);
    GPU_ERROR = hipFree(d_error);

    gettimeofday(&end_all, NULL);

    total_time = time_diff(start_all, end_all);
    
    printf("[FINAL RESULT]\n");
    printf("Total computation time: %lld us\n", total_time);
    printf("Average iteration time: %lld us\n", avg_time_per_iteration);
    printf("Minimum iteration time: %lld us\n", min_time_per_iteration);
    printf("Maximum iteration time: %lld us\n", max_time_per_iteration);
    printf("Iterations: %d\n", iter);
    printf("Error: %f\n", error);

}

int main() {
    finite_difference();
    return 0;
}
