#include <stdio.h>
#include <math.h>
#include <chrono>
#include <algorithm>
#include <iostream>
#include <mpi.h>

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
#else
#include <cuda_runtime.h>
#endif

#define NUM_TRIALS 5

/*
Task: Write an iterative solver to find the value of u(x,y) using the
shared memory model and offloading to GPU, and using MPI to exchange ghost rows/columns.

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
 * @brief CUDA kernel to update U and compute local maximum difference.
 */
__global__
void update_U(double* U_d, double* U_old_d, double* f_U_d, int N_global, int arr_size_x, int arr_size_y, double h, double* max_diff_per_block_d) {
    // Global indices (start from 1)
    int i = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int j = blockIdx.x * blockDim.x + threadIdx.x + 1;

    // Index in shared memory
    int tid = threadIdx.y * blockDim.x + threadIdx.x;

    // Shared memory for differences
    extern __shared__ double s_diff[];

    double diff = 0.0;

    if (i < arr_size_y - 1 && j < arr_size_x - 1) {
        double U_new = (U_old_d[(i + 1) * arr_size_x + j] + U_old_d[(i - 1) * arr_size_x + j] + U_old_d[i * arr_size_x + (j + 1)] + U_old_d[i * arr_size_x + (j - 1)] - f_U_d[i * arr_size_x + j] * h * h) / 4.0;
        diff = fabs(U_new - U_old_d[i * arr_size_x + j]);
        U_d[i * arr_size_x + j] = U_new;
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
 * @brief Solves the equation using second order finite difference on the GPU, using MPI for domain decomposition.
 */
double second_finite_difference(size_t N_global, double error_threshold = 1e-6) {
    const double x0 = 0;
    const double x1 = 1;
    const double y0 = 0;
    const double y1 = 1;
    const double h = (x1 - x0) / (N_global - 1);

    // Initialize MPI
    int rank, size;
    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    int device_count;
    cudaGetDeviceCount(&device_count);
    cudaSetDevice(rank % device_count);

    // Decide on domain decomposition
    int dims[2] = {0, 0};
    MPI_Dims_create(size, 2, dims); // let MPI decide dimensions
    int periods[2] = {0, 0};
    MPI_Comm cart_comm;
    MPI_Cart_create(comm, 2, dims, periods, 0, &cart_comm);

    int coords[2];
    MPI_Cart_coords(cart_comm, rank, 2, coords);
    int x_rank = coords[0];
    int y_rank = coords[1];

    int up_rank, down_rank, left_rank, right_rank;
    MPI_Cart_shift(cart_comm, 0, 1, &left_rank, &right_rank);
    MPI_Cart_shift(cart_comm, 1, 1, &up_rank, &down_rank);

    // Calculate local domain size
    size_t N = N_global - 2; // exclude boundary points
    size_t chunk_size_x = N / dims[0];
    size_t chunk_size_y = N / dims[1];

    size_t remainder_x = N % dims[0];
    size_t remainder_y = N % dims[1];

    size_t local_Nx = chunk_size_x + (x_rank < remainder_x ? 1 : 0);
    size_t local_Ny = chunk_size_y + (y_rank < remainder_y ? 1 : 0);

    // Starting indices
    size_t start_x = x_rank * chunk_size_x + std::min((int)x_rank, (int)remainder_x) + 1; // +1 for boundary
    size_t start_y = y_rank * chunk_size_y + std::min((int)y_rank, (int)remainder_y) + 1;

    // Local array sizes including ghost layers
    size_t arr_size_x = local_Nx + 2;
    size_t arr_size_y = local_Ny + 2;

    // Allocate host memory (contiguous)
    size_t arr_size = arr_size_x * arr_size_y;
    double* U_actual = new double[arr_size];
    double* f_U = new double[arr_size];
    double* U = new double[arr_size];
    double* U_old = new double[arr_size];

    // Allocate device memory
    double *U_d, *U_old_d, *f_U_d;
    cudaMalloc(&U_d, arr_size * sizeof(double));
    cudaMalloc(&U_old_d, arr_size * sizeof(double));
    cudaMalloc(&f_U_d, arr_size * sizeof(double));

    // Define grid and block dimensions
    int block_size = 16; // 16x16 threads
    dim3 blockDim(block_size, block_size);
    dim3 gridDim((arr_size_x - 2 + blockDim.x - 1) / blockDim.x, (arr_size_y - 2 + blockDim.y - 1) / blockDim.y);
    int num_blocks = gridDim.x * gridDim.y;

    // Allocate memory for maximum differences per block
    double *max_diff_per_block_d;
    cudaMalloc(&max_diff_per_block_d, num_blocks * sizeof(double));

    // Buffers for exchanging ghost cells
    double* send_left = new double[(arr_size_y - 2)];
    double* recv_left = new double[(arr_size_y - 2)];
    double* send_right = new double[(arr_size_y - 2)];
    double* recv_right = new double[(arr_size_y - 2)];
    double* send_top = new double[(arr_size_x - 2)];
    double* recv_top = new double[(arr_size_x - 2)];
    double* send_bottom = new double[(arr_size_x - 2)];
    double* recv_bottom = new double[(arr_size_x - 2)];

    int c = 0;

    double *max_diff_per_block_h = new double[num_blocks];

    double total_time = 0;
    double copy_time = 0;

    for (int t = 0; t < NUM_TRIALS; t++) {
        // Initialize U_actual and f_U
        for (size_t i = 0; i < arr_size_y; i++) {
            for (size_t j = 0; j < arr_size_x; j++) {
                size_t global_i = start_y + i - 1;
                size_t global_j = start_x + j - 1;
                double x = x0 + global_j * h;
                double y = y0 + global_i * h;
                U_actual[i * arr_size_x + j] = actual(x, y);
                f_U[i * arr_size_x + j] = f(x, y);
                U_old[i * arr_size_x + j] = 0.0;
                U[i * arr_size_x + j] = 0.0;
            }
        }

        // Apply boundary conditions
        // Left boundary
        if (left_rank == MPI_PROC_NULL) {
            for (size_t i = 0; i < arr_size_y; i++) {
                size_t global_i = start_y + i - 1;
                double y = y0 + global_i * h;
                U_old[i * arr_size_x + 0] = actual(x0, y);
                U[i * arr_size_x + 0] = U_old[i * arr_size_x + 0];
            }
        }
        // Right boundary
        if (right_rank == MPI_PROC_NULL) {
            for (size_t i = 0; i < arr_size_y; i++) {
                size_t global_i = start_y + i - 1;
                double y = y0 + global_i * h;
                U_old[i * arr_size_x + arr_size_x - 1] = actual(x1, y);
                U[i * arr_size_x + arr_size_x - 1] = U_old[i * arr_size_x + arr_size_x - 1];
            }
        }
        // Top boundary
        if (up_rank == MPI_PROC_NULL) {
            for (size_t j = 0; j < arr_size_x; j++) {
                size_t global_j = start_x + j - 1;
                double x = x0 + global_j * h;
                U_old[0 * arr_size_x + j] = actual(x, y0);
                U[0 * arr_size_x + j] = U_old[0 * arr_size_x + j];
            }
        }
        // Bottom boundary
        if (down_rank == MPI_PROC_NULL) {
            for (size_t j = 0; j < arr_size_x; j++) {
                size_t global_j = start_x + j - 1;
                double x = x0 + global_j * h;
                U_old[(arr_size_y - 1) * arr_size_x + j] = actual(x, y1);
                U[(arr_size_y - 1) * arr_size_x + j] = U_old[(arr_size_y - 1) * arr_size_x + j];
            }
        }

        // Copy data to device
        cudaMemcpy(f_U_d, f_U, arr_size * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(U_old_d, U_old, arr_size * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(U_d, U_old_d, arr_size * sizeof(double), cudaMemcpyDeviceToDevice);

        double error = 1.0;        

        size_t shared_mem_size = blockDim.x * blockDim.y * sizeof(double);

        MPI_Request reqs[8];

        // Start timer
        auto start = std::chrono::high_resolution_clock::now();

        while (error > error_threshold) {
            // Exchange ghost cells
            auto copy_start = std::chrono::high_resolution_clock::now();
            cudaMemcpy(U_old, U_old_d, arr_size * sizeof(double), cudaMemcpyDeviceToHost);
            auto copy_end = std::chrono::high_resolution_clock::now();

            copy_time += std::chrono::duration_cast<std::chrono::nanoseconds>(copy_end - copy_start).count();

            // Pack send buffers
            // Left
            if (left_rank != MPI_PROC_NULL) {
                for (size_t i = 1; i < arr_size_y - 1; i++) {
                    send_left[i - 1] = U_old[i * arr_size_x + 1]; // column 1
                }
            }
            // Right
            if (right_rank != MPI_PROC_NULL) {
                for (size_t i = 1; i < arr_size_y - 1; i++) {
                    send_right[i - 1] = U_old[i * arr_size_x + arr_size_x - 2]; // column arr_size_x -2
                }
            }
            // Top
            if (up_rank != MPI_PROC_NULL) {
                for (size_t j = 1; j < arr_size_x - 1; j++) {
                    send_top[j - 1] = U_old[1 * arr_size_x + j]; // row 1
                }
            }
            // Bottom
            if (down_rank != MPI_PROC_NULL) {
                for (size_t j = 1; j < arr_size_x - 1; j++) {
                    send_bottom[j - 1] = U_old[(arr_size_y - 2) * arr_size_x + j]; // row arr_size_y -2
                }
            }

            // Start non-blocking sends and receives
            int req_count = 0;
            MPI_Status statuses[8];

            // Left-right communication
            if (left_rank != MPI_PROC_NULL) {
                MPI_Isend(send_left, arr_size_y - 2, MPI_DOUBLE, left_rank, 0, cart_comm, &reqs[req_count++]);
                MPI_Irecv(recv_left, arr_size_y - 2, MPI_DOUBLE, left_rank, 1, cart_comm, &reqs[req_count++]);
            }
            if (right_rank != MPI_PROC_NULL) {
                MPI_Isend(send_right, arr_size_y - 2, MPI_DOUBLE, right_rank, 1, cart_comm, &reqs[req_count++]);
                MPI_Irecv(recv_right, arr_size_y - 2, MPI_DOUBLE, right_rank, 0, cart_comm, &reqs[req_count++]);
            }

            // Up-down communication
            if (up_rank != MPI_PROC_NULL) {
                MPI_Isend(send_top, arr_size_x - 2, MPI_DOUBLE, up_rank, 2, cart_comm, &reqs[req_count++]);
                MPI_Irecv(recv_top, arr_size_x - 2, MPI_DOUBLE, up_rank, 3, cart_comm, &reqs[req_count++]);
            }
            if (down_rank != MPI_PROC_NULL) {
                MPI_Isend(send_bottom, arr_size_x - 2, MPI_DOUBLE, down_rank, 3, cart_comm, &reqs[req_count++]);
                MPI_Irecv(recv_bottom, arr_size_x - 2, MPI_DOUBLE, down_rank, 2, cart_comm, &reqs[req_count++]);
            }

            // Wait for all communication to finish
            MPI_Waitall(req_count, reqs, statuses);

            // Unpack received data into U_old
            // Left
            if (left_rank != MPI_PROC_NULL) {
                for (size_t i = 1; i < arr_size_y - 1; i++) {
                    U_old[(i)*arr_size_x + 0] = recv_left[i - 1];
                }
            }
            // Right
            if (right_rank != MPI_PROC_NULL) {
                for (size_t i = 1; i < arr_size_y - 1; i++) {
                    U_old[(i)*arr_size_x + arr_size_x - 1] = recv_right[i - 1];
                }
            }
            // Top
            if (up_rank != MPI_PROC_NULL) {
                for (size_t j = 1; j < arr_size_x - 1; j++) {
                    U_old[0 * arr_size_x + j] = recv_top[j - 1];
                }
            }
            // Bottom
            if (down_rank != MPI_PROC_NULL) {
                for (size_t j = 1; j < arr_size_x - 1; j++) {
                    U_old[(arr_size_y - 1) * arr_size_x + j] = recv_bottom[j - 1];
                }
            }

            // Copy updated U_old back to device
            copy_start = std::chrono::high_resolution_clock::now(); 
            cudaMemcpy(U_old_d, U_old, arr_size * sizeof(double), cudaMemcpyHostToDevice);
            copy_end = std::chrono::high_resolution_clock::now(); 

            copy_time += std::chrono::duration_cast<std::chrono::nanoseconds>(copy_end - copy_start).count();

            // Set max_diff_per_block_d to zero
            // cudaMemset(max_diff_per_block_d, 0, num_blocks * sizeof(double));

            // Launch the kernel
            update_U<<<gridDim, blockDim, shared_mem_size>>>(U_d, U_old_d, f_U_d, N_global, arr_size_x, arr_size_y, h, max_diff_per_block_d);

            #ifdef TIMED
            cudaDeviceSynchronize();
            #endif

            // Copy max_diff_per_block_d to host
            copy_start = std::chrono::high_resolution_clock::now(); 
            cudaMemcpy(max_diff_per_block_h, max_diff_per_block_d, num_blocks * sizeof(double), cudaMemcpyDeviceToHost);
            copy_end = std::chrono::high_resolution_clock::now(); 

            copy_time += std::chrono::duration_cast<std::chrono::nanoseconds>(copy_end - copy_start).count();

            // Reduce to find maximum error locally
            double local_error = 0.0;
            for (int i = 0; i < num_blocks; i++) {
                local_error = std::max(local_error, max_diff_per_block_h[i]);
            }

            // Reduce the error across all processes
            MPI_Allreduce(&local_error, &error, 1, MPI_DOUBLE, MPI_MAX, comm);

            // Swap U_d and U_old_d pointers
            std::swap(U_d, U_old_d);

            c++;
            // if (c % 10000 == 0) {
            //     std::cout << c << ", Error: " << error << std::endl;
            // }
        }
        
        // Stop timer
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
        total_time += duration.count() * 1e-9;
    }

    // Copy U_old_d (latest U values) to U
    cudaMemcpy(U, U_old_d, arr_size * sizeof(double), cudaMemcpyDeviceToHost);

    if (rank == 0) {

        // convert to seconds
        copy_time = copy_time * 1e-9;
        double time = total_time / NUM_TRIALS;
        double no_copy = (total_time - copy_time) / NUM_TRIALS;

        double num_bytes = ((6 + 2) * (N - 2) * (N - 2)) * 8 * (c / NUM_TRIALS);
        double num_flop = (1 + 7) * (N - 2) * (N - 2) * (c / NUM_TRIALS);

        printf("Mean time taken: %lf (ms)\n", total_time * 1e3 / NUM_TRIALS);
        printf("Mean iterations: %d\n", (int) ((double) c / NUM_TRIALS));
        printf("Mean time per iteration: %lf (µs)\n", total_time * 1e6 / c );
        printf("%lf GFLOP/s, %lf GB/s\n", num_flop * 1e-9 / time, num_bytes * 1e-9 / time);
        #ifdef TIMED
        printf("Mean time copying between host and device per iteration: %lf (µs)\n", copy_time * 1e6 / c);
        printf("Mean time per iteration without copying: %lf (µs)\n", (total_time - copy_time) * 1e6 / c);
        printf("(without copying) %lf GFLOP/s, %lf GB/s\n", num_flop * 1e-9 / no_copy, num_bytes * 1e-9 / no_copy);
        #endif
    }

    // Compute true error (MSE) between U and U_actual
    double local_err = 0.0;
    for (size_t i = 1; i < arr_size_y -1; i++) {
        for (size_t j = 1; j < arr_size_x -1; j++) {
            double diff = U_actual[i * arr_size_x + j] - U[i * arr_size_x + j];
            local_err += diff * diff;
        }
    }
    double global_err = 0.0;
    MPI_Reduce(&local_err, &global_err, 1, MPI_DOUBLE, MPI_SUM, 0, comm);
    double mse = global_err / ((N_global - 2) * (N_global - 2));

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
    delete[] send_left;
    delete[] recv_left;
    delete[] send_right;
    delete[] recv_right;
    delete[] send_top;
    delete[] recv_top;
    delete[] send_bottom;
    delete[] recv_bottom;

    return mse;
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    size_t Ns[] = {10, 100, 200, 500, 700, 1000};
    double error;

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    for (auto N : Ns) {
        if (rank == 0) {
            std::cout << "N: " << N << std::endl;
        }
        error = second_finite_difference(N, 1e-6);
        if (rank == 0) {
            std::cout << "Mean squared error: " << error << std::endl;
            std::cout << std::endl;
        }
    }

    MPI_Finalize();
    return 0;
}
