#include <stdio.h>
#include <math.h>
#include <chrono>
#include <algorithm>
#include <iostream>
#include <mpi.h>

#include <hip/hip_runtime.h>


/* Task
    Using distributed message passing (passing with each layer k, along z axis)
    Goal: Solve for exact solution: phi = sin(n*π*x) * cos(m*π*y) * sin(k*π*z)
        n, m, k are integers
        x, y, z are in range 0 to 1

    Given: f(x,y,z) = -(n²+m²+k²) π² cos(mπy) sin(nπx) sin(kπz)
        ∂²f/∂x² = -n²π² * sin(nπx) * cos(mπy) * sin(kπz)
        ∂²f/∂y² = -m²π² * cos(mπy) * sin(nπx) * sin(kπz)
        ∂²f/∂z² = -k²π² * sin(kπz) * sin(nπx) * cos(mπy)

    3D Poisson equation: (∂²/∂x² + ∂²/∂y² + ∂²/∂z²) phi(x,y,z) = f(x,y,z)
        ∂²u/∂x² = (u[i-1,j,k] - 2*u[i,j,k] + u[i+1,j,k] ) / (Dx*Dx)
        ∂²u/∂y² = (u[i,j-1,k] - 2*u[i,j,k] + u[i,j+1,k] ) / (Dy*Dy)
        ∂²u/∂z² = (u[i,j,k-1] - 2*u[i,j,k] + u[i,j,k+1] ) / (Dz*Dz)

    Algorithm: (dx=dy=dz=h=1/(N-1))
        f(x,y,z) = (u[i-1,j,k]-2*u[i,j,k]+u[i+1,j,k] + u[i,j-1,k]-2*u[i,j,k]+u[i,j+1,k] + u[i,j,k-1]-2*u[i,j,k]+u[i,j,k+1]) / h^2
        u[i,j,k] = (sum of 6 neighbors - f(x,y,z) * h^2) / 6
*/

double const pi = 4 * atan(1.0);
double const n1 = 2;
double const m1 = 2;
double const k1 = 2;

int rank, size;

hipError_t GPU_ERROR;

double exact_phi(double x, double y, double z){
    return sin(n1*pi*x)*cos(m1*pi*y)*sin(k1*pi*z);
}

double f(double x, double y,  double z){
    // -(k^2 + m^2 + n^2) π^2 cos(m π y) sin(n π x) sin(k π z)
    return -(k1*k1 + m1*m1 + n1*n1) * (pi*pi) * exact_phi(x, y, z);
}

void generate_boundaries(double* mat, size_t N, size_t N_local, int size, double h, double x0, double x1, double y0, double y1, double z0, double z1, int rank) {
    for (size_t i = 0; i < N; i++) {
        for (size_t j = 0; j < N; j++) {
            for (size_t k = 0; k < N_local; k++) {
                double x = x0 + i * h;
                double y = y0 + j * h;
                double z = z0 + k * h + rank * N_local * h;

                if (i == 0 || i == N-1 || j == 0 || j == N-1 || (rank == 0 && k == 0) || (rank == size - 1 &&k == N-1)) {
                    mat[i + j*N + k*N*N] = exact_phi(x, y, z);
                }
            }
        }
    }
}

__global__
void update_phi_kernel(double* phi, const double* phi_old, const double* f_phi, size_t N, size_t N_local, double h) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    int k = threadIdx.z + blockIdx.z * blockDim.z;

    if (i >= 1 && i < N - 1 && j >= 1 && j < N - 1 && k >= 1 && k < N_local - 1) {
        size_t idx = i + j * N + k * N * N;
        phi[idx] = (
            phi_old[(i-1) + j * N + k * N * N] +
            phi_old[(i+1) + j * N + k * N * N] +
            phi_old[i + (j-1) * N + k * N * N] +
            phi_old[i + (j+1) * N + k * N * N] +
            phi_old[i + j * N + (k-1) * N * N] +
            phi_old[i + j * N + (k+1) * N * N] -
            f_phi[idx] * (h * h)) / 6.0;
    }
}

// HIP Kernel for boundary updates
__global__
void update_phi_boundary_kernel(double* phi, double* phi_old, double* f_phi, size_t N, size_t N_local, double h, double* top_buf, double* bottom_buf, int rank, int size) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    if (i >= 1 && i < N - 1 && j >= 1 && j < N - 1) {
        if (rank < size - 1) { // Top boundary
            int k = N_local - 1;
            size_t idx = i + j * N + k * N * N;
            phi[idx] = (
                phi_old[(i-1) + j * N + k * N * N] +
                phi_old[(i+1) + j * N + k * N * N] +
                phi_old[i + (j-1) * N + k * N * N] +
                phi_old[i + (j+1) * N + k * N * N] +
                phi_old[i + j * N + (k-1) * N * N] +
                top_buf[i + j * N] -
                f_phi[idx] * (h * h)) / 6.0;
        }
        if (rank > 0) { // Bottom boundary
            int k = 0;
            size_t idx = i + j * N + k * N * N;
            phi[idx] = (
                phi_old[(i-1) + j * N + k * N * N] +
                phi_old[(i+1) + j * N + k * N * N] +
                phi_old[i + (j-1) * N + k * N * N] +
                phi_old[i + (j+1) * N + k * N * N] +
                phi_old[i + j * N + (k+1) * N * N] +
                bottom_buf[i + j * N] -
                f_phi[idx] * (h * h)) / 6.0;
        }
    }
}

// HIP Kernel for calculating convergence and error and swapping phi and phi_old
__global__
void calculate_error_kernel(double* phi, double* phi_old, double* exact_phi, double* convergence, double* error, size_t N, size_t N_local) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    int k = threadIdx.z + blockIdx.z * blockDim.z;

    if (i >= 0 && i < N && j >= 0 && j < N && k >= 0 && k < N_local) {
        size_t idx = i + j * N + k * N * N;
        double diff_phi = phi[idx] - phi_old[idx];
        double diff_exact = phi[idx] - exact_phi[idx];
        atomicAdd(convergence, diff_phi * diff_phi);
        atomicAdd(error, diff_exact * diff_exact);

        phi_old[idx] = phi[idx];
    }
}

void finite_difference() {
    const double x0 = 0;
    const double x1 = 1;
    const double y0 = 0;
    const double y1 = 1;
    const double z0 = 0;
    const double z1 = 1;
    const double tol = 1e-6;
    const int N = 10;
    const double h = 1.0 / (N - 1);

    // Initialize MPI
    int rank, size;
    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    // Local domain size
    int N_local = N / size;
    if (rank < N % size) {
        N_local++;
    }

    // Define grid and block dimensions
    int block_size = 16; // 16x16 threads
    dim3 blockDim(block_size, block_size);
    dim3 gridDim((N - 2 + blockDim.x - 1) / blockDim.x, (N - 2 + blockDim.y - 1) / blockDim.y, (N_local - 2 + blockDim.z - 1) / blockDim.z);
    int num_blocks = gridDim.x * gridDim.y * gridDim.z;

    double top_buf[N * N]; // top boundary buffer
    double bottom_buf[N * N]; // bottom boundary buffer

    int arr_size = N * N * N_local;
    double phi_actual[arr_size]; // phi(x, y, z)
    double f_phi[arr_size]; // f(x, y, z)
    double phi[arr_size]; // intermediate "new" phi(x, y, z)
    double phi_old[arr_size]; // intermediate "old" phi(x, y, z)

    double local_conv = 0.0; // Local convergence
    double local_error = 0.0; // Local error

    // Allocate all of these on GPU
    double* d_top_buf;
    double* d_bottom_buf;
    double* d_phi_actual;
    double* d_f_phi;
    double* d_phi;
    double* d_phi_old;
    double* d_convergence;
    double* d_error;

    GPU_ERROR = hipMalloc(&d_top_buf, N * N * sizeof(double));
    GPU_ERROR = hipMalloc(&d_bottom_buf, N * N * sizeof(double));
    GPU_ERROR = hipMalloc(&d_phi_actual, arr_size * sizeof(double));
    GPU_ERROR = hipMalloc(&d_f_phi, arr_size * sizeof(double));
    GPU_ERROR = hipMalloc(&d_phi, arr_size * sizeof(double));
    GPU_ERROR = hipMalloc(&d_phi_old, arr_size * sizeof(double));
    GPU_ERROR = hipMalloc(&d_convergence, sizeof(double));
    GPU_ERROR = hipMalloc(&d_error, sizeof(double));
   
    // Initialize matrices
    for (size_t k = 0; k < N_local; k++) {
        for (size_t j = 0; j < N; j++) {
            for (size_t i = 0; i < N; i++) {

                // Determine x, y, z of current point
                double x = x0 + i * h;
                double y = y0 + j * h;
                double z = z0 + k * h + rank * N_local * h;

                // Initialize matrices
                phi_actual[i + j*N + k*N*N] = exact_phi(x, y, z);
                f_phi[i + j * N + k * N * N] = f(x, y, z);
                phi[i + j*N + k*N*N] = 0.0;
                phi_old[i + j*N + k*N*N] = 0.0;
            }
        }
    }

    // Copy f_phi, phi_actual, phi, phi_old to GPU
    GPU_ERROR = hipMemcpy(d_f_phi, f_phi, arr_size * sizeof(double), hipMemcpyHostToDevice);
    GPU_ERROR = hipMemcpy(d_phi_actual, phi_actual, arr_size * sizeof(double), hipMemcpyHostToDevice);
    GPU_ERROR = hipMemcpy(d_phi, phi, arr_size * sizeof(double), hipMemcpyHostToDevice);
    GPU_ERROR = hipMemcpy(d_phi_old, phi_old, arr_size * sizeof(double), hipMemcpyHostToDevice);

    // Generate boundaries
    generate_boundaries(phi, N, N_local, size, h, x0, x1, y0, y1, z0, z1, rank); // on phi
    generate_boundaries(phi_old, N, N_local, size, h, x0, x1, y0, y1, z0, z1, rank); // on phi old

    double total_error = INFINITY;
    double total_conv = 0.0;
    int iter = 1;

    // Print out MPI information
    if (rank == 0) {
        printf("Running with %d processes\n", size);
    }

    do {
        MPI_Request requests[4];
        
        // Send bottom to rank+1 & receive top
        if (rank < size - 1){
            MPI_Isend(phi + N * N * (N_local - 1), N * N, MPI_DOUBLE, rank + 1, 0, comm, &requests[0]);
            MPI_Irecv(top_buf, N * N, MPI_DOUBLE, rank + 1, 0, comm, &requests[2]);
            MPI_Wait(&requests[2], MPI_STATUS_IGNORE);
        }
        // Send top to rank-1 & receive bottom
        if (rank > 0){
            MPI_Isend(phi, N * N, MPI_DOUBLE, rank - 1, 0, comm, &requests[1]);
            MPI_Irecv(bottom_buf, N * N, MPI_DOUBLE, rank - 1, 0, comm, &requests[3]);
            MPI_Wait(&requests[3], MPI_STATUS_IGNORE);
        }

        // Copy top buffer, bottom buffer to GPU
        GPU_ERROR = hipMemcpy(d_top_buf, top_buf, N * N * sizeof(double), hipMemcpyHostToDevice);
        GPU_ERROR = hipMemcpy(d_bottom_buf, bottom_buf, N * N * sizeof(double), hipMemcpyHostToDevice);

        // Call update kernel (both normal and boundary)
        update_phi_kernel<<<gridDim, blockDim>>>(d_phi, d_phi_old, d_f_phi, N, N_local, h);
        update_phi_boundary_kernel<<<gridDim, blockDim>>>(d_phi, d_phi_old, d_f_phi, N, N_local, h, d_top_buf, d_bottom_buf, rank, size);
        GPU_ERROR = hipDeviceSynchronize();

        // Set d_error and d_convergence to 0
        GPU_ERROR = hipMemset(d_convergence, 0, sizeof(double));
        GPU_ERROR = hipMemset(d_error, 0, sizeof(double));

        // Call kernel to calculate error, convergence and swap
        calculate_error_kernel<<<gridDim, blockDim>>>(d_phi, d_phi_old, d_phi_actual, d_convergence, d_error, N, N_local);
        GPU_ERROR = hipMemcpy(&local_conv, d_convergence, sizeof(double), hipMemcpyDeviceToHost);
        GPU_ERROR = hipMemcpy(&local_error, d_error, sizeof(double), hipMemcpyDeviceToHost);
        GPU_ERROR = hipDeviceSynchronize();

        if ((iter < 10 || iter % 10 == 0)) {
            printf("Local Square difference: %f\n", local_conv);
            printf("Local Actual error: %f\n", local_error);
        }

        MPI_Allreduce(&local_conv, &total_conv, 1, MPI_DOUBLE, MPI_SUM, comm);
        MPI_Allreduce(&local_error, &total_error, 1, MPI_DOUBLE, MPI_SUM, comm);
         if (rank == 0 && (iter < 10 || iter % 10 == 0)) {
            printf("Square difference: %f\n", total_conv);
            printf("Actual error: %f\n", total_error);
         }

         iter++;
                
    } while (total_conv > tol);

}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    finite_difference();
    MPI_Finalize();
    return 0;
}