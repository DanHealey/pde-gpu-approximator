#include <stdio.h>
#include <math.h>
#include <chrono>
#include <algorithm>
#include <iostream>
#include <mpi.h>


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

                if (i == 0 || i == N-1 || j == 0 || j == N-1 || (rank == 0 && k == 0) || (rank == size - 1 &&k == N_local-1)) {
                    mat[i + j*N + k*N*N] = exact_phi(x, y, z);
                }
            }
        }
    }
}

void update_phi_boundary(double* phi, double* phi_old, double* f_phi, size_t N, size_t N_local, double h, double* top_buf, double* bottom_buf) {
    // Update boundary points with received buffer
    if (rank < size - 1) {
        for (size_t j = 0; j < N; j++) {
            for (size_t i = 0; i < N; i++) {
                int k = N_local - 1;
                phi[i + j*N + k*N*N] = (
                    phi_old[(i-1) + j*N + k*N*N] +
                    phi_old[(i+1) + j*N + k*N*N] +
                    phi_old[i + (j-1)*N + k*N*N] +
                    phi_old[i + (j+1)*N + k*N*N] +
                    phi_old[i + j*N + (k-1)*N*N] +
                    top_buf[i + j*N] -
                    f_phi[i + j*N + k*N*N] * (h*h)) / 6.0;
            }
        }
    }
    if (rank > 0) {
        for (size_t j = 0; j < N; j++) {
            for (size_t i = 0; i < N; i++) {
                int k = 0;
                phi[i + j*N + k*N*N] = (
                    phi_old[(i-1) + j*N + k*N*N] +
                    phi_old[(i+1) + j*N + k*N*N] +
                    phi_old[i + (j-1)*N + k*N*N] +
                    phi_old[i + j*N + (k+1)*N*N] -
                    bottom_buf[i + j*N] + 
                    f_phi[i + j*N + k*N*N] * (h*h)) / 6.0;
            }
        }
    }
}

void update_phi(double* phi, double* phi_old, double* f_phi, size_t N, size_t N_local, double h) {
    // u[i,j,k] = (sum of 6 neighbors - f(x,y,z) * h^2) / 6
    for (size_t k = 1; k < N_local - 1; k++) {
        for (size_t j = 1; j < N - 1; j++) {
            for (size_t i = 1; i < N - 1; i++) {
                phi[i + j*N + k*N*N] = (
                    phi_old[(i-1) + j*N + k*N*N] +
                    phi_old[(i+1) + j*N + k*N*N] +
                    phi_old[i + (j-1)*N + k*N*N] +
                    phi_old[i + (j+1)*N + k*N*N] +
                    phi_old[i + j*N + (k-1)*N*N] +
                    phi_old[i + j*N + (k+1)*N*N] -
                    f_phi[i + j*N + k*N*N] * (h*h)) / 6.0;
            }
        }
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
    double max_time, min_time, avg_time;
    double local_conv, local_error;

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

    double top_buf[N * N]; // top boundary buffer
    double bottom_buf[N * N]; // bottom boundary buffer

    int arr_size = N * N * N_local;
    double phi_actual[arr_size]; // phi(x, y, z)
    double f_phi[arr_size]; // f(x, y, z)
    double phi[arr_size]; // intermediate "new" phi(x, y, z)
    double phi_old[arr_size]; // intermediate "old" phi(x, y, z)
   
    // Initialize matrices
    for (size_t k = 0; k < N_local; k++) {
        for (size_t j = 0; j < N; j++) {
            for (size_t i = 0; i < N; i++) {

                // Determine x, y, z of current point
                double x = x0 + i * h;
                double y = y0 + j * h;
                double z = z0 + (k + (rank * N_local)) * h;

                // Initialize matrices
                phi_actual[i + j*N + k*N*N] = exact_phi(x, y, z);
                f_phi[i + j*N + k*N*N] = f(x, y, z);
                phi[i + j*N + k*N*N] = 0.0;
                phi_old[i + j*N + k*N*N] = 0.0;
            }
        }
    }

    // Generate boundaries
    generate_boundaries(phi, N, N_local, size, h, x0, x1, y0, y1, z0, z1, rank); // on phi
    generate_boundaries(phi_old, N, N_local, size, h, x0, x1, y0, y1, z0, z1, rank); // on phi old

    double total_error = INFINITY;
    double total_conv = 0.0;
    int iter = 1;
    double start_time = MPI_Wtime();
    do {
        double iter_start = MPI_Wtime(); // Start iteration time
        MPI_Request requests[4];
        
        // Send bottom to rank+1 & receive top
        if (rank < size - 1){
            MPI_Isend(phi + (N * N * (N_local-1)), N * N, MPI_DOUBLE, rank + 1, 0, comm, &requests[0]);
            MPI_Irecv(top_buf, N * N, MPI_DOUBLE, rank + 1, 0, comm, &requests[2]);
            MPI_Wait(&requests[2], MPI_STATUS_IGNORE);
        }
        // Send top to rank-1 & receive bottom
        if (rank > 0){
            MPI_Isend(phi, N * N, MPI_DOUBLE, rank - 1, 0, comm, &requests[1]);
            MPI_Irecv(bottom_buf, N * N, MPI_DOUBLE, rank - 1, 0, comm, &requests[3]);
            MPI_Wait(&requests[3], MPI_STATUS_IGNORE);
        }

        // Update values for non-boundary points
        update_phi(phi, phi_old, f_phi, N, N_local, h);
        
        // Update boundary
        update_phi_boundary(phi, phi_old, f_phi, N, N_local, h, top_buf, bottom_buf);

        // Calculate local convergence and error
        local_conv = 0;
        local_error = 0;
        for (size_t k = 1; k < N_local-1; k++) {
            for (size_t j = 1; j < N-1; j++) {
                for (size_t i = 1; i < N-1; i++) {
                    double diff = phi[i + j*N + k*N*N] - phi_old[i + j*N + k*N*N];
                    local_conv += diff * diff;

                    diff = phi[i + j*N + k*N*N] - phi_actual[i + j*N + k*N*N];
                    local_error += diff * diff;
                }
            }
        }

        MPI_Allreduce(&local_conv, &total_conv, 1, MPI_DOUBLE, MPI_SUM, comm);
        MPI_Allreduce(&local_error, &total_error, 1, MPI_DOUBLE, MPI_SUM, comm);

        // Swap phi and phi_old
        for (size_t k = 0; k < N_local; k++) {
            for (size_t j = 0; j < N; j++) {
                for (size_t i = 0; i < N; i++) {
                    phi_old[i + j*N + k*N*N] = phi[i + j*N + k*N*N];
                }
            }
        }

        double iter_end = MPI_Wtime(); // End iteration time
        double iter_time = iter_end - iter_start;
        if (iter >= 10){
        // Aggregate iteration timing across ranks
        if (iter == 10) {
            min_time = iter_time;
            max_time = iter_time;
            avg_time = iter_time;
        }

        if (iter_time < min_time) min_time = iter_time;
        if (iter_time > max_time) max_time = iter_time;
        }
        avg_time += iter_time;
        
        //  if (rank == 0 && (iter < 10 || iter % 100 == 0)) {
        //     printf("Iteration %d:\n", iter);
        //     printf("Square difference: %f\n", total_conv);
        //     printf("Actual error: %f\n", total_error);
        //  }
         iter++;
                
    } while (total_conv > tol);

    double end_time = MPI_Wtime(); // End total time
    if (rank == 0) {
        printf("[FINAL RESULT]\n");
        printf("Total computation time: %0.2f us\n", (end_time - start_time) * 1e6);
        printf("Average iteration time: %0.2f us\n", (avg_time / (iter-10))* 1e6);
        printf("Minimum iteration time: %0.2f us\n", min_time* 1e6);
        printf("Maximum iteration time: %0.2f us\n", max_time* 1e6);
        printf("Iterations: %d\n", iter);
        printf("Error: %f\n", total_error);
    }

}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    finite_difference();
    MPI_Finalize();
    return 0;
}