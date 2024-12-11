#include <stdio.h>
#include <math.h>
#include <chrono>
#include <algorithm>
#include <iostream>
#include <omp.h>
#include <sys/time.h>
#include <vector> // Added for storing errors in a dynamic array

/* Task
    Goal: Solve for exact solution: phi = sin(n*π*x) * cos(m*π*y) * sin(k*π*z)
        n, m, k are integers
        x, y, z are in range 0 to 1

    Given: f(x,y,z) = -(n²+m²+k²) π² cos(m π y) sin(n π x) sin(k π z)

    3D Poisson equation: (∂²/∂x² + ∂²/∂y² + ∂²/∂z²) phi(x,y,z) = f(x,y,z)
*/

double const pi = 4 * atan(1.0);
double const n1 = 2;
double const m1 = 2;
double const k1 = 2;

long long time_diff(struct timeval start, struct timeval end) {
    return (end.tv_sec - start.tv_sec) * 1000000LL + (end.tv_usec - start.tv_usec);
}

double exact_phi(double x, double y, double z){
    return sin(n1*pi*x)*cos(m1*pi*y)*sin(k1*pi*z);
}

double f(double x, double y,  double z){
    // -(k^2 + m^2 + n^2) π² * cos(m π y) * sin(n π x) * sin(k π z)
    return -(k1*k1 + m1*m1 + n1*n1) * (pi*pi) * exact_phi(x, y, z);
}

void generate_boundaries(double* mat, size_t N, double h, double x0,
    double x1, double y0, double y1, double z0, double z1) {
    #pragma omp parallel for collapse(3)
    for (size_t k = 0; k < N; k++) {
        for (size_t j = 0; j < N; j++) {
            for (size_t i = 0; i < N; i++) {
                double x = x0 + i * h;
                double y = y0 + j * h;
                double z = z0 + k * h;

                if (i == 0 || i == N-1 || j == 0
                    || j == N-1 || k == 0 || k == N-1) {
                    mat[i + j*N + k*N*N] = exact_phi(x, y, z);
                }
            }
        }
    }
}

void update_phi(double* phi, double* phi_old, double* f_phi, size_t N, double h) {
    #pragma omp parallel for collapse(3)
    for (size_t k = 1; k < N - 1; k++) {
        for (size_t j = 1; j < N - 1; j++) {
            for (size_t i = 1; i < N - 1; i++) {
                size_t idx = i + j * N + k * N * N;
                phi[idx] = (
                    phi_old[(i-1) + j*N + k*N*N] +
                    phi_old[(i+1) + j*N + k*N*N] +
                    phi_old[i + (j-1)*N + k*N*N] +
                    phi_old[i + (j+1)*N + k*N*N] +
                    phi_old[i + j*N + (k-1)*N*N] +
                    phi_old[i + j*N + (k+1)*N*N] -
                    f_phi[idx] * (h*h)) / 6.0;
            }
        }
    }
}

void finite_difference() {
    // METRICS
    long long min_iteration_time, max_iteration_time, total_time = 0, avg_iteration_time = 0;
    struct timeval start_time, end_time, start_iter_time, end_iter_time;

    gettimeofday(&start_time, NULL);

    const double x0 = 0, x1 = 1, y0 = 0, y1 = 1, z0 = 0, z1 = 1, tol = 1e-6;
    const int N = 60;
    const double h = 1.0 / (N - 1);

    double phi_actual[N * N * N]; 
    double f_phi[N * N * N]; 
    double phi[N * N * N]; 
    double phi_old[N * N * N]; 

    // Initialize matrices
    #pragma omp parallel for collapse(3)
    for (size_t k = 0; k < N; k++) {
        for (size_t j = 0; j < N; j++) {
            for (size_t i = 0; i < N; i++) {
                double x = x0 + i * h;
                double y = y0 + j * h;
                double z = z0 + k * h;
                phi_actual[i + j * N + k * N * N] = exact_phi(x, y, z);
                f_phi[i + j * N + k * N * N] = f(x, y, z);
                phi[i + j * N + k * N * N] = 0.0;
                phi_old[i + j * N + k * N * N] = 0.0;
            }
        }
    }

    // Generate boundaries
    generate_boundaries(phi, N, h, x0, x1, y0, y1, z0, z1);
    generate_boundaries(phi_old, N, h, x0, x1, y0, y1, z0, z1);

    double error = INFINITY, square_diff = 0.0;
    int iteration = 1;

    std::vector<double> errors; // store error each iteration

    do {
        gettimeofday(&start_iter_time, NULL);

        // Update phi
        update_phi(phi, phi_old, f_phi, N, h);

        // Swap phi and phi_old
        std::swap(phi, phi_old);

        // Calculate convergence difference
        square_diff = 0.0;
        #pragma omp parallel for reduction(+:square_diff) collapse(3)
        for (size_t k = 0; k < N; k++) {
            for (size_t j = 0; j < N; j++) {
                for (size_t i = 0; i < N; i++) {
                    double diff = phi[i + j * N + k * N * N] - phi_old[i + j * N + k * N * N];
                    square_diff += diff * diff;
                }
            }
        }

        // Calculate actual error
        error = 0.0;
        #pragma omp parallel for reduction(+:error) collapse(3)
        for (size_t k = 0; k < N; k++) {
            for (size_t j = 0; j < N; j++) {
                for (size_t i = 0; i < N; i++) {
                    double diff = phi[i + j * N + k * N * N] - phi_actual[i + j * N + k * N * N];
                    error += diff * diff;
                }
            }
        }

        // Store the error in the vector
        errors.push_back(error);

        // Start tracking from 10th iteration
        gettimeofday(&end_iter_time, NULL);
        if (iteration >= 10) {
            long long iteration_time = time_diff(start_iter_time, end_iter_time);
            if (iteration == 10) {
                min_iteration_time = iteration_time;
                max_iteration_time = iteration_time;
            } else {
                min_iteration_time = std::min(min_iteration_time, iteration_time);
                max_iteration_time = std::max(max_iteration_time, iteration_time);
            }
            total_time += iteration_time;
            avg_iteration_time = total_time / (iteration - 9);
        }

        iteration++;
    } while (square_diff > tol);

    gettimeofday(&end_time, NULL);

    // Print final results
    printf("[FINAL RESULT]\n");
    printf("Iterations: %d\n", iteration);
    printf("Final Error: %f\n", error);
    if (iteration > 10) {
        printf("Average iteration time: %lld us\n", avg_iteration_time);
        printf("Minimum iteration time: %lld us\n", min_iteration_time);
        printf("Maximum iteration time: %lld us\n", max_iteration_time);
    }

    // Print all errors so you can copy them into Excel or other tools
    printf("\n[ERROR BY ITERATION]\n");
    for (size_t i = 0; i < errors.size(); i++) {
        // iteration numbering starting from 1
        printf("%zu,%e\n", i+1, errors[i]);
    }

}

int main(int argc, char **argv) {
    finite_difference();   
    return 0;
}
