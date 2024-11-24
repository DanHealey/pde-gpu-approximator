#include <stdio.h>
#include <math.h>
#include <chrono>
#include <algorithm>
#include <iostream>


/* Task
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

double exact_phi(double x, double y, double z){
    return sin(n1*pi*x)*cos(m1*pi*y)*sin(k1*pi*z);
}

double f(double x, double y,  double z){
    // -(k^2 + m^2 + n^2) π^2 cos(m π y) sin(n π x) sin(k π z)
    return -(k1*k1 + m1*m1 + n1*n1) * (pi*pi) * exact_phi(x, y, z);
}

void generate_boundaries(double* mat, size_t N, double h, double x0, double x1, double y0, double y1, double z0, double z1) {
    for (size_t k = 0; k < N; k++) {
        for (size_t j = 0; j < N; j++) {
            for (size_t i = 0; i < N; i++) {
                double x = x0 + i * h;
                double y = y0 + j * h;
                double z = z0 + k * h;

                if (i == 0 || i == N-1 || j == 0 || j == N-1 || k == 0 || k == N-1) {
                    mat[i + j*N + k*N*N] = exact_phi(x, y, z);
                }
            }
        }
    }
}

void update_phi(double* phi, double* phi_old, double* f_phi, size_t N, double h) {
    // u[i,j,k] = (sum of 6 neighbors - f(x,y,z) * h^2) / 6
    for (size_t k = 1; k < N - 1; k++) {
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

    double phi_actual[N * N * N]; // phi(x, y, z)
    double f_phi[N * N * N]; // f(x, y, z)
    double phi[N * N * N]; // intermediate "new" phi(x, y, z)
    double phi_old[N * N * N]; // intermediate "old" phi(x, y, z)

    // Initialize matrices
    for (size_t k = 0; k < N; k++) {
        for (size_t j = 0; j < N; j++) {
            for (size_t i = 0; i < N; i++) {
                double x = x0 + i * h;
                double y = y0 + j * h;
                double z = z0 + k * h;
                phi_actual[i + j*N + k*N*N] = exact_phi(x, y, z);
                f_phi[i + j * N + k * N * N] = f(x, y, z);
                phi[i + j*N + k*N*N] = 0.0;
                phi_old[i + j*N + k*N*N] = 0.0;
            }
        }
    }

    // Generate boundaries
    generate_boundaries(phi, N, h, x0, x1, y0, y1, z0, z1); // on phi
    generate_boundaries(phi_old, N, h, x0, x1, y0, y1, z0, z1); // on phi old

    
    double error = INFINITY;
    double square_diff = 0.0;
    do {
        // Update phi
        update_phi(phi, phi_old, f_phi, N, h);

        // Swap phi and phi_old
        std::swap(phi, phi_old);

        // Calculate convergence difference
        square_diff = 0.0;
        for (size_t i = 0; i < N; i++) {
            for (size_t j = 0; j < N; j++) {
                for (size_t k = 0; k < N; k++) {
                    double diff = phi[i + j*N + k*N*N] - phi_old[i + j*N + k*N*N];
                    square_diff += diff * diff;
                }
            }
        }
        printf("Square difference: %f\n", square_diff);

        // Calculate actual error
        error = 0.0;
        for (size_t k = 0; k < N; k++) {
            for (size_t j = 0; j < N; j++) {
                for (size_t i = 0; i < N; i++) {
                    double diff = phi[i + j*N + k*N*N] - phi_actual[i + j*N + k*N*N];
                    error += diff * diff;
                }
            }
        }
        printf("Actual error: %f\n", error);
                
    } while (square_diff > tol);

}


int main(int argc, char **argv) {
    finite_difference();   
    return 0;
}