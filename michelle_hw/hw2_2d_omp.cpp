#include <stdio.h> 
#include <math.h>
#include <iostream>
#include <chrono>
#include <omp.h>

double convergence_threshold = 1e-6;

// pi, for our specific function
double PI = 4 * atan(1);

double f(double x, double y) {
  return -8 * PI * PI * sin(2 * PI * x) * cos(2 * PI * y);
}

double u_func(double x, double y) {
  return sin(2 * PI * x) * cos(2 * PI * y);
}

double stencil(size_t i, size_t j, double** u_xx_yy, double d, double** u_old) {
  return (d * d * u_xx_yy[i][j] - (u_old[i-1][j] + u_old[i+1][j] + u_old[i][j-1] + u_old[i][j+1])) / -4;
}

int main(int argc, char **argv){

  if (argc != 2) {
    throw std::runtime_error("Must provide N value as argument.");
  }

  size_t N = atoi(argv[1]);

  double d = 1 / ((double) N - 1);

  // initialize contiguous arrays
  double **u_xx_yy = new double*[N];
  double *u_xx_yy_contig = new double[N * N];
  double **u = new double*[N];
  double *u_contig = new double[N * N];
  double **u_ = new double*[N];
  double *u__contig= new double[N * N];
  
  for (size_t i = 0; i < N; i++) {
    u_xx_yy[i] = &u_xx_yy_contig[i * N];
    u[i] = &u_contig[i * N];
    u_[i] = &u__contig[i * N];
  }

  // init. boundaries and RHS
  for (size_t i = 0; i < N; i++) {
    for (size_t j = 0; j < N; j++) {
      if ((i == 0) || (i == N - 1) || (j == 0) || (j == N - 1)) {
        u[i][j] = u_func(d * i, d * j); // initalize boundaries
        u_[i][j] = u_func(d * i, d * j);
      }
      u_xx_yy[i][j] = f(d * i, d * j); // initialize u_xx + u_yy
    }
  }

  bool converged = false;
  double **tmp = new double*[N];
  double s = 0;
  int c = 0;

  auto start = std::chrono::high_resolution_clock::now();
  while (!converged) {
    // apply stencil to entire domain
    #pragma omp parallel for collapse(2)
    for (size_t i = 1; i < N - 1; i++) {
      for (size_t j = 1; j < N - 1; j++) {
        u_[i][j] = stencil(i, j, u_xx_yy, d, u);
      }
    }

    // reduce max absolute error
    s = 0;
    #pragma omp parallel for collapse(2) reduction(max:s)
    for (size_t i = 1; i < N-1; i++) {
      for (size_t j = 1; j < N-1; j++) {
        s = std::max(s, abs(u[i][j] - u_[i][j]));
      }
    }

    // check convergence
    converged = s < convergence_threshold;
    
    if (c % 500 == 0) {
      printf("it %d, err mulitplier: %lf\n", c, abs(s) / convergence_threshold);
    }

    // swap u and u_
    for (size_t i = 1; i < N - 1; i++) { // skip first and last rows (all boundary)
      tmp[i] = u[i];
      u[i] = u_[i];
      u_[i] = tmp[i];
    }

    c++;
  }

  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() * 1e-9;

  double num_bytes = (6 + 2) * (N - 2) * (N - 2) * 8 * c;
  double num_flop = (1 + 7) * (N - 2) * (N - 2) * c;

  std::cout << "solving time (s): " << duration << std::endl;
  printf("Total GB: %lf\n", num_bytes * 1e-9);
  printf("Total GFLOP: %lf\n", num_flop * 1e-9);
  printf("Bandwidth (GB/s): %lf, GFLOP/s: %lf\n", num_bytes * 1e-9 / duration, num_flop * 1e-9 / duration);
  printf("FLOP/Byte: %lf\n", num_flop / num_bytes);
  printf("%d iterations\n", c);

  // compare u and the correct values
  double err = 0;
  double t;
  for (size_t i = 1; i < N - 1; i++) {
    for (size_t j = 1; j < N - 1; j++) {
      t = u_func(d * i, d * j);
      err += (u[i][j] - t) * (u[i][j] - t);
    }
  }

  std::cout << "error " << err / ((N - 2) * (N - 2)) << std::endl;
  return 0;
}

