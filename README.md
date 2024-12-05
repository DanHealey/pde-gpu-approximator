# pde-gpu-approximator

This project utilizes shared and distributed memory solutions to solve the 3D Poisson equation.

## Compiling
- Devel parition in AMD: `salloc -p devel -t 00:30:00`
- Baseline compilation: `g++ original.cpp -o original.out`
- Baseline run: `./original.out`
- Shared compilation: `g++ shared.cpp -fopenmp -o shared.out`
- Shared run: `OMP_NUM_THREADS=1 ./shared.out`
- Distributed compilation: `mpic++ distributed.cpp -lm -o distributed.out`
- Distributed run: `mpirun -np 1 ./distributed.out`
- Shared + GPU compilation: `hipcc shared_with_gpu.cu -o shared_gpu.out`
- Shared + GPU run: `./shared_gpu.out`

## Notes
- We can use all techniques n the same code or in separate codes and compare
- Red-black is not a requirement but may be interesting to explore

Red-black method or some other optimization?
- https://www3.nd.edu/~zxu2/acms60212-40212-S12/Lec-10-02.pdf

https://edisciplinas.usp.br/pluginfile.php/41896/mod_resource/content/1/LeVeque%20Finite%20Diff.pdf
- page 41 (on pdf)

https://shenfun.readthedocs.io/en/latest/poisson3d.html