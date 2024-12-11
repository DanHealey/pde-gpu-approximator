# pde-gpu-approximator

This project utilizes shared and distributed memory solutions with OpenMP, MPI, and GPU methods to solve the 3D Poisson equation.

## Authors
- Melvin He
- Daniel Healey
- Michelle Liu

## Report
[View the report](./report.pdf)

## Compiling & Executing
- Devel parition in AMD: `salloc -p devel -t 00:30:00`
- Baseline compilation: `g++ original.cpp -o original.out`
- Baseline run: `./original.out`
- Shared compilation: `g++ shared.cpp -fopenmp -O3 -o shared.out`
- Shared run: `OMP_NUM_THREADS=6 ./shared.out`
- Distributed compilation: `mpic++ distributed.cpp -lm -o distributed.out`
- Distributed run: `mpirun -np 1 ./distributed.out`
- Shared + GPU compilation: `hipcc shared_with_gpu.cu -o shared_gpu.out`
- Shared + GPU run: `./shared_gpu.out`

## Profiler Inputs
```
   # Perf counters group 1
   pmc : Wavefronts VALUInsts SALUInsts SFetchInsts
   # Perf counters group 2
   pmc : TCC_HIT[0], TCC_MISS[0]
   # Filter by dispatches range, GPU index and kernel names
   # supported range formats: "3:9", "3:", "3"
   range: 1 : 4
   gpu: 0 1 2 3
   kernel: simple Pass1 simpleConvolutionPass2
```