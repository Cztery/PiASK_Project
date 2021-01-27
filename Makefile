SHELL:=/bin/bash
.PHONY : cubuild ompbuild mpibuild

all: curun omprun mpirun




cu_integral: integral.cu
	command -v nvcc || module add plgrid/apps/cuda; \
	nvcc integral.cu -o cu_integral
cubuild: cu_integral


omp_integral: integral_omp.c
	command -v icc || module add plgrid/tools/intel; \
	icc -openmp integral_omp.c my_timers.c -o omp_integral
ompbuild: omp_integral


mpi_integral: integral_mpi.c
	command -v mpicc || module add plgrid/tools/openmpi; \
        srun -N1 --ntasks-per-node=1 -p plgrid-c7 -t 00:10:00 mpicc integral_mpi.c -o mpi_integral -lm -ldl -std=c99
mpibuild: mpi_integral




curun: cu_integral
	sbatch integral_cu.sh && squeue

omprun: omp_integral
	sbatch integral_omp.sh && squeue

mpirun: mpi_integral
	sbatch integral_mpi.sh && squeue
