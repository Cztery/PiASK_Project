SHELL:=/bin/bash

all: curun omprun mpirun

cubuild:
	command -v nvcc || module add plgrid/apps/cuda; \
	nvcc integral.cu -o cu_integral

ompbuild:
	command -v icc || module add plgrid/tools/intel; \
	icc -openmp integral_omp.c my_timers.c -o omp_integral

mpibuild:
#because we can compile only in interactive mode, open bash and manually enter compile command
	command -v mpicc || module add plgrid/tools/openmpi; \
	srun -N 1 --ntasks-per-node=4 -p plgrid-c7 -t 01:30:00 --pty /bin/bash

curun: cubuild
	sbatch integral_cu.sh && squeue

omprun: ompbuild
	sbatch integral_omp.sh && squeue

mpirun: mpibuild
	sbatch integral_mpi.sh && squeue
