#!/bin/bash -l
#SBATCH -J omp_integral 
## Get one node
#SBATCH -N 1
#SBATCH --partition=plgrid-c7
#SBATCH --reservation=piask
#SBATCH --ntasks-per-node=4
## Specify exec time, queue and output
#SBATCH --time=00:10:00
#SBATCH --output=omp_integral.out
## Select module and run task

module add plgrid/tools/intel
cd $HOME/labwork/PIASK_Project
./omp_integral
