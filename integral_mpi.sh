#!/bin/bash -l
#SBATCH -J mpi_integral 
## Get one node
#SBATCH -N 1
#SBATCH --partition=plgrid-c7
#SBATCH --reservation=piask
#SBATCH --ntasks-per-node=4
#SBATCH --time=00:10:00
#SBATCH --output=mpi_integral.out

## Select module and run task
module add plgrid/tools/openmpi
cd $HOME/labwork/PiASK_Project
mpiexec mpi_integral
