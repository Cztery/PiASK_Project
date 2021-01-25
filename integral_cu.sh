#!/bin/bash -l
#SBATCH -J first_gpu_job
## Get one node, one CPU-GPU pair
#SBATCH -N 1
#SBATCH --partition=plgrid-gpu
#SBATCH --reservation=piask_gpu
#SBATCH --gres=gpu:1
## Specify exec time, queue and output
#SBATCH --time=00:10:00
#SBATCH --partition=plgrid-gpu
#SBATCH --output=cu_integral.out
## Select module and run task
cd $HOME/labwork/PiASK_Project
./cu_integral
