#!/bin/bash

#SBATCH --job-name=pytorchtest
#SBATCH --time=0-0:05
#SBATCH --mem-per-cpu=16G
#SBATCH --output torch_test.txt
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --exclude=smp-6-[5,6]

module load pytorch 
python3 torch_test.py 
