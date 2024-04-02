#!/bin/bash

#SBATCH --job-name=pytorchtest
#SBATCH --time=0-5:00
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=1G
#SBATCH --output torch_test.txt


module load anaconda3/2020.11
module load cuda/10.0

python3 torch_test.py 
