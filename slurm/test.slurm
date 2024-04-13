#!/bin/bash

#SBATCH --job-name=pytorchtest
#SBATCH --time=0-0:05
#SBATCH --mem=16G
#SBATCH --output torch_test.txt
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --exclude=smp-6-[5,6]

module load pytorch 
python 3 -m pip install --user -e .
python3 scripts/torch_test.py 
