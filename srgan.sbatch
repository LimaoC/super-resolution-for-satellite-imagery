#!/bin/bash

#SBATCH --job-name=srgan
#SBATCH --time=0-0:50
#SBATCH --mem=32G
#SBATCH --output torch_test.txt
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --exclude=smp-6-[5,6]

module load pytorch 
python3 prototype_srgan.py 
