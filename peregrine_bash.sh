#!/bin/bash

#SBATCH --partition=gpushort
#SBATCH --gres=gpu:1

module load Python PyTorch

python3 launch.py