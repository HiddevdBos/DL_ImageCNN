#!/bin/bash

#SBATCH --partition=gpushort
#SBATCH --gres=gpu:1

module load Python PyTorch matplotlib
module load libpciaccess/0.14-GCCcore-8.3.0

python3 launch.py