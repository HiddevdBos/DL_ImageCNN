#!/bin/bash

#SBATCH --time=01:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

module load Python matplotlib
module load libpciaccess/0.16-GCCcore-9.3.0

#python3 launch.py
#python3 launch.py optimizer adam
python3 launch.py optimizer rmsprop
#python3 launch.py optimizer sgd
