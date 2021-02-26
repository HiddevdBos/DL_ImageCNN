#!/bin/bash

#SBATCH --time=04:30:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

module load Python matplotlib
module load libpciaccess/0.16-GCCcore-9.3.0

python3 launch.py
python3 launch.py optimizer adam
python3 launch.py optimizer rmsprop
python3 launch.py type one-layer
python3 launch.py type sigmoid
python3 launch.py type leaky-relu
python3 launch.py type no-batch
python3 launch.py dropout
python3 launch.py type lenet5relu
python3 launch.py type lenet5tanh
#python3 launch.py type learning_rate lenet5

#python3 launch.py type learning_rate standard
