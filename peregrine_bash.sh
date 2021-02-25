#!/bin/bash

#SBATCH --time=01:30:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

module load Python matplotlib
module load libpciaccess/0.16-GCCcore-9.3.0

python3 launch.py
python3 launch.py optimizer adam
python3 launch.py optimizer rmsprop
python3 launch.py type one-layer
python3 launch.py type softmax
python3 launch.py type sigmoid
python3 launch.py type leaky-relu
python3 launch.py type no-batch