#!/bin/bash

#SBATCH --time=14:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -C TitanX
#SBATCH --gres=gpu:1


# Load GPU drivers
module load cuda10.0/toolkit
module load cuDNN/cuda10.0


# This loads the anaconda virtual environment with our packages
source /home/mms496/.bashrc


# Base directory for the experiment
cd /home/mms496/StyleVAE_Experiments/mnist

# Simple trick to create a unique directory for each run of the script

echo $$
mkdir o`echo $$`
cd o`echo $$`


# Run the actual experiment

python -u /home/mms496/StyleVAE_Experiments/code/StyleVAE/stylevae.py  --task mnist --numplots 4000 -z 512 -e 10 10 10 10 10 10000 -l 0.0001 -b 128  --betas 0.2 1 0 0 0 0 0 0 0.05 --dropouts 0 0 0 0 0 0 0 --mapping-layers 3 -EU 1  --skip-test -EN 1 -DE 2 -D /var/scratch/mms496/data/mnist/
