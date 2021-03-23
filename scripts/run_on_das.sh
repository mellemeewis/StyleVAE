#!/bin/bash

#SBATCH --time=00:15:00
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
cd /home/mms496/


# Simple trick to create a unique directory for each run of the script

echo $$
mkdir o`echo $$`
cd o`echo $$`


# Run the actual experiment

python -u /home/mms496/code/StyleVAE/stylevae.py --task ffhq --numplots 300 -z 256 -e 10 10 10 10 10 1000 -l 0.00005 -b 32 --betas 0.6 1 1 1 1 1 1 1 --dropouts 0.01 0.01 0.01 0.01 0.01 0.01 0.01 --mapping-layers 6 -D /home/mms496/data/thumbnails128x128
