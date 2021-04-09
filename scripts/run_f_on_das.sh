#!/bin/bash

#SBATCH --time=48:00:00
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
cd /home/mms496/StyleVAE_Experiments/full

# Simple trick to create a unique directory for each run of the script

echo $$
mkdir o`echo $$`
cd o`echo $$`


# Run the actual experiment

python -u /home/mms496/StyleVAE_Experiments/code/StyleVAE/stylevae.py --task ffhq-gs --numplots 4000 -z 512  -e 100 100 100 100 100 100000 -l 0.00005 -b 128 --betas 1 1 0 0 0 0 0 1 --dropouts 0 0 0 0 0 0 0 --mapping-layers 3 --skip-test -EN 2 -DE 2 -D /var/scratch/mms496/data/ffhq/full/thumbnails128x128 #--channels 64 128 256 512 1024 --zchannels 4 8 16 32 64 128
