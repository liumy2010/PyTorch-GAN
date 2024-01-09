#!/bin/bash
#SBATCH --gres=gpu:volta:1
#SBATCH -n 8
#SBATCH -N 1

# Run the script
python dcgan.py --channels 3 --img_size 32 --opt_level 1 &
python dcgan.py --channels 3 --img_size 32 --opt_level 2 &
python dcgan.py --channels 3 --img_size 32 --opt_level 4 &
python dcgan.py --channels 3 --img_size 32 --opt_level 8 &
python dcgan.py --channels 3 --img_size 32 --opt_level 16 