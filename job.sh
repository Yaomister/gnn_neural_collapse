#!/bin/bash

#SBATCH --job-name=neural-collapse
#SBATCH --parition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpu-per-taks=4
#SBATCH --time=4:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err


mkdir -p figures logs results

module load anaconda3

source activate myenv

cd $SLURM_SUBMIT_DIR

python main.py