#!/bin/bash
#SBATCH --partition=das
#SBATCH --account=das
#SBATCH --gres=gpu:0
#SBATCH --mem=5G
#SBATCH --cpus-per-task=2
#SBATCH --time=48:00:00
#SBATCH --output=my-experiment-%j.out
#SBATCH --error=my-experiment-%j.err
#SBATCH --mail-user=mliang
#SBATCH --mail-type=BEGIN,END,FAIL

. /ceph/das-scratch/users/mliang/venv/bin/activate
# srun python gan_vocab.py 2>&1 | tee my-experiment.log