#!/bin/bash
#SBATCH --partition=das
#SBATCH --account=das
#SBATCH --gres=gpu:1
#SBATCH --mem=10G
#SBATCH --cpus-per-task=6
#SBATCH --time=48:00:00
#SBATCH --output=logs/test/my-experiment-%j.out
#SBATCH --error=logs/test/my-experiment-%j.err
#SBATCH --mail-user=name
#SBATCH --mail-type=BEGIN,END,FAIL

source ../../venv/pytorch-cuda11-3-venv/bin/activate
srun python ML-Classifier/multi-label-text-classification-with-bert-test.py
