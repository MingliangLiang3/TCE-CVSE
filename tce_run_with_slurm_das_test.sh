#!/bin/bash
#SBATCH --partition=das
#SBATCH --account=das
#SBATCH --gres=gpu:rtx_3090:1
#SBATCH --mem=10G
#SBATCH --cpus-per-task=2
#SBATCH --time=6:00:00
#SBATCH --output=logs/test/my-experiment-%j.out
#SBATCH --error=logs/test/my-experiment-%j.err
#SBATCH --mail-user=name
#SBATCH --mail-type=BEGIN,END,FAIL

source ../../venv/barebones-pytorch-venv/bin/activate
DATA_PATH="/ceph/das-scratch/users/mliang/data/scan_data/data"

srun python evaluate.py --data_path $DATA_PATH \
  --data_name "coco_precomp" \
  --model_path "./runs/coco/TCE_CVSE_COCO/model_best.pth.tar" \
  --data_name_vocab coco_precomp \
  --split testall \
  --test_on "prediction" |& tee -a logs/my-experiment_test.log
