#!/bin/bash
#SBATCH --partition=das
#SBATCH --account=das
#SBATCH --gres=gpu:1
#SBATCH --mem=10G
#SBATCH --cpus-per-task=4
#SBATCH --time=48:00:00
#SBATCH --output=logs/train/my-experiment-%j.out
#SBATCH --error=logs/train/my-experiment-%j.err
#SBATCH --mail-user=name
#SBATCH --mail-type=BEGIN,END,FAIL

source ../../venv/barebones-pytorch-venv/bin/activate
DATA_PATH="./data/coco_annotations/Concept_annotations_coco_vocab"
LOG_PATH="./runs/coco_new/"

srun python train_coco.py --data_path "../../data/scan_data/data" \
  --batch_size 512 \
  --num_attribute 300 \
  --model_name "$LOG_PATH/CVSE_COCO_data_omp_train_top_300_five_caption/" \
  --concept_name "$DATA_PATH/data_omp_train_top_300_five_caption/category_concepts.json" \
  --inp_name "$DATA_PATH/data_omp_train_top_300_five_caption/coco_concepts_glove_word2vec.pkl" \
  --resume "none" \
  --adj_file "$DATA_PATH/data_omp_train_top_300_five_caption/coco_adj_concepts.pkl" \
  --adj_gen_mode "ReComplex" \
  --t 0.3 \
  --alpha 0.9 \
  --attribute_path "$DATA_PATH/data_omp_train_top_300_five_caption/" \
  --test_on "five" \
  --re_weight 0.2 \
  --logger_name "$LOG_PATH/CVSE_COCO/data_omp_train_top_300_five_caption_prediction/"