#!/bin/bash
#SBATCH -J eval_qwen2_1.5b
#SBATCH -A 
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=36:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
##SBATCH --no-requeue
#SBATCH -p ampere

export HF_HOME=~/rds/hpc-work/cache
BASE_DIR=~/rds/rds-mlmi-2020-21-xyBFuSj0hm0/MLMI8.L2026/VLM

python eval/eval_okvqa_vlm.py \
    --model_dir $BASE_DIR/checkpoints/instruction_qwen2_1.5b/final_model \
    --okvqa_questions $BASE_DIR/okvqa/val_questions.json \
    --okvqa_annotations $BASE_DIR/okvqa/val_annotations.json \
    --images_dir $BASE_DIR/okvqa/val2014 \
    --output_dir ./output/okvqa_val/vlm_qwen2_1.5b \
    --batch_size 16 \
    --num_workers 12