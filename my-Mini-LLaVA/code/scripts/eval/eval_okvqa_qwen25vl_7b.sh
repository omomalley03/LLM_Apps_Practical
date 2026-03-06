#!/bin/bash
#SBATCH -J eval_qwen25vl_7b
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

python eval/eval_okvqa_hf.py \
    --model_name Qwen/Qwen2.5-VL-7B-Instruct  \
    --okvqa_questions $BASE_DIR/okvqa/val_questions.json \
    --okvqa_annotations $BASE_DIR/okvqa/val_annotations.json \
    --images_dir $BASE_DIR/okvqa/val2014 \
    --output_dir ./output/okvqa_val/qwen2.5-vl-7b \
    --batch_size 4 \
    --num_workers 12