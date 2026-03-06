#!/bin/bash
#SBATCH -J alignment_qwen2_1.5b
#SBATCH -A 
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=36:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
##SBATCH --no-requeue
#SBATCH -p ampere

# Set up wandb 
export WANDB_PROJECT=Mphil_VLM
export WANDB_ENTITY=byrne-lab

BASE_DIR=~/rds/rds-gvdd-Yuap0gjVpKM/shared_space/Mphil-VLM-Data/LLaVA-CC3M-Pretrain-595K

python train.py \
    --stage alignment \
    --train_json $BASE_DIR/chat.json \
    --images_dir $BASE_DIR/image/images \
    --output_dir ./checkpoints/alignment_qwen2_1.5b \
    --llm_name Qwen/Qwen2-1.5B-Instruct \
    --vision_encoder_name openai/clip-vit-large-patch14 \
    --batch_size 16 \
    --gradient_accumulation_steps 8 \
    --epochs 1 \
    --learning_rate 1e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --dataloader_num_workers 12 \
    --run_name "VLM_Alignment_Qwen2_1.5B_$(date +%Y%m%d_%H%M%S)"
