#!/bin/bash
#SBATCH -J instruct_qwen2_1.5b
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

BASE_DIR=~/rds/rds-gvdd-Yuap0gjVpKM/shared_space/Mphil-VLM-Data/LLaVA-Instruct-150K

python train.py \
    --stage instruction \
    --model_dir ./checkpoints/alignment_qwen2_1.5b/final_model \
    --train_json $BASE_DIR/complex_reasoning_77k.json $BASE_DIR/detail_23k.json $BASE_DIR/conversation_58k.json \
    --images_dir $BASE_DIR/train2014 \
    --output_dir ./checkpoints/instruction_qwen2_1.5b \
    --batch_size 8 \
    --gradient_accumulation_steps 8 \
    --epochs 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --dataloader_num_workers 12 \
    --run_name "VLM_Instruct_Qwen2_1.5B_$(date +%Y%m%d_%H%M%sS)" \
    --check_img \