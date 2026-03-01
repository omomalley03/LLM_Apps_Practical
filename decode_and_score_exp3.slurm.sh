#!/bin/bash
#SBATCH -A MLMI-omo26-SL2-GPU
#SBATCH -J DECODE_SCORE_GPT2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=0:30:00
#SBATCH --mail-type=FAIL
#SBATCH -p ampere
#SBATCH --array=0-5   # one job per checkpoint

. /etc/profile.d/modules.sh
module purge
module load rhel8/default-amp
module load cuda/11.1 intel/mkl/2017.4
export OMP_NUM_THREADS=1

module load slurm
module load anaconda/3.2019-10
eval "$(conda shell.bash hook)"
source /rds/project/rds-xyBFuSj0hm0/MLMI8.L2024/envs/README.MLMI8.SEQ2SEQDST.activate

# -------------------------------------------------------
# Checkpoints to decode - corresponds to SBATCH --array=0-5
# Add or remove entries here if needed, and update --array above
# -------------------------------------------------------
CHECKPOINTS=(
    model.60000
    model.100000
    model.120000
    model.160000
    model.180000
    model.200000
)

CHECKPOINT=${CHECKPOINTS[$SLURM_ARRAY_TASK_ID]}
# EXPERIMENT_NAME=gpt2_nlu
EXPERIMENT_NAME=gpt2_exp3


echo "=============================="
echo "Array task:  $SLURM_ARRAY_TASK_ID"
echo "Checkpoint:  $CHECKPOINT"
echo "Time:        $(date)"
echo "Host:        $(hostname)"
echo "=============================="

# -------------------------------------------------------
# Paths
# -------------------------------------------------------
TRAIN_SCRIPT=$BDIR/src/decode-dst.py
SCORE_SCRIPT=$BDIR/src/multiwoz_dst_score.py

# TEST_DATA=$BDIR/data_preparation/data/multiwoz21/processed/test/version_1/data.json
# DEV_DATA=$BDIR/data_preparation/data/multiwoz21/processed/dev/version_1/data.json
TEST_DATA=/rds/user/$USER/hpc-work/data_preparation/data/multiwoz21/processed/test/exp3/data.json
DEV_DATA=/rds/user/$USER/hpc-work/data_preparation/data/multiwoz21/processed/dev/exp3/data.json


TEST_REFS=$BDIR/data_preparation/data/multiwoz21/refs/test/test_v2.1.json
DEV_REFS=$BDIR/data_preparation/data/multiwoz21/refs/dev/dev_v2.1.json

TEST_HYP_DIR=hyps/test/$EXPERIMENT_NAME/$CHECKPOINT
DEV_HYP_DIR=hyps/dev/$EXPERIMENT_NAME/$CHECKPOINT

SCORE_DIR=scores/$EXPERIMENT_NAME
mkdir -p $TEST_HYP_DIR $DEV_HYP_DIR $SCORE_DIR

SCORE_FILE=$SCORE_DIR/$CHECKPOINT.txt

# -------------------------------------------------------
# Decode: test set
# -------------------------------------------------------
echo ""
echo "--- Decoding TEST set for $CHECKPOINT ---"
python $BDIR/src/decode-dst.py \
    decode=gpt2 \
    override=True \
    decode.hyp_dir=hyps/test \
    decode.test_data=$TEST_DATA \
    decode.checkpoints_dir=checkpoints/$EXPERIMENT_NAME \
    decode.checkpoint=$CHECKPOINT \
    decode.experiment_name=$EXPERIMENT_NAME

# -------------------------------------------------------
# Score: test set (NLU, turn-level)
# -------------------------------------------------------
echo ""
echo "--- Scoring TEST set for $CHECKPOINT ---"
echo "=== TEST NLU (turn-level) ===" >> $SCORE_FILE
python $SCORE_SCRIPT \
    dst_reference=$TEST_REFS \
    hypothesis=$TEST_HYP_DIR/belief_states.json \
    field=dst_belief_state \
    model_type=decoder | tee -a $SCORE_FILE

# -------------------------------------------------------
# Decode: dev set
# -------------------------------------------------------
echo ""
echo "--- Decoding DEV set for $CHECKPOINT ---"
python $BDIR/src/decode-dst.py \
    decode=gpt2 \
    override=True \
    decode.hyp_dir=hyps/dev \
    decode.test_data=$DEV_DATA \
    decode.checkpoints_dir=checkpoints/$EXPERIMENT_NAME \
    decode.checkpoint=$CHECKPOINT \
    decode.experiment_name=$EXPERIMENT_NAME

# -------------------------------------------------------
# Score: dev set (NLU, turn-level)
# -------------------------------------------------------
echo ""
echo "--- Scoring DEV set for $CHECKPOINT ---"
echo "=== DEV NLU (turn-level) ===" >> $SCORE_FILE
python $SCORE_SCRIPT \
    dst_reference=$DEV_REFS \
    hypothesis=$DEV_HYP_DIR/belief_states.json \
    field=dst_belief_state \
    model_type=decoder | tee -a $SCORE_FILE

# -------------------------------------------------------
# CC-DST: generate dialogue-level belief states
# -------------------------------------------------------
echo ""
echo "--- Running CC-DST for $CHECKPOINT ---"

CC_TEST_OUT=hyps/test/${EXPERIMENT_NAME}_cc_dst/$CHECKPOINT
CC_DEV_OUT=hyps/dev/${EXPERIMENT_NAME}_cc_dst/$CHECKPOINT
mkdir -p $CC_TEST_OUT $CC_DEV_OUT

# python $BDIR/src/cc-dst.py \
#     --nlu_bs $TEST_HYP_DIR/belief_states.json \
#     --nlu_turns $TEST_DATA \
#     --dst_out $CC_TEST_OUT/belief_states.json \
#     --field_name predicted_belief_state

# python $BDIR/src/cc-dst.py \
#     --nlu_bs $DEV_HYP_DIR/belief_states.json \
#     --nlu_turns $DEV_DATA \
#     --dst_out $CC_DEV_OUT/belief_states.json \
#     --field_name predicted_belief_state

python $BDIR/src/cc-dst.py \
    --nlu_bs $TEST_HYP_DIR/belief_states.json \
    --nlu_turns $BDIR/data_preparation/data/multiwoz21/processed/test/version_1/data.json \  # ← original
    --dst_out $CC_TEST_OUT/belief_states.json \
    --field_name predicted_belief_state

python $BDIR/src/cc-dst.py \
    --nlu_bs $DEV_HYP_DIR/belief_states.json \
    --nlu_turns $BDIR/data_preparation/data/multiwoz21/processed/dev/version_1/data.json \   # ← original
    --dst_out $CC_DEV_OUT/belief_states.json \
    --field_name predicted_belief_state
    
# -------------------------------------------------------
# Score: CC-DST (dialogue-level)
# -------------------------------------------------------
echo ""
echo "--- Scoring CC-DST for $CHECKPOINT ---"
echo "=== TEST CC-DST (dialogue-level) ===" >> $SCORE_FILE
python $SCORE_SCRIPT \
    dst_reference=$TEST_REFS \
    hypothesis=$CC_TEST_OUT/belief_states.json \
    field=dst_belief_state \
    model_type=decoder | tee -a $SCORE_FILE

echo "=== DEV CC-DST (dialogue-level) ===" >> $SCORE_FILE
python $SCORE_SCRIPT \
    dst_reference=$DEV_REFS \
    hypothesis=$CC_DEV_OUT/belief_states.json \
    field=dst_belief_state \
    model_type=decoder | tee -a $SCORE_FILE

echo ""
echo "=============================="
echo "Done. Scores written to $SCORE_FILE"
echo "Time: $(date)"
echo "=============================="