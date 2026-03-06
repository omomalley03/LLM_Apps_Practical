# MLMI8 LLM Applications Practical

University of Cambridge MLMI8 practical coursework exploring two applications of large language models: **Dialogue State Tracking (DST)** with GPT-2 on MultiWOZ 2.1, and a **Mini Vision Language Model** (Mini-LLaVA).

---

## Repository Structure

```
.
├── train-dst.py                    # GPT-2 DST training script
├── prepare_dst_data.py             # Data preprocessing for Exp2 and Exp3
├── dstdataset.py                   # PyTorch Dataset for DST training
├── analyse_predictions.py          # Prediction error analysis tool
├── analyse_llm_judge.py            # LLM-as-judge analysis script
├── decode_and_score_exp3.slurm.sh  # SLURM job: decode + score Exp3
├── decode_and_score_all.slurm.sh   # SLURM job: decode + score all experiments
├── modelcard.md                    # Model card for GPT2-HistoryDST
├── data_preparation/               # Processed MultiWOZ 2.1 data
│   └── data/multiwoz21/processed/
└── my-Mini-LLaVA/                  # Mini Vision Language Model project
    └── code/
        ├── model.py                # VLM architecture (CLIP + MLP + Qwen2)
        ├── train.py                # Unified training script
        ├── data_preprocess.py      # Data preprocessing for alignment/instruct stages
        ├── interactive_inference.ipynb
        ├── eval/                   # OK-VQA evaluation scripts
        └── scripts/                # SLURM/HPC training scripts
```

---

## Part 1: GPT-2 Dialogue State Tracking

Fine-tuning GPT-2 on the [MultiWOZ 2.1](https://github.com/budzianowski/multiwoz) benchmark for Dialogue State Tracking (DST). Two experimental setups are explored:

| Experiment | Input | Target |
|------------|-------|--------|
| **Exp2** (NLU) | Full dialogue history | Turn-level belief state |
| **Exp3** (DST) | Full dialogue history | Cumulative belief state |

The model receives the full dialogue history up to each turn (formatted as `<SYS> ... <USR> ...`) and generates a linearized belief state: `domain slot value <SEP> domain slot value ...`.

### Results (GPT2-HistoryDST, Exp3)

| Checkpoint | Dev Joint Accuracy | Test Joint Accuracy |
|------------|--------------------|---------------------|
| 60,000     | 51.10%             | 49.29%              |
| 100,000    | **52.45%**         | **51.26%**          |
| 120,000    | 51.06%             | 50.15%              |
| 200,000    | 52.41%             | 51.19%              |

Best checkpoint: `model.100000` (52.45% dev, 51.26% test joint accuracy).

### Data Preparation

```bash
# Experiment 2: history input + turn-level target
python prepare_dst_data.py \
    --input_file data_preparation/data/multiwoz21/processed/train/version_1/data.json \
    --output_dir data_preparation/data/multiwoz21/processed/train/exp2 \
    --mode exp2

# Experiment 3: history input + cumulative target (dev split, with refs)
python prepare_dst_data.py \
    --input_file data_preparation/data/multiwoz21/processed/dev/version_1/data.json \
    --refs_file data_preparation/data/multiwoz21/refs/dev/dev_v2.1.json \
    --output_dir data_preparation/data/multiwoz21/processed/dev/exp3 \
    --mode exp3
```

Run for `train`, `dev`, and `test` splits separately. For the train split, cumulative belief states are constructed from turn-level annotations (no refs file needed).

### Training

Training uses [Hydra](https://hydra.cc/) for configuration management and logs to [WandB](https://wandb.ai/).

```bash
python train-dst.py
```

Configuration is loaded from `config/train_conf.yaml`. Key hyperparameters:
- **Optimizer:** AdamW
- **Batch size:** 8
- **Hardware:** NVIDIA A100-SXM4-80GB
- **Framework:** HuggingFace Transformers + PyTorch

### Decoding and Scoring (HPC/SLURM)

```bash
# Submit array job to decode and score all Exp3 checkpoints
sbatch decode_and_score_exp3.slurm.sh
```

This runs decoding on both test and dev sets for 6 checkpoints in parallel, scores NLU (turn-level) and CC-DST (dialogue-level) accuracy, and writes results to `scores/`.

### Prediction Analysis

```bash
# Analyse errors for DST (dialogue-level)
python analyse_predictions.py \
    --refs data_preparation/data/multiwoz21/refs/test/test_v2.1.json \
    --hyps hyps/test/gpt2_exp3/model.100000/belief_states.json \
    --field dst_belief_state \
    --n 5
```

Outputs a breakdown of correct predictions, errors (missing/hallucinated slots), and bad-format cases.

---

## Part 2: Mini-LLaVA

A minimalistic Vision Language Model (VLM) following a two-stage training approach inspired by [LLaVA](https://llava-vl.github.io/). See [`my-Mini-LLaVA/code/README.md`](my-Mini-LLaVA/code/README.md) for full details.

### Architecture

- **Vision Encoder:** OpenAI CLIP ViT-L/14 (frozen)
- **Mapping Network:** Custom MLP (trained from scratch) to project visual features into the LLM's token embedding space
- **LLM:** Qwen2-1.5B-Instruct

### Training Stages

1. **Alignment:** Train only the mapping network to align visual and language representations
2. **Instruction Tuning:** Fine-tune the mapping network (and optionally the LLM) on visual instruction-following data

### Evaluation

Evaluated on [OK-VQA](https://okvqa.allenai.org/) using VQA accuracy. Baselines include Qwen2.5-VL-3B and Qwen2.5-VL-7B.

### Setup

```bash
cd my-Mini-LLaVA/code
pip install -r requirements.txt
```

---

## Dependencies

- Python 3.10+
- PyTorch
- HuggingFace Transformers
- Hydra (`omegaconf`, `hydra-core`)
- WandB
- tqdm

---

## References

- Budzianowski et al., [MultiWOZ — A Large-Scale Multi-Domain Wizard-of-Oz Dataset](https://arxiv.org/abs/1810.00278), EMNLP 2018
- Wolf et al., [HuggingFace Transformers](https://arxiv.org/abs/1910.03771)
- Liu et al., [Visual Instruction Tuning (LLaVA)](https://arxiv.org/abs/2304.08485), NeurIPS 2023
- Marino et al., [OK-VQA: A Visual Question Answering Benchmark Requiring External Knowledge](https://arxiv.org/abs/1906.00067), CVPR 2019
