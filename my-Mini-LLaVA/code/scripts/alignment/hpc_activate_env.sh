#!/bin/bash
module purge
module load slurm
module load rhel8/default-amp
# module load cuda/11.4
module load cuda/12.1
module load gcc/9

export HF_HOME='../HF_HOME'
