#!/bin/bash
#SBATCH -J trainning
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --output=log.%j.out
#SBATCH -t 30:00:00
#SBATCH --gres=gpu:1
module load anaconda3/5.3.0 cuda/9.0 cudnn/7.0.5
source activate base
python -u train.py

