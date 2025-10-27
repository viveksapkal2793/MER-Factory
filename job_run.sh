#!/bin/bash
#SBATCH --job-name=3_train_annotation
#SBATCH --output=/scratch/data/bikash_rs/vivek/MER-Factory/logs/%x_%j.out
#SBATCH --error=/scratch/data/bikash_rs/vivek/MER-Factory/logs/%x_%j.err
#SBATCH --partition=fat
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=5-00:00:00
#SBATCH --qos=fatqos
#SBATCH -D /scratch/data/bikash_rs/vivek/MER-Factory

# Create logs directory
mkdir -p logs

# Load CUDA module (adjust version based on your system)
# module load cuda/11.8

# Activate virtual environment
source mer-factory-env/bin/activate

# Set environment variables
export HF_HOME=/scratch/data/bikash_rs/vivek/huggingface_cache
export CUDA_VISIBLE_DEVICES=0
export TRANSFORMERS_CACHE=$HF_HOME
export HF_DATASETS_CACHE=$HF_HOME
mkdir -p $HF_HOME

# Run test script
# python test.py
python main.py /scratch/data/bikash_rs/vivek/dataset/MELD.Raw/train_splits_3 /scratch/data/bikash_rs/vivek/dataset/Meld_feat_ext/train_annotation --type mer --huggingface-model "Qwen/Qwen2.5-Omni-3B" --cache --threshold 0.8 --peak_dis 15 --concurrency 1