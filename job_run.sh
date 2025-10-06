#!/bin/bash
#SBATCH --job-name=test_qwen
#SBATCH --output=/scratch/data/bikash_rs/vivek/MER-Factory/logs/%x_%j.out
#SBATCH --error=/scratch/data/bikash_rs/vivek/MER-Factory/logs/%x_%j.err
#SBATCH --partition=dgx
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
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
python test.py