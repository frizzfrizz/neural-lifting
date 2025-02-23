#!/bin/bash
# Source environment variables

BASE_DIR="/data/user_data/$USER/neural-lifting"

#SBATCH --job-name=train_lenet
#SBATCH --output=${BASE_DIR}/logs/train_nl_%j.out
#SBATCH --error=${BASE_DIR}/logs/train_nl_%j.err
#SBATCH --gres=gpu:1
#SBATCH --time=1:00:00
#SBATCH --mem=32G
#SBATCH --partition=debug

# venv
source /home/$USER/miniconda/etc/profile.d/conda.sh
conda activate venv 

# Set base directories
BASE_DIR="/data/user_data/$USER/neural-lifting"
mkdir -p ${BASE_DIR}/{results/run_val,logs}

# Set CUDA device
# Check if NVIDIA GPU is available
if command -v nvidia-smi &> /dev/null && nvidia-smi &> /dev/null; then
    echo "GPU available, using GPU"
    export CUDA_VISIBLE_DEVICES=0
else
    echo "No GPU available, using CPU"
    export CUDA_VISIBLE_DEVICES=""
fi

# Run training script
python3 main_nl.py \
    --dataset "${BASE_DIR}/datasets/cifar10" \
    --results_directory "${BASE_DIR}/results/cifar_10_results/run_val/run_1/" \
    --config_path "${BASE_DIR}/configs/cifar_10/model_configs.json" \
    --seed 42 \
    --models "LeNetLifted" \
    --with_val 2>&1 | tee "${BASE_DIR}/logs/train_nl_%j.log"

# EOF