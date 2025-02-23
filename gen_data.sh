#!/bin/bash
#SBATCH --job-name=generate_cifar
#SBATCH --output=/data/user_data/$USER/neural-lifting/logs/generate_cifar.out
#SBATCH --error=/data/user_data/$USER/neural-lifting/logs/generate_cifar.err
#SBATCH --time=01:00:00
#SBATCH --partition=debug
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G

set -e  # Stop the script if an error occurs

# Source the conda.sh script and activate the conda environment
source /home/$USER/miniconda/etc/profile.d/conda.sh
conda activate venv

python "/data/user_data/$USER/neural-lifting/gen_data.py"  # Run the script