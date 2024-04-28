#!/usr/bin/env bash
#SBATCH --job-name=my_job # Job name
#SBATCH --output=logs/my_job_log_%j.out
#SBATCH --ntasks=1                    # Run on a single Node
#SBATCH --cpus-per-task=10
#SBATCH --mem=160gb                     # Job memory request
#SBATCH --time=96:00:00               # Time limit hrs:min:sec
#SBATCH --partition=compsci-gpu
#SBATCH --gres=gpu:4

source /home/users/aak61/music_genre_classification/venv
python3 train_backbone.py --configs="configs/gtzan.yaml"