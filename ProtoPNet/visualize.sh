#!/usr/bin/env bash

#SBATCH --job-name=protopnet_audio     # Job name
#SBATCH --ntasks=1                    # Run on a single Node
#SBATCH --cpus-per-task=4
#SBATCH --mem=160gb                  # Job memory request
#SBATCH --time=15:00:00                # Time limit hrs:min:sec
#SBATCH --partition=compsci-gpu
#SBATCH --gres=gpu:1

source /home/users/aak61/music_genre_classification/venv
python3 visualize_prototypes_tsne.py --configs="configs/gtzan.yaml"
