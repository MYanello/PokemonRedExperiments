#!/bin/bash
#SBATCH --job-name=pokemon-rl
#SBATCH --output=video/logs/std.out
#SBATCH --error=video/logs/std.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus=1

mkdir -p video
mkdir -p video/logs
cd /mnt/data/PokemonRedExperiments/v2
source .venv/bin/activate
python video.py
