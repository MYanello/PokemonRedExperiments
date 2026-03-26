#!/bin/bash
#SBATCH --job-name=pokemon-rl
#SBATCH --output=logs/std.out
#SBATCH --error=logs/std.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --export=WANDB_API_KEY
#SBATCH --gpus=1

mkdir -p logs
mkdir -p runs/checkpoints

cd /mnt/data/PokemonRedExperiments/v2
source .venv/bin/activate

latest_file=$(ls -lt runs/checkpoints/*.zip 2>/dev/null | awk 'NR==1 {print $9}' | sed 's/\.zip$//')
if [ -n "$latest_file" ]; then
    echo "Resuming from checkpoint: $latest_file"
    echo "$latest_file" | python baseline_fast_v2.py
else
    echo "Starting fresh"
    python baseline_fast_v2.py
fi
