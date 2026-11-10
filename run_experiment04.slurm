#!/bin/bash
#SBATCH --job-name=grok-rescue-04
#SBATCH --output=experiment04_%j.out
#SBATCH --error=experiment04_%j.err
#SBATCH --gres=gpu:1
#SBATCH --partition=USCB-V100-32gb
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=48:00:00

# Established partition; this evaluation-only sweep runs on CPU.
set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
source /work/apps/python3/anaconda/2023.9/etc/profile.d/conda.sh
conda activate topology
set -u
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1
baseline="${1:-outputs/experiment01_22004279/baseline}"
if [[ ! -f "$baseline/checkpoint_00040000.pt" ]]; then
    echo "Missing completed baseline: $baseline/checkpoint_00040000.pt" >&2
    echo "Usage: sbatch run_experiment04.slurm /path/to/baseline" >&2
    exit 1
fi
srun python -u -m grokking.rescue \
    --checkpoints "$baseline/checkpoint_*000.pt" \
    --clusters 8 --strengths 0 .25 .5 1 --repeats 5 --seed 601 \
    --output-dir "outputs/experiment04_${SLURM_JOB_ID}"
