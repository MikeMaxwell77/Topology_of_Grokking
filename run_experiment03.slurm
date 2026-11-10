#!/bin/bash
#SBATCH --job-name=grok-timing-03
#SBATCH --output=experiment03_%j.out
#SBATCH --error=experiment03_%j.err
#SBATCH --gres=gpu:1
#SBATCH --partition=USCB-V100-32gb
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=48:00:00

set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
module load cuda/12.1
source /work/apps/python3/anaconda/2023.9/etc/profile.d/conda.sh
conda activate topology
set -u
export CUDA_HOME=/work/apps/cuda/12.1
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1
checkpoint="${1:?Usage: sbatch run_experiment03.slurm CHECKPOINT DURATION}"
duration="${2:?Specify intervention duration in optimizer updates}"
test -f "$checkpoint"
srun python -u -m grokking.training_interventions \
    --checkpoint "$checkpoint" --duration "$duration" \
    --device cuda --threads 1 --clusters 8 --strength .5 --repeats 5 --seed 601 \
    --output-dir "outputs/experiment03_${SLURM_JOB_ID}"
