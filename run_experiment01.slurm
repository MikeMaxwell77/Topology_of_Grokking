#!/bin/bash
#SBATCH --job-name=grok-clusters-01
#SBATCH --output=experiment01_%j.out
#SBATCH --error=experiment01_%j.err
#SBATCH --gres=gpu:1
#SBATCH --partition=USCB-V100-32gb
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=48:00:00

# Submit from the repository root: sbatch run_experiment01.slurm
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

experiment_dir="outputs/experiment01_${SLURM_JOB_ID}"
mkdir -p "$experiment_dir"
echo "Experiment 1: GPU baseline followed by CPU evaluation-only interventions"
echo "Results: $experiment_dir"
nvidia-smi
srun python -c 'import torch; print("Torch:", torch.__version__); assert torch.cuda.is_available(), "CUDA-enabled PyTorch is required"; print("GPU:", torch.cuda.get_device_name(0))'

srun python -u -m grokking.study train \
    --device cuda --threads 1 \
    --modulus 113 --train-fraction 0.3 \
    --d-model 128 --n-heads 4 --d-ff 512 \
    --learning-rate 0.001 --weight-decay 1.0 \
    --beta1 0.9 --beta2 0.98 --init-std 0.1 \
    --seed 42 --split-seed 598 --probe-seed 599 \
    --steps 40000 --checkpoint-interval 50 --evaluation-interval 50 \
    --output-dir "$experiment_dir/baseline"

# Analysis intentionally uses CPU and loads checkpoints without modifying them.
# Quote the pattern: the Python CLI expands it, including on Windows.
srun python -u -m grokking.interventions \
    --checkpoints "$experiment_dir/baseline/checkpoint_*000.pt" \
    --clusters 8 --strengths 0 0.25 0.5 1 --repeats 5 --seed 601 \
    --output-dir "$experiment_dir/sweep"

echo "Complete. Read $experiment_dir/sweep/report.md and contrasts.csv"
