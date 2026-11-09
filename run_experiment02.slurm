#!/bin/bash
#SBATCH --job-name=grok-clusters-02
#SBATCH --output=experiment02_%j.out
#SBATCH --error=experiment02_%j.err
#SBATCH --gres=gpu:1
#SBATCH --partition=USCB-V100-32gb
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=48:00:00

# GPU allocation uses the established cluster partition; offline sweeps run on CPU.
# Optional argument: path to the completed Experiment 1 baseline directory.
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
    echo "Submit with: sbatch run_experiment02.slurm /path/to/baseline" >&2
    exit 1
fi
output="outputs/experiment02_${SLURM_JOB_ID}"
echo "Experiment 2: evaluating saved checkpoints from $baseline"
for intervention in tighten separate; do
    srun python -u -m grokking.interventions \
        --checkpoints "$baseline/checkpoint_*000.pt" \
        --intervention "$intervention" \
        --clusters 8 --strengths 0 .25 .5 1 --repeats 5 --seed 601 \
        --output-dir "$output/$intervention"
done
echo "Complete. Reports: $output/tighten/report.md and $output/separate/report.md"
