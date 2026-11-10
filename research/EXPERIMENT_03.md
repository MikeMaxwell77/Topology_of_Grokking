# Experiment 3: temporary interference during learning

Exploratory timing experiment, implemented but not yet run on the cluster.
Branch one trusted Experiment 1 checkpoint into a clean continuation, center
contraction, five independent random-direction controls, and five shared-rotation
controls. Each arm restores the same model, AdamW state, RNG state, split, and
full-batch order. Continue to the original 40,000-update budget.

Choose and record the checkpoint from untouched baseline behavior before examining
branch effects: after memorization, during the poor-generalization plateau, before
generalization. Use Experiment 1's behavioral thresholds and report when no such
window exists. The script deliberately requires an explicit checkpoint and duration.
Record the selection rationale and fixed duration before submission. Additional
windows, strengths, and seeds are separate exploratory comparisons.

Fit training-only k=8 k-means at branch start, seed 601. Freeze centers and each
example's assignments, including nearest-center assignments for held-out examples.
Apply `delta_i = -.5 * (center_i - training_mean)` for exactly D updates after
the branch step. Controls preserve each example's displacement norm and remain
fixed throughout training. No held-out labels enter fitting. The backward derivative
of this fixed translation is identity. Remove it for all remaining updates.

Every evaluation records clean and counterfactually active loss, accuracy, and
geometry for both splits. Active evaluations after removal are diagnostic only.
Geometry uses original assignments; changing geometry need not retain those groups.
All arms save their initial state, periodic checkpoints, removal boundary, and final
state. Source provenance and fixed perturbations are saved beside per-arm histories.
The source baseline is never overwritten. Branch checkpoints are for inspection,
not supported by the ordinary study resume command.

Primary outcome: clean held-out generalization onset using >=95% accuracy at five
consecutive evaluations, compared separately with clean continuation and each control
family. Report absent onsets as censored at the budget, not as zero delay. Also report
training effects, clean loss, recovery after removal, geometry, and all arms. Use
the same timing rules as Experiment 1; the added branch/removal evaluations must be
identified when interpreting timing resolution. A deficit confined to active
evaluation is execution disruption, not evidence of delayed learning. Control draws
are not independent model replicates. Fresh baseline seeds precede causal claims.
Timing aggregation is manual from saved histories in this first implementation.

Submit from the cluster repository root after syncing the branch:

```bash
sbatch run_experiment03.slurm /path/to/baseline/checkpoint_STEP.pt DURATION
```

Replace STEP and DURATION with the behavior-selected step and predeclared number
of updates. The runner requires the original device and Torch thread count; use
the original Torch version and GPU backend for trajectory comparisons. Twelve arms
run sequentially, so budget substantially more runtime and storage than one baseline.
The 48-hour allocation has not been benchmarked. No cluster job is submitted by
creating this script. Results go to `outputs/experiment03_JOBID`.
