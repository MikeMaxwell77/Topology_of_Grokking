# Experiment 4: restore activations after disruption

Evaluation-only rescue control using trusted untouched Experiment 1 checkpoints.
No retraining or checkpoint changes. This implements the immediate inverse control
in paper.md; subtracting a translation is algebraically guaranteed to recover the
original representation up to floating-point roundoff. Recovery does not independently
establish that clusters cause grokking. Selective component rescue and post-training
recovery studies remain future work.

Fit k=8 k-means on training equals-token post-block residuals only, seed 601.
Assign held-out examples to their nearest training-fitted center, without labels.
For each strength 0, .25, .5, 1, translate by
`delta = -strength * (assigned_center - training_mean)`.
Evaluate clean `z`, disrupted `z + delta`, and rescued `(z + delta) - delta`.
Reuse exactly the same displacement for rescue; do not refit or reassign groups.
Also disrupt and invert five independent random-direction and five shared-rotation
controls with matching per-example displacement lengths. Zero strength runs once.

Use all coarse baseline checkpoints at steps 0,1000,...,40000. Interpret phases
using Experiment 1's untouched behavioral rules, without selecting maximum damage.
Save every condition's train/held-out loss, accuracy, geometry using original
assignments, and maximum activation/logit deviation from clean. The report summarizes
maximum rescued activation, logit, and loss errors across all conditions. Examine
the actual disruption effects too; recovery from negligible damage is uninformative.
Control draws are not independent model replicates.

Submit from the cluster repository root after syncing this branch:

```bash
sbatch run_experiment04.slurm
# Override the default Experiment 1 job 22004279 baseline location:
sbatch run_experiment04.slurm /path/to/baseline
```

The script requires the completed step-40000 baseline checkpoint and runs offline
on CPU in the established GPU partition. The 48-hour allocation is unbenchmarked.
It creates a fresh `outputs/experiment04_JOBID` containing results.json, protocol.json,
report.md, and per-checkpoint perturbations with centers, assignments, split indices,
and the exact displacement tensors. Creating this setup submits no job.

Direct invocation (quote the wildcard for Python to expand):

```bash
python -m grokking.rescue --checkpoints 'BASELINE/checkpoint_*000.pt' --output-dir outputs/experiment04_rescue
```

Local validation:

```powershell
.venv/Scripts/python.exe -m unittest discover -s tests -p test_rescue.py -v
```

The tiny test checks recovery after nonzero disruption for both control families,
saved perturbation replay, output completeness, and source checkpoint preservation.
This setup has not been executed on the cluster.
