# Experiment 2: does strengthening clustering improve generalization?

Exploratory evaluation-only sufficiency test. Reuse the untouched Experiment 1
baseline checkpoints; do not retrain or modify the saved model or readout.

## Protocol

Keep Experiment 1's split, representation site, training-only k=8 k-means fit,
seed 601, strengths 0, .25, .5, 1, and five draws per matched-control family.
Evaluate two separate interventions with original nearest-center assignments:

- Tighten: `z' = z - beta * (z - center)`. Residuals shrink by `1-beta`.
- Separate: `z' = z + beta * (center - training_mean)`. Center offsets expand
  by `1+beta`, preserving within-group offsets.

No held-out labels enter fitting or transformation. Held-out labels are used only
for scoring. Independent random directions and shared orthogonal rotations match
each example's targeted displacement magnitude, separately for each intervention.

Use all checkpoints at steps 0,1000,...,40000 for the coarse sweep, regardless of
observed intervention effects. Interpret pre-generalization checkpoints using the
unchanged behavioral timing rules in EXPERIMENT_01.md. Any denser timing windows
must be chosen from untouched behavior and labeled exploratory.

Primary contrast is targeted held-out loss minus mean matched-control loss,
reported separately for each control family and intervention. Negative is favorable;
also require lower loss than the clean baseline. Report accuracy, training effects,
geometry, displacement sizes, every checkpoint, and every strength. Do not select
only favorable modes or checkpoints. Controls are not independent model replicates.

A positive result is conditional on the existing readout, not proof that clustering
alone causes learning. A negative result may reflect an immature readout. Changes
after generalization do not establish pre-grokking sufficiency. Fresh seeds and
cluster-count sensitivity precede confirmatory claims. Training-time interventions
and rescue remain separate future experiments.

## Run

Replace BASELINE with the completed Experiment 1 baseline directory. Use fresh
output destinations. Both commands run offline on CPU:

```bash
python -m grokking.interventions --checkpoints 'BASELINE/checkpoint_*000.pt' --intervention tighten --clusters 8 --strengths 0 .25 .5 1 --repeats 5 --seed 601 --output-dir outputs/experiment02_tighten
python -m grokking.interventions --checkpoints 'BASELINE/checkpoint_*000.pt' --intervention separate --clusters 8 --strengths 0 .25 .5 1 --repeats 5 --seed 601 --output-dir outputs/experiment02_separate
```

Each destination contains results.json, protocol.json, fitted centers, contrasts.csv,
and report.md. Existing Experiment 1 commands default to center contraction.
