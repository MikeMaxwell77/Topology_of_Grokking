# Experiment 1: does cluster separation support generalization?

Status: exploratory single-seed pilot, specified 2026-10-29 while the untouched
baseline is still training. The initial protocol is in ../paper.md. This is an
evaluation-only necessity test; sufficiency, training-time interventions, and
rescue remain subsequent experiments.

## Fixed setup

- Baseline: `outputs/cluster_causal_baseline`, existing StudyTransformer, p=113,
  one layer, width 128, four heads, ReLU width 512, no LayerNorm or biases.
- Fixed split: 30% train, split seed 598; initialization seed 42; 40,000 full-batch
  AdamW updates, lr .001, decay 1. Checkpoints every 50 updates.
- Site: equals-token final residual, immediately before the unchanged unembedding.
- Fit: k=8 Euclidean k-means on all training activations, seed 601; no labels.
  Assign held-out activations using training-fitted centers.
- Intervention: subtract beta times the displacement from each center to the
  training global mean. beta = 0, .25, .5, 1.
- Controls: independent random directions and a shared random orthogonal rotation
  of the targeted displacement vectors; five draws each, matched per-example norms.
- Primary outcome: held-out cross-entropy increase for targeted intervention minus
  the corresponding mean control increase. Report the two control families separately.
- Secondary: train loss, accuracy, between-scatter fraction, activation RMS and
  displacement RMS. Inspect geometry change before interpreting performance effects.

## Checkpoint selection

The queued primary coarse sweep evaluates steps 0,1000,...,40000, irrespective of
performance or intervention effects. This gives 41 checkpoints and 2,788 raw
condition/split rows. The untouched run continues without intervention.

An early operational check uses steps 0 and 500, chosen to test the complete pipeline
at initialization and early memorization, before inspecting intervention effects.
This check cannot establish dependence after grokking. Its outputs are separate
from the queued coarse sweep.

After training, identify memorization (train >=99%) and generalization (held-out >=95%)
using five consecutive evaluations; use the start of each qualifying sequence.
Require a poor-generalization plateau of at least 500 measured steps with train >=99%
and held-out <50% to classify delayed generalization under this protocol. Report absent
events within the budget. Add denser windows around behavioral timing only, explicitly
labeling them exploratory. Do not choose windows from maximum intervention damage.

## Interpretation and next decision

An excess targeted loss above zero suggests sensitivity to the targeted directions,
conditional on the fixed representation/readout, not proof that clustering caused
learning. Check whether separation was actually reduced, whether training predictions
were also damaged, and whether norm matching holds. Control SD describes direction
variability only. Do not pool checkpoints as independent experimental replicates.

If this baseline does not grok, report that outcome; the sweep is still a valid
perturbation measurement but does not answer the post-grokking necessity question.
If it does grok and targeted disruption is selectively damaging, repeat with fresh
model seeds and cluster-count sensitivity before training-time causal claims.

## Running and inspecting

The existing Python wrapper is already running and queues the coarse sweep. Do not
start another baseline. Logs: `outputs/cluster_causal_run.log` and `.err`.

```powershell
# Early check, in a fresh destination:
.venv/Scripts/python.exe -m grokking.interventions --checkpoints outputs/cluster_causal_baseline/checkpoint_00000000.pt outputs/cluster_causal_baseline/checkpoint_00000500.pt --output-dir outputs/experiment01_early
# Regenerate a report from saved raw measurements:
.venv/Scripts/python.exe -m grokking.intervention_report outputs/experiment01_early
```

Each sweep saves `results.json`, `protocol.json`, fitted centers, `contrasts.csv`,
and `report.md`. Reports include all measured effects, not only favorable ones.

## Early operational result (2026-10-29)

All three intervention/report tests passed. The early check completed at steps 0
and 500 with 136 raw rows. At step 500, clean held-out accuracy was 13.79%; full
center contraction gave 11.41%. However, targeted held-out cross-entropy was lower
than both mean matched controls (excess loss approximately -1.23). This check
therefore does not support the predicted selective loss damage at that early
checkpoint. Accuracy and cross-entropy need not move together, especially when
predictions are confidently wrong. Do not change the primary metric after seeing this.
No post-grokking conclusion is available. Baseline training continues; the latest
observed step 1000 had 100% train and 19.25% held-out accuracy.
