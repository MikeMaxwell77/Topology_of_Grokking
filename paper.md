# Do representation clusters cause grokking?

Research protocol and exploratory experiment notebook. Created 2026-10-29.
This document distinguishes proposed experiments from executed results. It is not
a report of an established causal effect.

## Question and competing explanations

Does formation of a specified hidden-representation structure enable or accelerate
delayed generalization? Alternatively, does the structure merely accompany circuit
formation, reflect a readout that has learned to use an existing circuit, or result
from removal of memorizing components? An entropy or topology spike may identify
an intervention window without being the cause of generalization.

Nanda et al. reverse-engineered Fourier computations for modular addition and
identified memorization, circuit formation, and cleanup. Their findings motivate
testing whether geometric changes are necessary for computation, but do not
establish cluster formation as the trigger of grokking [1]. Circular structure,
discrete clustering, low effective rank, and persistent-homology loops are distinct
properties. We will not infer one from another or assume every run uses the same circuit.

Dataset entropy, representation spectral entropy, and predictive entropy are also
different. For deterministic modular addition, conditional label entropy H(Y|X)
is zero. Spectral entropy measures how covariance variance is distributed; it is
not the Shannon entropy of the dataset or proof of useful compression.

## Transformer audit and selected baseline

The historical `grokking/network.py: TinyTransformer` uses PyTorch encoder layers,
LayerNorm, biased projections, ReLU, zero dropout, learned positions, and a final
token readout. Its attention is not causally masked. Its detached hidden-state
helper supports observation, not differentiable training penalties. The historical
modular data path computes `(a*b+a+b) % p`, and the existing
`run_grokking_modular.slurm` launches that path. It is not our modular-addition baseline.

Use the already implemented `grokking/study.py: StudyTransformer` instead:

| Component | Selected study baseline |
|---|---|
| Task | `(a+b) % 113`, each ordered residue pair once |
| Input/output | `[a,b,equals]`; 113 output classes |
| Split | Fixed 30% train, exhaustive disjoint held-out remainder |
| Transformer | One causal attention/MLP block, two residual additions |
| Width | 128 residual dimensions, four heads, 512 ReLU MLP units |
| Normalization/bias/dropout | No LayerNorm, no biases, no dropout |
| Embedding/readout | Learned positions; untied embedding and unembedding |
| Optimizer | Full-batch AdamW, lr .001, weight decay 1, betas (.9,.98) |
| Budget | 40,000 optimizer updates; evaluation/checkpoints every 50 |
| Initialization | Independent normal std .1; model seed 42 |

Task, dimensions, absence of LayerNorm, learning rate, weight decay, and full-batch
budget align with the published setup [1]. Betas and other implementation details
are documented in `research/TOPOLOGY_STUDY.md`. This is not an exact reproduction:
native PyTorch implementation, initialization, seed, and RNG draw order differ.
The existing study architecture needs no speculative redesign before the pilot.
Keep it fixed so an intervention is not confounded with an architecture change.

## Measurement plan

Preserve example IDs, split, checkpoint step, layer, token position, and activation
arrays. Primary intervention site: equals-token post-block residual before the
linear unembedding. This isolates a direct dependency on the final representation;
later studies must test earlier sites to locate where structure forms.

Track train/held-out cross-entropy and accuracy, centered activation RMS, uncentered
RMS, spectral entropy/effective rank and total variance. Quantify clustering using
between-cluster scatter divided by total scatter, with fixed original assignments.
Inspect occupancy and robustness to cluster count. K-means always partitions a
cloud: its output alone is not evidence of naturally discrete clusters.

The existing offline topology analysis supplies raw and globally RMS-normalized
H0/H1 persistence. Coordinate whitening changes the target geometry and is not
interchangeable with global scale normalization. Supplementary future measurements:
linear probes, CKA across checkpoints, Fourier concentration, and task-specific
restricted/excluded losses. Do not describe these future measurements as implemented
in the intervention runner. Do not infer a circuit from a two-dimensional plot.

## Four causal interventions

### 1. Necessity: disrupt after generalization

At fixed checkpoints, reduce cluster-center separation and immediately evaluate the
remaining network. A selective held-out loss increase compared with matched controls
supports dependence on the targeted directions. Compare training loss too: damage
to both splits can indicate general computation damage. This alone does not establish
that cluster formation caused the training transition.

### 2. Sufficiency: strengthen before generalization

Before grokking, tighten within-group residuals via
`z' = mu_g + alpha*(z-mu_g)` with `0 < alpha < 1`, or increase separation of
training-fitted centers. Groups and transformations must be fitted without held-out
answers. A pre-existing immature representation may lack a usable readout; a negative
result does not refute every cluster hypothesis. A positive result is conditional
on the already-trained downstream network and must exceed matched-control effects.
This intervention is proposed, not implemented in the first sweep.

### 3. Timing: interfere during learning

Branch a saved pre-transition checkpoint into untouched, targeted, and matched-random
training arms, preserving model/optimizer/RNG state and batch order. Apply the operation
for a fixed duration, then remove it. Specify whether centers stay fixed or are refitted
on a fixed schedule; for the initial branch design freeze centers and assignments to
training examples at branch start. For a differentiable intervention, specify its
backward rule explicitly: fixed center translations have identity derivative.

Evaluate with interventions both active and switched off. A deficit only while active
is execution disruption; delayed clean generalization is evidence of altered learning.
Monitor alternate representations and re-formation. Compare onset and duration windows
and include runs that never generalize (censored at the budget). This training-branch
experiment is proposed, not implemented in the evaluation-only runner.

### 4. Rescue: restore after disruption

For immediate evaluation, save the original perturbation and subtract that same vector
to restore the activation, checking recovery of logits and loss. This is an important
implementation control but is algebraically guaranteed, not independent proof that
clusters are the mechanism. A stronger future rescue selectively restores the candidate
component under an otherwise maintained disruption and compares random-component rescue.
After training interventions, removal is not an inverse of intervening optimization:
test recovery of clean performance and geometry over additional matched training.

## How to calculate the first intervention

Fit k-means on all training activations at each checkpoint, without labels. Primary
exploratory k=8, seed 601, at most 100 Lloyd updates; raw Euclidean geometry. This k is
a convenient exploratory choice, not a theory-derived count. Before confirmatory
claims, check k=4,16,32 and multiple clustering initializations without selecting only
the most damaging result. Assign held-out inputs to their nearest training-fitted center.

Let `mu_k` be a center and `mu` the training activation mean. Define

    delta_i = -beta * (mu_g(i) - mu)
    z'_i = z_i + delta_i

with beta in {0, .25, .5, 1}. This translates each assigned group without changing its
internal pairwise offsets. At beta=1, converged training cluster means coincide; held-out
group means need not coincide exactly. Shapes and covariance may still identify groups.
Measure actual geometry change rather than calling the operation complete destruction.

Two controls use identical per-example displacement magnitudes:

1. Independent random unit directions: `delta_control_i = ||delta_i|| * u_i`.
2. Shared random orthogonal rotation: `delta_control_i = delta_i @ Q`, preserving
   the Gram matrix of all displacement vectors as well as their lengths.

Use five control draws initially. Reuse draws across strengths for paired comparisons.
Zero strength is the untouched evaluation baseline. Controls may also alter clusters;
record their geometric effects instead of assuming they preserve structure.

Primary exploratory contrast at each checkpoint/strength is targeted held-out loss
increase minus mean control loss increase, reported separately for each control family.
Control draws quantify random-direction variability, not independent training replicates.
Report all strengths/checkpoints, training effects, and control spread. Confirmatory
uncertainty needs independent model seeds; do not treat examples as model replicates.

## Timing and selection rules

Two controls have different roles: the untouched trajectory locates learning phases;
matched perturbations test whether damage is specific to targeted directions.

Use the existing behavioral definitions: train accuracy >=.99 and held-out accuracy
>=.95 for five consecutive evaluations, with event time at the first qualifying point.
A provisional poor-generalization plateau requires at least 500 measured steps with
train >=.99 and held-out <.5 before generalization. Report missing events explicitly.
Generalization without this delay is not a demonstrated grokking baseline.

Initially sweep a coarse checkpoint grid spanning the entire untouched run. Densely
sample saved checkpoints before, during, and after the behaviorally identified transition.
Also inspect times with apparent clusters but poor generalization, if any. Never choose
the reported checkpoint because it maximizes intervention damage. Pilot timing choices
are exploratory; freeze a selection/trigger rule before fresh-seed confirmation.

## Threats to causal interpretation

Matched norm does not guarantee matched functional severity: targeted shifts can align
more closely with the readout, alter activation scale, or leave the distribution of natural
states. Interventions can induce unnatural internal computations [2]. Record logit/loss
effects, geometry, and scale together; later add readout-sensitivity-matched controls.

An invertible change of coordinates `z'=Az` compensated by `W'=WA^-1` preserves logits
while changing Euclidean geometry. Thus raw clustering/entropy cannot be a universal,
coordinate-independent cause. The operational hypothesis concerns structure in the
fixed model coordinates as used by its downstream computation.

Distinguish the representation state after a spike from the transient event itself.
Testing whether traversing a spike matters requires temporal interventions, not just
ablating the endpoint. Neither an entropy decline nor an accuracy threshold proves a
discontinuous internal phase transition.

## First experiment and reproducible commands

Stage A: validate intervention mathematics and run on tiny existing study checkpoints.
This checks the workflow only. Stage B: run one untouched p=113 baseline and establish
whether it actually groks. Stage C: run the checkpoint-by-strength necessity sweep.
Do not launch multi-seed training intervention sweeps before reviewing these results.

```powershell
.venv/Scripts/python.exe -m unittest discover -s tests -p test_interventions.py -v
.venv/Scripts/python.exe -m grokking.interventions --checkpoints 'outputs/topology_smoke_validated/checkpoint_*.pt' --clusters 2 --repeats 3 --output-dir outputs/cluster_intervention_smoke
.venv/Scripts/python.exe -m grokking.study train --device cpu --output-dir outputs/cluster_causal_baseline
# After baseline finishes, coarse exploratory grid (every 1000 updates):
.venv/Scripts/python.exe -m grokking.interventions --checkpoints 'outputs/cluster_causal_baseline/checkpoint_*000.pt' --output-dir outputs/cluster_causal_sweep
```

The runner never changes checkpoint weights; it extracts residuals and evaluates modified
activations through the original unembedding. It saves per-condition/split results,
training-fitted centers, configuration, source, and checkpoint paths in a fresh directory.
Load only trusted local checkpoints because they include pickled optimizer/RNG objects.

## Execution record

On 2026-10-29, both new intervention unit tests passed. The evaluation sweep completed
on five tiny checkpoints (steps 0,2,4,6,8), producing 220 condition/split rows under
`outputs/cluster_intervention_smoke`. Maximum zero-intervention loss discrepancy from
the saved training history was 6.81e-10. These are workflow results, not causal evidence.

The existing seven-test study suite had five passes, one error, and one failure.
Windows Application Control blocked SciPy's `_qmvnt_cy` DLL during topology import;
the analysis-isolation test also reported invalid topology rows. Topology validation
therefore remains unresolved on this environment. Neither baseline training nor the
new intervention runner imports that dependency.

The full untouched baseline was launched using `run_causal_baseline.py`, which queues
the coarse sweep only after successful training. The initial PowerShell script launcher
was blocked by the machine's script policy and was replaced with the Python launcher;
no execution policy was changed. Wrapper PID at launch: 45868. Training was verified
through step 50: train accuracy .1467, held-out accuracy .00604. This early result is
not evidence of grokking. CPU-only Torch version: 2.14.0+cpu. The existing benchmark
suggests hours of runtime and roughly 2.6 GB of checkpoints for the full budget.

Live stdout: `outputs/cluster_causal_run.log`; stderr: `outputs/cluster_causal_run.err`;
PID record: `outputs/cluster_causal_run.pid`. Training checkpoints/history:
`outputs/cluster_causal_baseline`. Queued sweep destination: `outputs/cluster_causal_sweep`.
For a fresh combined run, execute `.venv/Scripts/python.exe -u run_causal_baseline.py`;
do not start a second copy while this run is active. The laptop must remain awake.
If interrupted, use the study runner's documented resume command with matching settings
and the latest complete checkpoint, then invoke the sweep separately in a fresh directory.
A running baseline is not a completed experiment or evidence of grokking.

## References

1. Nanda et al., *Progress measures for grokking via mechanistic interpretability*.
   https://arxiv.org/abs/2301.05217 (setup and circuit findings checked 2026-10-29).
2. Geiger et al., *Causal Abstraction for Mechanistic Interpretability*.
   https://jmlr.org/papers/volume26/23-0058/23-0058.pdf
3. Existing repository methods and initialization/reproducibility details:
   [research/TOPOLOGY_STUDY.md](research/TOPOLOGY_STUDY.md).
