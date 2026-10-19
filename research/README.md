# Arithmetic circuit development study

This extension lives in `grokking/circuits.py` and `grokking/circuit_analysis.py`.
Existing training code and results are preserved. The new experiment is exploratory;
a completed paper replication or an identified circuit must not be inferred from it.

## Audit of the existing project

Repository inspected at base commit `5045e40`, branch `experiment/dyck-language`.
No AGENTS.md was found in the repository or its checked ancestor directories.
Read README, configuration, data, model, training, experiment, topology and saved histories.
`legacy_audit.json` contains the per-run configurations, threshold checks and split audit.
Run `python research/audit_legacy.py` to regenerate it from trusted local pickle files.
The untracked `outputs.zip` is preserved and was not extracted over existing results.

| History | Train max | Held-out max | Five consecutive train observations >=99% | Five consecutive held-out observations >=99% |
|---|---:|---:|---|---|
| 21870574 | .9993 | 1.0000 | absent | absent |
| 21872428-3layers | .9231 | .9857 | absent | absent |
| 21872584-2attentionHeads | .7537 | .8251 | absent | absent |
| 21876780-mod223 | .9973 | .9956 | absent | absent |
| 21880203 | .9991 | .9999 | absent | absent |
| 21884225_mod59 | 1.0000 | 1.0000 | absent | epoch 3430 |
| agreement_smoke | .9812 | 1.0000 | insufficient observations | insufficient observations |
| dyck_smoke | .5332 | .5000 | absent | absent |
| tinystories_smoke | .3789 | .3125 | insufficient observations | insufficient observations |

No inspected run establishes delayed stable generalization after stable memorization
under this rule. This is an operational conclusion, not a claim that no transient
learning occurred. Legacy arithmetic train metrics aggregate changing weights within
an epoch; held-out accuracy is measured afterward. Arithmetic held-out loss is absent.
Rows copied forward between TDA evaluations must not be counted as independent TDA samples.
Language smoke runs are not arithmetic experiments.

Current data code computes `(a*b+a+b) % p`, despite the exponent field and older README
examples. Default stages change the exponent token and concatenate new datasets after
held-out accuracy >.95. Current labels do not depend on the exponent. Older git revisions
implemented powers, so histories without code hashes cannot establish which function
produced their labels. Saved names are not configurations: all six archived arithmetic
histories report p=113, input_range=300 and the same split settings, including names
containing mod223/mod59. Do not relabel these runs based on their directory names.

Within a stage all 90,000 ordered integer pairs are unique, and the split contains no
identical four-token inputs in both sets. Reducing operands mod113 yields 11,475 ordered
residue pairs in both splits; 87.927% of validation integer pairs have their ordered
residue pair represented in training. Swapping operands introduces further arithmetic
equivalences, distinct from exact token duplicates. Additional exponent stages use distinct
exponent tokens but repeat residue computations in current source. These overlaps are a
confound for generalization to new modular problems, even though embeddings for a and a+p
are different parameters and the inputs are not literally identical.

Topology reads the final token (position 3, modulus token for arithmetic), after each
post-LayerNorm encoder block, before the output head. It does not separately observe
heads or MLP activations. Current `_sample_indices` uses a fresh fixed RNG seed0 and
validation is unshuffled: subset and order are fixed while the dataset stays fixed.
Stage concatenation changes dataset length and hence sampled examples; historical
code versions and absent saved probe IDs limit retrospective guarantees. The newer
spectral-entropy probe is a saved fixed initial training subset and remains so at stages.

Legacy `betti_0/1/2` are diagram interval counts over the filtration, not Betti numbers
at a specified scale. Plotting now includes separate true Betti curves. Coordinatewise
standardization is recomputed per checkpoint for TDA. It removes global scaling but
also changes anisotropic geometry; spectral entropy uses raw centered representations.
Changing activation magnitudes, low-variance coordinates and resampling can influence
apparent topology changes. Raw distances, a scalar normalization and fixed probes are
needed to distinguish these effects. H0 essential intervals require explicit treatment.

Each saved run has a final `grokking_model.pt` state dictionary and a history, not a
sequence of model/optimizer/RNG checkpoints. Historical diagrams, behavior and spectra
(if present) can be reanalyzed; earlier activations and interventions cannot be recovered
from final weights. New training is needed for temporal circuit analysis. Model provenance
must be checked before even final-state analysis of legacy weights.

## Baseline sources and differences

Consulted [Nanda et al., paper sections 3 and appendix A](https://arxiv.org/html/2301.05217v2),
the [author's repository](https://github.com/neelnanda-io/Grokking), including `grokking.py`,
and the [author's executable training demo](https://github.com/neelnanda-io/TransformerLens/blob/main/demos/Grokking_Demo.ipynb).
The original repository README points to a Colab whose contents were inaccessible;
the accessible author-provided training demo supplies the optimizer details.

The paper uses p113, 30% ordered pairs, a one-layer ReLU transformer, width128,
4 heads of width32, MLP512, no LayerNorm, untied embedding/unembedding, full-batch
AdamW lr.001, weight decay1 and 40,000 updates. The demo uses betas(.9,.98),
25,000 epochs, checkpoints every100, split seed598 and model seed999, and freezes biases.
Our default matches these architecture and optimizer choices but uses a small direct
PyTorch implementation, omits zero frozen biases, one seed for both split and model,
and normal(.8/sqrt(width)) initialization for every learned tensor. It has causal
attention and learned positional embeddings. This initialization is explicit, not an
assertion of identical TransformerLens parameter initialization or numerical trajectories.
Cross entropy is evaluated in float64 after float32 forward computation. The first
p113 smoke run used PyTorch defaults instead; it is retained with that configuration.
The smaller p31 validation changes the modulus only and is not the p113 replication.

## Commands

From the repository root with the existing environment and requirements installed:

```powershell
python -m unittest discover -s tests -v
python research/audit_legacy.py
python -m grokking.circuits --output outputs/new_addition_pilot --p 113 --seed 0 --steps 40000 --log-every 50 --save-every 100 --device cuda
python -m grokking.circuit_analysis outputs/new_addition_pilot --stride 1
```

Use `--device cpu --threads 1` on a CPU. Output directories must be new, to prevent
overwrites. Training never imports ripser or computes topology. One full-batch optimizer
update equals one step; initialization is step0. Checkpoints contain weights, optimizer,
configuration, ordered split/probe IDs, Python/NumPy/Torch/CUDA RNG and Torch version.
The complete lexicographic grid is deterministic. Resume into a fresh directory:

```powershell
python -m grokking.circuits --resume outputs/new_addition_pilot/step_010000.pt --output outputs/addition_continued --steps 40000 --device cuda
```

Only load trusted local pickle/checkpoint files. Resumed training uses saved config,
split and probes; steps is the cumulative target, not an additional count. Log/checkpoint
intervals remain command options. Cross-device bitwise reproducibility is not guaranteed.

## Definitions and intervention interpretation

Behavior is evaluated at one frozen state on both complete splits. Memorization and
generalization onset require >=99% train or held-out accuracy for five consecutive
logged evaluations; onset is the first of those evaluations. Delayed generalization
additionally requires a >=500-update gap and five consecutive held-out observations
below50% after memorization. Timing uncertainty is at least the logging interval;
absence of a crossing is censored, not assigned to the end of training.

Activation sites are named embedding (token+position), per-head attention values before
output projection, attention output, resid_mid, post-ReLU MLP, projected mlp_out and
resid_post. Position labels a/b/equals mean indices0/1/2. Every full-grid example is in
lexicographic a-then-b order. The saved probe is a fixed ordered uniform subset without
replacement, not independently resampled at checkpoints. Fourier diagnostics use the
complete grid, not that subset or a PCA projection.

Primary exploratory topology site is `mlp.equals`. Additional default sites include
resid_post.equals, attention.equals, embedding.a/b and mlp_out.equals. Raw and centered
RMS-normalized Euclidean Vietoris-Rips diagrams are measured in H0/H1. Normalization
divides the entire point cloud by sqrt(mean squared centered vector norm), preserving
relative distances and anisotropy. Scale and sample count are saved. Zero scale maps to
zero vectors; it is not evidence of a rich topology. Fixed nested half-probes diagnose
sample-size sensitivity; they are not independent seeds or uncertainty estimates.

Finite interval count, summed lifetime and maximum lifetime are distinct statistics.
Betti curves count birth<=scale<death on fixed scales0..3 (raw units or normalized units).
The grid need not span all raw deaths. Infinite intervals are retained for curves and
counted separately, excluded from persistence sums and persim Wasserstein-1 distances
with matching to the diagonal. JSON diagrams use Python's Infinity encoding for essential
intervals; load with Python json. Initial Wasserstein change is unavailable. Cross-checkpoint
comparisons require identical site, probe size/order, normalization and homology dimension.

The centered full-grid 2D DFT separates energy on b-frequency0 (a only), a-frequency0
(b only), equal nonzero frequencies (sum), opposite frequencies (difference), and all
remaining modes. Embedding, unembedding and composed neuron-to-logit maps have class-axis
frequency power. Periodicity is a candidate signature, not a forced model description.
These energy measures are descriptive; they do not demonstrate the full Fourier algorithm.

Exploratory interventions remove the projection of final-position MLP activations onto
sine/cosine functions of the sum for the two selected sum frequencies, individually and
jointly. This is a full-grid, offline, label-defined diagnostic using the known task's
sum coordinate, not a deployed online intervention or a learned feature decoder. All
weights, inputs and other positions stay fixed; downstream residual addition and output
mapping recompute. Random sample-space subspaces have the same rank, then their removed
activation norm is scaled to exactly match the candidate's Frobenius norm. Scaling means
controls are not pure orthogonal projections. Per-example norms and distribution shift
can still differ. Candidate frequencies are frozen across checkpoints, selected from the last analyzed
checkpoint unless explicitly supplied with `--candidate-frequencies`. This retrospective
selection is exploratory. Three deterministic random controls per group are exploratory, not
a calibrated significance test. Candidate selection sees activation grid structure, not
held-out performance; confirmatory frequencies must be frozen on a different pilot.

Single heads and every pair can also be zeroed, exposing some redundancy. Those head
ablations have no magnitude-matched head controls and remain secondary diagnostics.
Predicted answers for all interventions, clean answers, target and held-out IDs are saved
in npz. Clean and intervened held-out loss/accuracy, changed predictions, raw/RMS topology
and distance to clean are aligned at each checkpoint. Removing features can impair
performance because of intervention artifacts; dependence is not uniqueness or completeness.
Clean activation restoration and cross-checkpoint patching are not yet substantive tests
and are not presented as evidence.

## Confirmatory protocol (specified before fresh runs)

After validating a pilot with delayed generalization, freeze its candidate frequencies
and select fresh seeds1,2,3,4,5, preserving p113 and the architecture. Do not start these
runs until the pilot and resource estimate are reviewed. Primary layer/site: sole block,
post-ReLU MLP, equals token. Primary statistic: normalized H1 total finite persistence
with a fixed128-example probe. Primary timing: step of maximum absolute successive
change in this statistic minus behavioral generalization onset, considering only actual
saved checkpoints and reporting checkpoint resolution and censoring. Check the corresponding
before/during/after checkpoints and the memorization plateau. Raw scaling, half-probe,
other sites/statistics are sensitivity analyses. This is a plan; no fresh-seed test is
claimed to have run.

Primary intervention test: joint frozen candidate frequencies, with >=20 matched random
controls on each fresh run. Require held-out loss increase beyond the95th percentile of
controls AND a normalized H1 change beyond the95th percentile at the same state. Report
accuracy, predictions, removed magnitude, null direction and controls even when criteria
fail. Report per-seed paired effects and seed-level uncertainty, not checkpoint counts as
independent replications. A conservative cross-seed criterion is the joint directional
effect in at least4/5 seeds; this is not automatically a formal p-value. Comparison with
other input relationships (a, b, difference and remaining DFT modes), interventions at
other sites and raw topology can falsify an overly specific Fourier interpretation.

A = temporal association. B = loss/accuracy dependence under intervention. C = the SAME
intervention changes the specified topological observation and generalization beyond
matched controls. Separate A and B do not establish C. Even C would not show that topology
causes computation, or that a unique complete arithmetic circuit has been identified.

Training uses an equivalent final-query/final-MLP path for this single-layer model.
Tests compare its logits and all parameter gradients with the fully hooked analysis
forward. This reduces work without adding a different architecture.
