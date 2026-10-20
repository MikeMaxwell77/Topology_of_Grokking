# Controlled topology study

Research question: does delayed generalization coincide with reproducible changes in hidden-representation topology beyond activation scale? This path is separate from historical experiments: use `grokking.study` and `grokking.study_analysis`.

## Audit of the existing path

Audit performed before editing on branch `study/topology-during-grokking`. No AGENTS.md or circuits/pilot.py was present in the repository. Historical implementation and saved outputs are preserved.

- `ModularArithmeticDataset` enumerates ordered inputs over `input_range=300`, labels `(a*b+a+b) % p`, and emits `[a,b,exponent,modulus]`. The exponent token does not change those labels. This is not modular addition. Split construction currently passes the input range consistently to both splits, so the suspected missing validation argument is not present in this revision. Inputs exceeding p repeat residue classes.
- `experiment.py` creates an exponent schedule; the actual stage-switch block is commented out. Configurations record initial choices rather than a complete checkpoint of runtime/model/optimizer state. Epochs contain multiple optimizer steps. There is no resume path.
- `TinyTransformer` uses PyTorch encoder blocks with LayerNorm, ReLU, zero dropout, learned positional embeddings and a biased output head. Its final-token post-layer state is not a no-LayerNorm baseline.
- Entropy uses a fixed training probe. Topology uses a deterministic validation subset, but coordinate-wise standardization changes geometry. Several interval counts are called Betti numbers, percentile-based counts are called long-lived, invalid activations are replaced and failed TDA metrics include zeros. H2 and input-coordinate 'ideal' comparisons complicate interpretation. The new path does neither.
- Dataset shuffling uses local generators, but DataLoader iteration can consume the global Torch RNG even without shuffled inputs. Online analysis shares the training process. The new full-batch path has no loaders and runs topology offline under an RNG-preserving context.

## Baseline and relation to the authors' implementation

The [paper, section 3](https://arxiv.org/html/2301.05217v3) specifies p=113, a 30% split, one layer, width 128, four heads of width 32, ReLU MLP width 512, no LayerNorm, untied embeddings, and full-batch AdamW at lr=.001, wd=1 for 40,000 updates. These settings are implemented. All 12,769 ordered pairs occur exactly once; 3,830 train and 8,939 held out. Inputs are `[a,b,113]`, with 113 representing equals. Output classes are residues 0..112. The task never changes.

Consulted the [authors' repository](https://github.com/neelnanda-io/Grokking) and [author-provided training demo source](https://github.com/neelnanda-io/TransformerLens/blob/main/demos/Grokking_Demo.ipynb), accessed 2026-10-20. The demo specifies betas=(.9,.98), split seed 598, disables biases, and casts logits to float64 for log-softmax. This implementation uses those betas, split seed, bias-free matrices and float64 cross-entropy with float32 model weights. AdamW epsilon is explicitly 1e-8, constant lr, no scheduler, clipping, dropout or minibatching. All parameters receive weight decay. Attention is causal with 1/sqrt(head width) scaling and attention/MLP residual additions.

This is **not an exact reproduction**. It uses a small native PyTorch implementation rather than the authors' model library. Initialization is explicitly independent N(0, .1^2) for all matrices and positional embeddings (configurable), rather than claiming equivalence to the library's version-dependent initializer. Fused QKV layout and RNG draw order differ. Model seed defaults to 42 rather than the demo's 999. The paper uses 40,000 updates; the consulted demo uses 25,000 and checkpoints every 100, whereas this study uses 40,000 and 50. Checkpoint zero here means before any update. A pilot must establish whether this implementation groks. No circuit reconstruction is attempted.

## Checkpoints, resume, and storage

Every checkpoint includes model and AdamW states, completed optimizer steps, actual config, train/held-out/probe indices, dedicated split/probe/training generator states, Python/NumPy/Torch/CUDA random states, evaluation history, source commit, dirty flag, tracked diff and training/config source snapshots. Full-batch optimization is deterministic and consumes no sampling randomness; the reserved training generator is independent of split and probe generators. Evaluation uses no gradients, no random sampling, and separate chunks. Offline analysis preserves global RNG states and records its own seed, package versions, source and exact checkpoint paths.

Resume requires the same configuration, training/config source, device and Torch thread count. Use the same PyTorch version and hardware/backend for trajectory equivalence; CUDA equivalence has not been tested. Checkpoints are loaded with pickle support for RNG tuples: load only trusted files. New runs and analyses require fresh directories. Resume refuses destinations with later checkpoints to preserve results.

`estimate` reports 226,048 parameters, 801 checkpoints and roughly 2.59 GB (2.42 GiB) including conservative history/metadata allowances. This is an estimate, not a filesystem quota guarantee; representation exports and plots are extra. A three-update baseline-size CPU benchmark on this machine measured about .206 seconds/update, suggesting roughly 2.3 hours for optimization alone. Evaluation/checkpoint I/O adds time. The full pilot has not been run.

## Measurements

A fixed probe is selected without replacement from the complete input domain with independent seed 599, stored in every checkpoint. Default count is 256. The probe can include training and held-out examples; it is a fixed population sample rather than a held-out performance estimate.

Primary representation: block 0, token index 2 (equals), residual after attention and MLP residual additions, before unembedding. Secondary: block 0, token 2, post-ReLU MLP hidden activations before projection. Extraction uses eval mode and no gradients. NPZ exports contain both clouds, inputs/pairs, labels, probe indices, layer, token position and step.

Raw Euclidean point clouds and `(X-centroid)/RMS` are analyzed separately, where `RMS=sqrt(mean_i sum_j (X_ij-centroid_j)^2)`. This removes a single global scale; it does not whiten coordinates. Collapsed clouds retain raw measurements but normalized geometry is undefined and marked failed. Nonfinite values and failed persistence are explicit errors with absent metrics, never fabricated zeros.

Primary outcome: **total finite H1 persistence of RMS-normalized post-block residuals**. Ripser computes Vietoris-Rips H0/H1 with Euclidean distance, no filtration cutoff. Exports preserve essential intervals in diagrams. Counts are named finite feature counts. Betti curves count `birth <= scale < death`, including essential H0, on an explicitly saved grid (default 201 scales from 0 to 4). The grid affects curves only; raw clouds may require a larger grid to show their complete curves.

Persim computes consecutive 1-Wasserstein distances using Euclidean ground cost in birth/death space, with diagonal matching. Essential infinite intervals are removed explicitly. Empty vs nonempty diagrams cost the sum of lifetimes/sqrt(2); two empty diagrams cost zero. Spacing in optimizer steps and distance/step are exported. A per-step average is a descriptive normalization, not evidence of linear motion across missing checkpoints. Failed measurements break the comparison chain.

Also exported: train/held-out cross-entropy and accuracy, parameter L2 norm, representation RMS, and covariance effective rank `exp(-sum_i p_i log(p_i))` with `p_i=lambda_i/sum(lambda)`, covariance `X_centered.T @ X_centered/N`. Zero covariance has undefined effective rank. It is not the participation ratio in the historical module.

Behavioral timing uses train accuracy >=.99 and held-out accuracy >=.95 for five consecutive evaluations, reporting the **beginning** of each qualifying sequence and their delay. Missing events are null/absent within budget. Evaluation spacings are recorded. An explicit provisional poor plateau rule requires a contiguous measured span of >=500 steps with train >=.99 and held-out <.5 after memorization and before generalization. These choices are configurable; sensitivity exports vary train thresholds (.98,.99), held-out thresholds (.90,.95,.99), and consecutive counts (3,5). A qualifying generalization event alone does not establish delayed generalization. Boundaries never use topology peaks.

Analysis produces aligned plots, diagrams and Betti curves with common axes within each geometry, CSV/JSON measurements, NPZ arrays and a concise report. All selected checkpoints receive diagram/curve plots, allowing selection before/during/after behavioral generalization without retraining. Large selections can produce many files. Correlation or aligned changes establish neither causation nor a circuit.

## Commands (PowerShell, repository root)

Use a fresh output directory for each new run or analysis. `python` below can be replaced by `.venv\Scripts\python.exe`.

```powershell
python -m unittest discover -s tests -v
python -m grokking.study estimate

# Tiny smoke: deliberately different size, not grokking evidence.
python -m grokking.study train --modulus 7 --d-model 16 --n-heads 4 --d-ff 32 --probe-size 16 --steps 8 --checkpoint-interval 2 --evaluation-interval 2 --output-dir outputs/topology_smoke --stop-at 4
python -m grokking.study train --modulus 7 --d-model 16 --n-heads 4 --d-ff 32 --probe-size 16 --steps 8 --checkpoint-interval 2 --evaluation-interval 2 --output-dir outputs/topology_smoke --resume outputs/topology_smoke/checkpoint_00000004.pt
python -m grokking.study_analysis --checkpoints 'outputs/topology_smoke/checkpoint_*.pt' --output-dir outputs/topology_smoke_analysis

# One baseline pilot; run after reviewing storage/runtime estimates.
python -m grokking.study train --output-dir outputs/topology_pilot --device cpu
python -m grokking.study_analysis --checkpoints 'outputs/topology_pilot/checkpoint_*.pt' --output-dir outputs/topology_pilot_analysis

# Resume the same planned budget/config after interruption (example checkpoint).
python -m grokking.study train --output-dir outputs/topology_pilot --resume outputs/topology_pilot/checkpoint_00010000.pt --device cpu

# Selected checkpoints only; substitute steps based on behavioral timing.
python -m grokking.study_analysis --checkpoints outputs/topology_pilot/checkpoint_00000000.pt outputs/topology_pilot/checkpoint_00010000.pt outputs/topology_pilot/checkpoint_00040000.pt --output-dir outputs/topology_selected

# Later comparison pilot: training fraction is an experimental difference.
python -m grokking.study train --train-fraction .8 --output-dir outputs/topology_fraction80
python -m grokking.study_analysis --checkpoints 'outputs/topology_fraction80/checkpoint_*.pt' --output-dir outputs/topology_fraction80_analysis
```

## Validation and stages remaining

All 32 tests passed with the local environment. The complete suite required execution outside the sandbox because three existing language-task tests use restricted temporary directories; those tests passed without code changes.

The smoke run completed eight optimizer updates with interruption/resume at step four and five analyzed checkpoints. All 20 representation/geometry measurements succeeded; CSV/JSON/NPZ exports and plots were generated under `outputs/topology_smoke_validated_analysis` (final implementation) and `outputs/topology_smoke_analysis` (initial validation). Neither behavioral event occurred within this tiny budget. This is workflow validation only.

Tests cover unique pairs, correct labels, exhaustive disjoint splits, default population/probe counts, fixed probes without global RNG consumption, model/extraction shapes and actual final-token MLP activation, RMS invariance and collapse/nonfinite input, circle H1 and line no-H1, essential/empty diagrams, diagonal matching, known timing/absence, explicit failed TDA, optimizer and parameter resume equivalence, evaluation history equality, and analysis preserving RNG states and resumed training. Tests retain small artifacts in ignored `outputs/study_*test_*` folders for inspection because restricted Python 3.14 temporary directories are inaccessible in this sandbox.

Stage A is validated. Stage B remains: run the full baseline pilot, inspect a clear memorization/poor-generalization plateau and later held-out improvement, including timing sensitivity. The full pilot and larger-fraction comparison were not launched during implementation; this environment has a CPU-only Torch build. No grokking or topology-transition result is claimed. Do not launch a sweep until the pilot passes behavioral review.

Before Stage C, freeze the primary outcome, probe count, architecture/initialization, split protocol, analysis scales, timing/plateau rules and sensitivity plan in a committed protocol. Start with at least five fresh model seeds (suggested 101..105) while keeping split/probe seeds fixed to isolate initialization variation. Report each run, including failures to memorize/generalize; do not drop failures or retune measurements to produce a spike. Split-seed robustness is a separate subsequent analysis. Stage D uses the same architecture/task with .8 train fraction and must be classified from observed behavior rather than assumed to generalize ordinarily. Multi-seed aggregate inference and circuit discovery remain future work.
