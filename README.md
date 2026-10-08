# Topology of Grokking

The code is split by responsibility so individual pieces can be imported and tested without
starting an experiment:

- `grokking/data.py` - deterministic dataset splits and data loaders
- `grokking/network.py` - the transformer model
- `grokking/training.py` - training, evaluation, residuals, and hidden-state extraction
- `grokking/topology.py` - topology and neural-collapse metrics
- `grokking/plotting.py` - topology tracking graphs
- `grokking/analysis.py` - loading and summarizing saved histories
- `grokking/experiment.py` - configuration-driven orchestration and CLI
- `model.py` - backward-compatible entry point

## Setup and tests

```powershell
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

Show every configurable parameter with:

```powershell
python model.py --help
```

For example, run a smaller experiment and save its history, model, and graphs under
`outputs/demo`:

```powershell
python model.py --epochs 500 --input-range 113 --batch-size 256 `
  --log-interval 10 --tda-interval 50 --stage-exponents 1,2,3 `
  --output-dir outputs/demo
```

For Bash or SLURM, use backslashes instead of PowerShell backticks for line continuation.

The plot bundle contains training dynamics, persistence-diagram feature counts, long-lived
features, total persistence, Wasserstein shifts, distance to the dataset diagram, representation
geometry, and an accuracy/topology phase plot. Pass `--no-plots` to skip graph generation.
The bundle also includes true Betti curves for the latest TDA checkpoint. Unlike the raw
feature-count plot, a Betti curve counts only intervals that are alive at each filtration scale.

Example SLURM command:

```bash
srun python model.py \
  --device cuda \
  --epochs 10000 \
  --log-interval 10 \
  --tda-interval 100 \
  --stage-exponents 1,2,3 \
  --output-dir "outputs/${SLURM_JOB_ID}"
```

Regenerate graphs later from an existing history without retraining:

```bash
python pickle_analyser.py outputs/JOB_ID/grokking_history.pkl \
  --plot --plot-dir outputs/JOB_ID/plots
```

Programmatic callers can construct an `ExperimentConfig` and pass it to
`grokking.experiment.run_experiment`. Topology callables are injectable, allowing unit tests to
replace expensive persistent-homology calculations.

## Simple Agreement

Download the authors' low-diversity Simple Agreement data once, then train offline:

```powershell
python -m grokking.agreement
python model.py --task simple_agreement --epochs 500 --batch-size 512 `
  --entropy-interval 5 --tda-interval 25 --tda-max-samples 128 `
  --output-dir outputs/simple_agreement
```

Source: [Learning Syntax Without Planting Trees](https://github.com/kabirahuja2431/transformers-hg),
revision `4744caf2fac1cc94beb8775b9b36d3fbf517cbae`, grammar
`agreement_hr_agreement_linear`. The downloader saves source URLs and SHA-256 hashes;
the loader verifies file integrity. Downloaded files live in ignored `data/simple_agreement/`.
Use `--agreement-data-dir` for a different location.

This is an adaptation: each sentence is truncated before its main verb and the target
is that verb's number (`0=singular`, `1=plural`). For example, `the newt near our xylophones`
has target singular. Six positions hold up to five words, masked padding, and a final
readout token. There is no verb or sentence suffix in the input. The two-class output
avoids asking the model to guess which interchangeable verb the generator selected.

The authors' splits are retained, with duplicate prefixes removed and validation prefixes
already in training removed. The pinned data yields 25,810 training, 473 validation,
9,469 grammatical-generalization (`g1_test`), and 9,505 linear-rule (`g2_test`) prefixes.
The latter tests agreement with the nearest noun: **it is a shortcut diagnostic, not
grammatical correctness**. All four accuracies are plotted separately. `train_fraction`,
modulus, exponent, and input range apply only to arithmetic; arithmetic stage arguments
are rejected for language. No arithmetic dataset geometry is assigned to word IDs.

The supplied low-diversity grammar has no relative clauses: selecting the first noun's
number can solve this adapted task. This experiment tests generalization against a
nearest-noun shortcut, not unrestricted hierarchical syntax, and does not reproduce
the authors' language-model objective or guarantee delayed grokking.

## TinyStories

Prepare a small subset once, then train offline:

```bash
python -m pip install -r requirements.txt
python -m grokking.tinystories --train-stories 128 --validation-stories 128
python model.py --task tinystories --context-length 64 --train-examples 512 --validation-examples 512 --batch-size 64 --epochs 500 --entropy-interval 5 --tda-interval 25 --tda-max-samples 128 --output-dir outputs/tinystories
```

Uses [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories), with
fixed UTF-8 byte contexts predicting the following byte (256 classes). This
adapts the existing single-target model; it is not the original subword
language-model setup. Contexts never cross story boundaries. Training and
validation use different stories; identical training contexts are excluded
from validation. Examples are sampled once from the prepared corpus using the
experiment seed and reused every epoch. Requested counts are capped by the
number of available unique examples.

Training and validation loss/accuracy are evaluated at the same checkpoint;
both loss curves are plotted. Entropy and topology remain available. A small
training sample encourages memorization; delayed generalization is not
guaranteed, and natural text need not reach 100% held-out accuracy. The downloader
records the upstream revision and file hashes, verified during offline loading.
`run_language.slurm` now runs this task; prepare or copy the data on the cluster first.

## Dyck-2 bracket classification

```bash
python model.py --task dyck --dyck-length 32 --train-examples 512 --validation-examples 512 --batch-size 64 --epochs 500 --entropy-interval 5 --tda-interval 25 --tda-max-samples 128 --output-dir outputs/dyck
```

This task classifies valid/invalid strings of `()` and `[]`. Both classes have
balanced opening/closing counts for each type and balanced nesting when bracket
types are ignored. Invalid examples swap two differently typed closing brackets,
so counting alone cannot solve the task. The generator is a constrained random
walk, not a uniform sample over all Dyck strings. Each split is balanced, unique,
fixed across epochs, and disjoint from the other split; the seed and generation
settings are saved in history. Length must be even and at least 4; split sizes
must be even. Small lengths may not support the requested number of unique examples.

The model reads all brackets plus a final readout token and predicts two classes.
Training and held-out loss/accuracy are measured at the same checkpoint; both
loss curves are plotted. Reduce `--train-examples` to encourage memorization.
This measures generalization to unseen strings at the same length, not longer
strings or deeper nesting explicitly. Delayed generalization is not guaranteed.
Entropy and topology tracking remain available; no download is required.

## Entropy and topology tracking

Both tasks now save complementary measurements:

- **Spectral entropy:** for each layer, center the final-position activations of a fixed
  training probe and compute covariance `C = X.T @ X / N` in float64. For eigenvalues
  `lambda`, use `p = lambda / sum(lambda)` and `H = -sum(p * ln(p))`. Save `H`,
  `H / ln(d_model)`, effective rank `exp(H)`, total variance, and the full spectrum.
  Activations are not standardized per coordinate for this measurement.
- **Persistent entropy:** separately for H0, H1, and H2, use positive finite bar
  lifetimes `l = death - birth` and compute `-sum((l / sum(l)) * ln(l / sum(l)))`.
  Also save entropy divided by `ln(number of positive bars)` and the bar count.
  Infinite and zero-length bars are excluded; a single positive bar has entropy zero.
- **Wasserstein shift:** the existing persistence-diagram distance from the preceding
  computed checkpoint, including matching disappearing/appearing features to the
  diagonal. It measures topological change, not entropy. A first checkpoint or failed
  comparison is unavailable (`NaN`); arithmetic stage changes reset the comparison.

Zero covariance and diagrams with no positive finite bars have **undefined entropy
(`NaN`)**, rather than being reported as evidence of entropy collapse. Topology failures
are saved with `topology_valid=False` and an error message. The TDA analysis uses a fixed
validation subsample and coordinate standardization, so persistent entropy and spectral
entropy describe different aspects of the representations. Persistent entropy is a
summary of bar-length diversity, not total model uncertainty or information content.

Spectral measurements run at initialization (epoch -1, optimizer step 0), every
`--entropy-interval` epochs, and after the final epoch. `--entropy-probe-size` defaults
to 512; the probe indices are saved and remain fixed even across arithmetic stages.
For interpreting normalized entropy, note that probes with `N <= d_model` impose a
sample-rank ceiling. Topology uses `--tda-interval` and `--tda-max-samples` (default 800);
pass `--tda-interval 0` to disable expensive TDA while retaining spectral measurements.

`grokking_history.pkl` stores `entropy` checkpoints, `entropy_definition`, optimizer
steps, parameter norms, dataset provenance/vocabulary, model options, and the original
topology diagrams. Persistent entropy is under each layer's `topology` metrics.
The plot bundle adds `spectral_entropy.png`, `persistent_entropy.png`, and
`normalized_persistent_entropy.png` alongside `topology_shift.png`.

These are exploratory diagnostics, not paper replications or a claimed universal
grokking threshold. Spectral terminology is informed by
[2604.13123](https://arxiv.org/abs/2604.13123) and
[2603.29262](https://arxiv.org/abs/2603.29262); persistence-lifetime entropy follows
the [persistent entropy literature](https://arxiv.org/abs/1605.02885).
