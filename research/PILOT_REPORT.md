# Pilot report

The pipeline runs, but these exploratory pilots did **not** establish delayed
generalization. No confirmatory seed sweep was started. The established p113,
40,000-step baseline remains to be completed before claiming a replication.
Existing histories also do not meet the sustained memorization/generalization rule;
see the audit and provenance caveats in [README.md](README.md).

## Observations

Three sequential development checks used the same task, architecture and seed0.
The latter two use a reduced modulus to check the pipeline at lower cost; these
are implementation checks, not independent replications or a hyperparameter study.
The first two use PyTorch initialization; the third uses explicit scaled normal
initialization after consulting the author training demo. None is an exact author
checkpoint reproduction. All are preserved in `outputs/` with their configurations.

| Run | Modulus | Updates | Train accuracy | Held-out accuracy | Stable generalization |
|---|---:|---:|---:|---:|---|
| circuit_pilot | 113 | 1,000 | 100% | 26.334% | absent |
| circuit_p31_validation | 31 | 12,000 | 100% | 9.361% | absent |
| circuit_scaled_p31 | 31 | 12,000 | 100% | see behavior_summary.json | absent |

`behavior_summary.json` is the precise source for loss, accuracy and behavior-only
timing. At p113 memorization onset is step350 using five consecutive >=99% training
evaluations, with step50 logging resolution; held-out performance remains poor.
There is no generalization timing to align with a topology event in this window.
Loss/accuracy are both measured on frozen checkpoints, unlike legacy epoch averages.

For p113, post-ReLU MLP at equals has normalized H1 total persistence7.014 at
initialization,3.609 at step100 and4.622 at step1000. Its centered RMS vector norm
changes from1.459 to22.249 to33.309. These measurements show why activation scale
must be examined separately from topology. They do not show circuit formation.
The fraction of centered full-grid MLP activation energy on sum DFT modes is
0.000407 at initialization and0.010520 at step1000; most variance is elsewhere.
These are energy fractions, not decoded arithmetic accuracy or a complete mechanism.

## Interventions and causal limits

At p113 step1000 the two exploratory sum frequencies are31 and42, selected from
activation DFT power. Joint projection removal has rank4 and Frobenius norm132.240.
Held-out cross entropy increases from21.453 to22.124, accuracy falls from26.334% to
25.137%, and9.061% of held-out predicted answers change. Three matched rank/norm
random controls give losses21.449,21.455,21.456. The candidate intervention's normalized
H1 Wasserstein distance from clean is0.705; controls give0.663,0.663,0.693.
Single-frequency interventions, random controls, head singles/pairs and exact predicted
answers are retained in the machine-readable results.

This is exploratory evidence that these sum-related activation components affect a
partially accurate model under this intervention. The topological effect is close to
controls, selection was exploratory, the model has not generalized and the control
sample is tiny. It does not establish the requested link to a generalizing circuit.
It also does not establish that topological change causes performance, identify an
entire circuit, or rule out intervention artifacts. No statistical significance is
claimed. Random controls match global rank and removed norm, not every example's
perturbation magnitude or activation distribution.

A (temporal association): measurable representation changes during memorization;
no generalization transition in the observed window. B (performance dependence):
small held-out effects under candidate removal, subject to intervention artifacts.
C (connection to topology and generalization): not established. Separate A and B
are insufficient, and a small excess over three control distances is insufficient.

## Reproduction and artifacts

The full audit, definitions and future protocol are in [README.md](README.md).
The commands used for the development pilots were:

```powershell
python -m grokking.circuits --output outputs/circuit_pilot --p 113 --steps 1000 --log-every 50 --save-every 100 --initialization pytorch
python -m grokking.circuits --output outputs/circuit_p31_validation --p 31 --steps 12000 --log-every 100 --save-every 500 --threads 1 --initialization pytorch
python -m grokking.circuits --output outputs/circuit_scaled_p31 --p 31 --steps 12000 --log-every 100 --save-every 250 --threads 1 --initialization scaled_normal
python -m grokking.circuit_analysis outputs/circuit_pilot --output outputs/circuit_pilot/analysis_fixed --steps 0,100,300,500,700,1000 --candidate-frequencies 31,42 --threads 1
python -m grokking.circuit_analysis outputs/circuit_scaled_p31 --output outputs/circuit_scaled_p31/analysis_fixed --steps 0,250,2000,6000,10000,12000 --threads 1
```

Use fresh output names when rerunning. The final analysis freezes frequencies across
time; exploratory earlier development analyses that selected them per checkpoint are
retained separately and should not be mixed with this comparison. Candidate frequencies
can also be supplied from a distinct pilot for fresh runs.

Tracked summaries and exported SVG plots are in this directory. Detailed diagrams,
Betti curves, spectra, prediction arrays and intermediate checkpoints live under each
run's ignored `outputs/` directory; preserve or archive those separately from Git.
The development runs preceded addition of checkpoint source hashes; future runs record
the training source SHA256 and schema version. See `environment.json` for versions.

## Validation and next steps

All30 unit tests pass, including synthetic Fourier signatures, actual H1 on a circle,
fixed split uniqueness, causal masking, exact clean activation restoration, and
normalization invariance. The full suite needed temporary-directory access for three
pre-existing tests. A step1000 checkpoint successfully resumed into a fresh directory
and performed update1001. That smoke check is not a cross-device reproducibility test.

The next scientific gate is a completed p113 baseline with the documented initialization
and optimizer, checking sustained delayed generalization using inexpensive logging.
Do not infer success from low training loss. If it fails, compare initialization,
parameterization and training details directly with the author implementation rather
than tuning against desired topological results. Once behavior is validated, analyze
the full checkpoint series, freeze candidate selection, and run the prespecified fresh
seeds and stronger matched controls. Until then, cross-seed consistency, topology timing
relative to grokking, topology-specific intervention evidence and clean cross-checkpoint
patching remain unanswered.
