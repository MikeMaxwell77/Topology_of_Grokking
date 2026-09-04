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
