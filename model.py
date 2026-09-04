"""Backward-compatible entry point for the modular grokking experiment.

New code should import focused modules from :mod:`grokking` directly.
"""

from grokking.data import ModularArithmeticDataset
from grokking.experiment import main, run_experiment
from grokking.network import TinyTransformer
from grokking.plotting import (
    plot_history_dashboard,
    plot_results,
    save_tracking_plots,
    visualize_dataset_topology,
)
from grokking.topology import (
    analyze_topology_all_layers,
    compute_dataset_topology,
    compute_neural_collapse_metrics,
    compute_simplex_score,
    compute_topology,
    compute_wasserstein_distance_to_ideal,
    intrinsic_dimension,
    remove_infinite,
)
from grokking.training import (
    evaluate,
    extract_all_hidden_states,
    extract_all_hidden_states_with_labels,
    print_residuals,
    train_epoch,
)

__all__ = [
    "ModularArithmeticDataset",
    "TinyTransformer",
    "analyze_topology_all_layers",
    "compute_dataset_topology",
    "compute_neural_collapse_metrics",
    "compute_simplex_score",
    "compute_topology",
    "compute_wasserstein_distance_to_ideal",
    "evaluate",
    "extract_all_hidden_states",
    "extract_all_hidden_states_with_labels",
    "intrinsic_dimension",
    "plot_history_dashboard",
    "plot_results",
    "print_residuals",
    "remove_infinite",
    "run_experiment",
    "save_tracking_plots",
    "train_epoch",
    "visualize_dataset_topology",
]


if __name__ == "__main__":
    main()
