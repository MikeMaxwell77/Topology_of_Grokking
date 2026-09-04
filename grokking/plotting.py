"""Plotting functions, isolated so core modules do not require Matplotlib."""

from pathlib import Path


def visualize_dataset_topology(dataset_topology, save_path="dataset_topology.png") -> None:
    import matplotlib.pyplot as plt
    from persim import plot_diagrams

    diagrams = dataset_topology["diagrams"]
    figure = plt.figure(figsize=(12, 4))
    diagram_axis = figure.add_subplot(1, 2, 1)
    plot_diagrams(diagrams, ax=diagram_axis, show=False)
    diagram_axis.set_title("Dataset Topology: Persistence Diagrams")
    point_axis = figure.add_subplot(1, 2, 2, projection="3d")
    data = dataset_topology["data_points"]
    sample = data[: min(1_000, len(data))]
    point_axis.scatter(sample[:, 0], sample[:, 1], sample[:, 2], alpha=0.3, s=1, c=sample[:, 2])
    point_axis.set(xlabel="Input a", ylabel="Input b", zlabel="Output", title="Dataset Embedding")
    figure.tight_layout()
    figure.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def plot_results(history, save_path="grokking_topology_analysis.png", *, show=False) -> None:
    import matplotlib.pyplot as plt

    epochs = history["epoch"]
    figure, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes[0, 0].plot(epochs, history["train_acc"], label="Train Acc")
    axes[0, 0].plot(epochs, history["val_acc"], label="Val Acc")
    axes[0, 0].set(title="Grokking: Train vs Val Accuracy", xlabel="Epoch", ylabel="Accuracy")
    axes[0, 0].legend()

    if history["topology"]:
        for layer_idx in history["topology"][0]:
            betti = [checkpoint[layer_idx]["betti_1"] for checkpoint in history["topology"]]
            persistence = [checkpoint[layer_idx]["total_persistence_1"] for checkpoint in history["topology"]]
            axes[0, 1].plot(epochs, betti, label=f"Layer {layer_idx}")
            axes[1, 0].plot(epochs, persistence, label=f"Layer {layer_idx}")
            axes[1, 1].scatter(betti, history["val_acc"], c=epochs, cmap="viridis", alpha=0.6)
        axes[0, 1].legend()
        axes[1, 0].legend()
    axes[0, 1].set(title="Topological Complexity: Betti-1", xlabel="Epoch", ylabel="Betti-1")
    axes[1, 0].set(title="Total Persistence", xlabel="Epoch", ylabel="H1 persistence")
    axes[1, 1].set(title="Topology vs Performance", xlabel="Betti-1", ylabel="Validation Accuracy")
    for axis in axes.flat:
        axis.grid(True, alpha=0.3)
    figure.tight_layout()
    figure.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(figure)


def plot_history_dashboard(
    history, save_path="ExtendedTopologyAnalysis.png", *, show=False
) -> None:
    """Plot training and all available topology dimensions in one dashboard."""
    import matplotlib.pyplot as plt

    if not history.get("topology"):
        raise ValueError("history contains no topology data")
    epochs = history["epoch"]
    first_checkpoint = history["topology"][0]
    layers = sorted(first_checkpoint)
    sample = first_checkpoint[layers[0]]
    dimensions = sorted(int(key.removeprefix("betti_")) for key in sample if key.startswith("betti_"))
    columns = max(3, len(dimensions))
    figure, axes = plt.subplots(4, columns, figsize=(6 * columns, 18), squeeze=False)
    axes[0, 0].plot(epochs, history["train_acc"], label="Train")
    axes[0, 0].plot(epochs, history["val_acc"], label="Validation")
    axes[0, 0].set_title("Accuracy")
    axes[0, 0].legend()
    axes[0, 1].plot(epochs, history["train_loss"])
    axes[0, 1].set_title("Training Loss")
    for layer in layers:
        axes[0, 2].plot(
            epochs,
            [checkpoint[layer]["intrinsic_dim"] for checkpoint in history["topology"]],
            label=f"Layer {layer}",
        )
    axes[0, 2].set_title("Intrinsic Dimension")
    axes[0, 2].legend()

    metric_rows = ((1, "betti", "Betti"), (2, "total_persistence", "Total Persistence"),
                   (3, "wasserstein_shift", "Wasserstein Shift"))
    for column, dimension in enumerate(dimensions):
        for row, key, title in metric_rows:
            for layer in layers:
                values = [checkpoint[layer][f"{key}_{dimension}"] for checkpoint in history["topology"]]
                axes[row, column].plot(epochs, values, label=f"Layer {layer}")
            axes[row, column].set_title(f"{title} H{dimension}")
    for axis in axes.flat:
        axis.grid(True, alpha=0.3)
    figure.tight_layout()
    figure.savefig(save_path, dpi=300)
    if show:
        plt.show()
    plt.close(figure)


def topology_sample_indices(history) -> list[int]:
    """Return checkpoints where persistent homology was actually recomputed."""
    flags = history.get("tda_computed")
    if flags is None:
        return list(range(len(history.get("topology", []))))
    return [index for index, computed in enumerate(flags) if computed]


def betti_curve(diagram, thresholds) -> list[int]:
    """Count persistence intervals alive at each filtration threshold."""
    return [
        sum(float(birth) <= threshold < float(death) for birth, death in diagram)
        for threshold in thresholds
    ]


def _topology_layout(history):
    indices = topology_sample_indices(history)
    if not indices:
        raise ValueError("history contains no computed topology checkpoints")
    checkpoint = history["topology"][indices[0]]
    layers = sorted(checkpoint)
    dimensions = sorted(
        int(key.removeprefix("betti_"))
        for key in checkpoint[layers[0]]
        if key.startswith("betti_")
    )
    return indices, layers, dimensions


def _draw_stage_changes(axes, history) -> None:
    exponents = history.get("exponent")
    if not exponents:
        return
    changes = [
        (history["epoch"][index], exponents[index])
        for index in range(1, len(exponents))
        if exponents[index] != exponents[index - 1]
    ]
    for axis in axes:
        for epoch, exponent in changes:
            axis.axvline(epoch, color="black", linestyle="--", alpha=0.35)
            axis.annotate(f"c={exponent}", (epoch, 1), xycoords=("data", "axes fraction"),
                          xytext=(4, -4), textcoords="offset points", va="top", fontsize=8)


def _save_metric_grid(history, metric: str, title: str, ylabel: str, path: Path) -> None:
    import matplotlib.pyplot as plt

    indices, layers, dimensions = _topology_layout(history)
    epochs = [history["epoch"][index] for index in indices]
    figure, axes = plt.subplots(1, len(dimensions), figsize=(6 * len(dimensions), 4), squeeze=False)
    for column, dimension in enumerate(dimensions):
        axis = axes[0, column]
        key = f"{metric}_{dimension}"
        for layer in layers:
            values = [history["topology"][index][layer].get(key, 0.0) for index in indices]
            axis.plot(epochs, values, marker="o", markersize=3, label=f"Layer {layer}")
        axis.set(title=f"{title} H{dimension}", xlabel="Epoch", ylabel=ylabel)
        axis.grid(True, alpha=0.3)
        axis.legend()
    _draw_stage_changes(axes.flat, history)
    figure.tight_layout()
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def save_tracking_plots(history, output_dir: str | Path = "plots") -> list[Path]:
    """Save a bundle of graphs for training and representation topology."""
    import matplotlib

    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    training_path = directory / "training_dynamics.png"
    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history["epoch"], history["train_acc"], label="Train")
    axes[0].plot(history["epoch"], history["val_acc"], label="Validation")
    axes[0].set(title="Accuracy", xlabel="Epoch", ylabel="Accuracy")
    axes[0].legend()
    axes[1].plot(history["epoch"], history["train_loss"])
    axes[1].set(title="Training Loss", xlabel="Epoch", ylabel="Cross-entropy")
    for axis in axes:
        axis.grid(True, alpha=0.3)
    _draw_stage_changes(axes, history)
    figure.tight_layout()
    figure.savefig(training_path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    written.append(training_path)

    metric_graphs = (
        ("betti", "Persistence Diagram Feature Count", "Number of bars", "topology_feature_counts.png"),
        ("long_lived", "Long-lived Feature Count", "Features above 75th percentile", "topology_long_lived.png"),
        ("total_persistence", "Total Persistence", "Summed lifetime", "topology_persistence.png"),
        ("wasserstein_shift", "Change From Previous Checkpoint", "Wasserstein distance", "topology_shift.png"),
        ("wasserstein_to_ideal", "Distance to Dataset Diagram", "Wasserstein distance", "topology_to_ideal.png"),
    )
    for metric, title, ylabel, filename in metric_graphs:
        path = directory / filename
        _save_metric_grid(history, metric, title, ylabel, path)
        written.append(path)

    indices, layers, dimensions = _topology_layout(history)
    epochs = [history["epoch"][index] for index in indices]
    geometry_path = directory / "representation_geometry.png"
    figure, axes = plt.subplots(1, 3, figsize=(18, 4))
    geometry_metrics = (
        ("intrinsic_dim", "Intrinsic Dimension"),
        ("nc_within_class_var", "Within-class Variance"),
        ("nc_simplex_score", "Class-mean Simplex Score"),
    )
    for axis, (key, title) in zip(axes, geometry_metrics):
        for layer in layers:
            values = [history["topology"][index][layer].get(key, 0.0) for index in indices]
            axis.plot(epochs, values, marker="o", markersize=3, label=f"Layer {layer}")
        axis.set(title=title, xlabel="Epoch")
        axis.grid(True, alpha=0.3)
        axis.legend()
    _draw_stage_changes(axes, history)
    figure.tight_layout()
    figure.savefig(geometry_path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    written.append(geometry_path)

    betti_curve_path = directory / "topology_betti_curves_latest.png"
    latest = indices[-1]
    figure, axes = plt.subplots(1, len(dimensions),
                               figsize=(6 * len(dimensions), 4), squeeze=False)
    for column, dimension in enumerate(dimensions):
        axis = axes[0, column]
        for layer in layers:
            diagrams = history["topology"][latest][layer].get("diagrams", [])
            if dimension >= len(diagrams) or not len(diagrams[dimension]):
                continue
            diagram = diagrams[dimension]
            finite_deaths = [float(death) for _, death in diagram if float(death) < float("inf")]
            max_scale = max(finite_deaths, default=1.0)
            thresholds = [max_scale * index / 199 for index in range(200)]
            axis.plot(thresholds, betti_curve(diagram, thresholds), label=f"Layer {layer}")
        axis.set(title=f"Betti Curve H{dimension} at Epoch {history['epoch'][latest]}",
                 xlabel="Filtration scale", ylabel="Features alive")
        axis.grid(True, alpha=0.3)
        axis.legend()
    figure.tight_layout()
    figure.savefig(betti_curve_path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    written.append(betti_curve_path)

    phase_path = directory / "topology_accuracy_phase.png"
    figure, axis = plt.subplots(figsize=(7, 5))
    accuracies = [history["val_acc"][index] for index in indices]
    for layer in layers:
        persistence = [
            history["topology"][index][layer].get("total_persistence_1", 0.0)
            for index in indices
        ]
        axis.scatter(persistence, accuracies, c=epochs, cmap="viridis", label=f"Layer {layer}")
    axis.set(title="Validation Accuracy vs H1 Persistence",
             xlabel="Total H1 persistence", ylabel="Validation accuracy")
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(phase_path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    written.append(phase_path)
    return written
