"""Topology metrics with expensive third-party dependencies loaded on demand."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from .training import extract_all_hidden_states_with_labels


def _ripser(points: np.ndarray, *, maxdim: int):
    from ripser import ripser

    return ripser(points, maxdim=maxdim)


def _wasserstein(first: np.ndarray, second: np.ndarray) -> float:
    from persim import wasserstein

    return float(wasserstein(first, second))


def _sample_indices(length: int, max_samples: int, rng=None) -> np.ndarray:
    # A fixed default keeps checkpoint-to-checkpoint distances from measuring
    # random resampling noise instead of representation change.
    generator = rng if rng is not None else np.random.default_rng(0)
    count = min(length, max_samples)
    return generator.choice(length, count, replace=False)


def intrinsic_dimension(hidden_states: np.ndarray) -> float:
    """Estimate intrinsic dimension using the covariance participation ratio."""
    centered = hidden_states - hidden_states.mean(axis=0)
    covariance = np.atleast_2d(np.cov(centered, rowvar=False))
    eigenvalues = np.maximum(np.linalg.eigvalsh(covariance), 1e-12)
    return float(eigenvalues.sum() ** 2 / np.square(eigenvalues).sum())


def remove_infinite(diagram: np.ndarray) -> np.ndarray:
    diagram = np.asarray(diagram)
    if diagram.size == 0:
        return diagram.reshape(0, 2)
    return diagram[np.isfinite(diagram[:, 1])]


def compute_simplex_score(means_matrix: np.ndarray) -> float:
    """Score class means by uniformity of their off-diagonal cosine values."""
    if len(means_matrix) < 2:
        return 0.0
    norms = np.linalg.norm(means_matrix, axis=1, keepdims=True)
    normalized = means_matrix / (norms + 1e-8)
    similarities = normalized @ normalized.T
    off_diagonal = similarities[~np.eye(len(means_matrix), dtype=bool)]
    return float(-np.std(off_diagonal))


def compute_neural_collapse_metrics(
    hidden_states: np.ndarray,
    labels: np.ndarray,
    max_samples: int = 1_000,
    *,
    rng=None,
    persistence_fn: Callable | None = None,
) -> dict[str, float | int]:
    if len(hidden_states) > max_samples:
        indices = _sample_indices(len(hidden_states), max_samples, rng)
        hidden_states, labels = hidden_states[indices], labels[indices]

    classes = np.unique(labels)
    class_points = [hidden_states[labels == label] for label in classes]
    nonempty = [points for points in class_points if len(points)]
    total = sum(len(points) for points in nonempty)
    within_variance = (
        sum(np.var(points, axis=0).mean() * len(points) for points in nonempty) / total
        if total
        else 0.0
    )
    means = np.array([points.mean(axis=0) for points in nonempty])
    diagrams = (persistence_fn or _ripser)(hidden_states, maxdim=2)["dgms"]
    dimension_two = remove_infinite(diagrams[2]) if len(diagrams) > 2 else np.empty((0, 2))
    return {
        "nc_within_class_var": float(within_variance),
        "nc_simplex_score": compute_simplex_score(means),
        "nc_betti_2": len(dimension_two),
        "nc_total_persistence_2": float(np.diff(dimension_two).sum()),
        "nc_num_classes": len(nonempty),
    }


def compute_dataset_topology(
    dataset,
    p: int,
    c: int,
    max_samples: int = 2_000,
    *,
    rng=None,
    persistence_fn: Callable | None = None,
    verbose: bool = True,
) -> dict:
    """Compute persistent homology of ``(a, b, result)`` dataset points."""
    if not len(dataset):
        raise ValueError("cannot analyze an empty dataset")
    indices = _sample_indices(len(dataset), max_samples, rng)
    points = []
    for index in indices:
        inputs, target = dataset[int(index)]
        a, b = inputs[:2].cpu().numpy()
        points.append([a, b, target.item()])
    normalized = np.asarray(points, dtype=float) / p
    diagrams = (persistence_fn or _ripser)(normalized, maxdim=2)["dgms"]

    if verbose:
        print(f"Dataset topology: (a^{c} + b^{c}) mod {p}; shape={normalized.shape}")
        for dimension, diagram in enumerate(diagrams[:3]):
            finite = remove_infinite(diagram)
            maximum = np.diff(finite).max() if len(finite) else 0.0
            print(f"  H{dimension}: {len(diagram)} features, max persistence={maximum:.4f}")

    return {
        "diagrams": diagrams,
        "data_points": normalized,
        "stats": {
            f"betti_{dimension}": len(diagrams[dimension]) if dimension < len(diagrams) else 0
            for dimension in range(3)
        },
    }


def compute_wasserstein_distance_to_ideal(
    model_diagrams,
    ideal_diagrams,
    maxdim: int = 2,
    *,
    distance_fn: Callable[[np.ndarray, np.ndarray], float] | None = None,
) -> dict[str, float]:
    distance = distance_fn or _wasserstein
    result = {}
    for dimension in range(maxdim + 1):
        value = 0.0
        if dimension < len(model_diagrams) and dimension < len(ideal_diagrams):
            learned = remove_infinite(model_diagrams[dimension])
            ideal = remove_infinite(ideal_diagrams[dimension])
            if len(learned) and len(ideal):
                value = float(distance(ideal, learned))
        result[f"wasserstein_to_ideal_{dimension}"] = value
    return result


def _empty_metrics(maxdim: int, include_labels: bool) -> dict:
    metrics = {"intrinsic_dim": 0.0, "diagrams": []}
    for dimension in range(maxdim + 1):
        for name in ("total", "avg", "max", "var"):
            metrics[f"{name}_persistence_{dimension}"] = 0.0
        metrics[f"betti_{dimension}"] = 0
        metrics[f"long_lived_{dimension}"] = 0
        metrics[f"wasserstein_shift_{dimension}"] = 0.0
        metrics[f"wasserstein_to_ideal_{dimension}"] = 0.0
    if include_labels:
        metrics.update(
            nc_within_class_var=0.0,
            nc_simplex_score=0.0,
            nc_betti_2=0,
            nc_total_persistence_2=0.0,
            nc_num_classes=0,
        )
    return metrics


def compute_topology(
    hidden_states: np.ndarray,
    labels: np.ndarray | None = None,
    prev_diagrams=None,
    ideal_diagrams=None,
    max_samples: int = 800,
    maxdim: int = 2,
    *,
    rng=None,
    persistence_fn: Callable | None = None,
    distance_fn: Callable | None = None,
    raise_on_error: bool = False,
) -> dict:
    """Compute topology statistics; dependencies can be injected in unit tests."""
    if len(hidden_states) > max_samples:
        indices = _sample_indices(len(hidden_states), max_samples, rng)
        hidden_states = hidden_states[indices]
        labels = labels[indices] if labels is not None else None
    hidden_states = np.nan_to_num(hidden_states, nan=0.0, posinf=1e6, neginf=-1e6)
    standard_deviation = hidden_states.std(axis=0)
    standard_deviation[standard_deviation == 0] = 1.0
    normalized = (hidden_states - hidden_states.mean(axis=0)) / standard_deviation

    try:
        persistence = persistence_fn or _ripser
        diagrams = persistence(normalized, maxdim=maxdim)["dgms"]
        metrics: dict = {"diagrams": diagrams, "intrinsic_dim": intrinsic_dimension(normalized)}
        distance = distance_fn or _wasserstein
        for dimension in range(maxdim + 1):
            diagram = diagrams[dimension] if dimension < len(diagrams) else np.empty((0, 2))
            finite = remove_infinite(diagram)
            lifetimes = np.diff(finite).ravel()
            metrics[f"betti_{dimension}"] = len(diagram)
            metrics[f"total_persistence_{dimension}"] = float(lifetimes.sum())
            metrics[f"avg_persistence_{dimension}"] = float(lifetimes.mean()) if len(lifetimes) else 0.0
            metrics[f"max_persistence_{dimension}"] = float(lifetimes.max()) if len(lifetimes) else 0.0
            metrics[f"var_persistence_{dimension}"] = float(lifetimes.var()) if len(lifetimes) else 0.0
            metrics[f"long_lived_{dimension}"] = (
                int(np.sum(lifetimes > np.percentile(lifetimes, 75))) if len(lifetimes) else 0
            )
            shift = 0.0
            if prev_diagrams is not None and dimension < len(prev_diagrams):
                previous = remove_infinite(prev_diagrams[dimension])
                if len(previous) and len(finite):
                    shift = float(distance(previous, finite))
            metrics[f"wasserstein_shift_{dimension}"] = shift

        if ideal_diagrams is not None:
            metrics.update(
                compute_wasserstein_distance_to_ideal(
                    diagrams, ideal_diagrams, maxdim, distance_fn=distance
                )
            )
        if labels is not None:
            metrics.update(
                compute_neural_collapse_metrics(
                    normalized,
                    labels,
                    rng=rng,
                    persistence_fn=persistence,
                )
            )
        return metrics
    except Exception:
        if raise_on_error:
            raise
        return _empty_metrics(maxdim, labels is not None)


def analyze_topology_all_layers(
    model,
    loader,
    device,
    n_layers: int,
    prev_topology=None,
    ideal_topology=None,
    maxdim: int = 2,
    **topology_options,
) -> dict[int, dict]:
    result = {}
    ideal_diagrams = ideal_topology["diagrams"] if ideal_topology is not None else None
    for layer_idx in range(n_layers):
        states, labels = extract_all_hidden_states_with_labels(model, loader, device, layer_idx)
        previous = prev_topology[layer_idx]["diagrams"] if prev_topology is not None else None
        result[layer_idx] = compute_topology(
            states,
            labels,
            previous,
            ideal_diagrams,
            maxdim=maxdim,
            **topology_options,
        )
    return result
