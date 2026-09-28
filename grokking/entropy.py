"""Representation spectral entropy and persistence-lifetime entropy (natural logs)."""

import numpy as np


def distribution_entropy(values):
    """Zero-mass distributions are undefined, represented by NaN, never fake collapse."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("expected a finite, nonnegative vector")
    total = float(values.sum())
    if total == 0:
        return {"entropy": float("nan"), "normalized_entropy": float("nan"),
                "effective_rank": float("nan"), "defined": False}
    probabilities = values[values > 0] / total
    entropy = max(0.0, float(-np.sum(probabilities * np.log(probabilities))))
    normalized = entropy / np.log(len(values)) if len(values) > 1 else 0.0
    return {"entropy": entropy, "normalized_entropy": float(normalized),
            "effective_rank": float(np.exp(entropy)), "defined": True}


def spectral_entropy(states):
    """Entropy of centered covariance X.T @ X / N, without coordinate standardization."""
    states = np.asarray(states, dtype=np.float64)
    if states.ndim != 2 or not states.shape[0] or not states.shape[1] or not np.isfinite(states).all():
        raise ValueError("expected a nonempty finite sample-by-feature matrix")
    centered = states - states.mean(axis=0)
    eigenvalues = np.maximum(np.linalg.eigvalsh(centered.T @ centered / len(states)), 0)[::-1]
    result = distribution_entropy(eigenvalues)
    result.update(eigenvalues=eigenvalues, sample_count=len(states), feature_count=states.shape[1],
                  total_variance=float(eigenvalues.sum()))
    return result


def persistent_entropy(diagram):
    """Shannon entropy of positive finite lifetimes; essential/zero bars excluded."""
    diagram = np.asarray(diagram, dtype=np.float64).reshape(-1, 2)
    finite = diagram[np.isfinite(diagram).all(axis=1)]
    lifetimes = finite[:, 1] - finite[:, 0]
    if (lifetimes < 0).any():
        raise ValueError("persistence death precedes birth")
    positive = lifetimes[lifetimes > 0]
    result = distribution_entropy(positive)
    result.update(positive_bar_count=len(positive), excluded_bar_count=len(diagram) - len(positive),
                  total_lifetime=float(positive.sum()))
    return result
