"""Small, testable training and representation-extraction functions."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch
import torch.nn as nn


def classification_residuals(outputs: torch.Tensor, targets: torch.Tensor) -> np.ndarray:
    """Return logits minus one-hot targets for diagnostics."""
    one_hot = torch.nn.functional.one_hot(targets, num_classes=outputs.shape[1])
    return (outputs.detach().cpu() - one_hot.cpu()).numpy()


def print_residuals(outputs: torch.Tensor, targets: torch.Tensor) -> None:
    print(f"Residuals: {classification_residuals(outputs, targets)}")


def train_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device | str,
    *,
    criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
    residual_callback: Callable[[torch.Tensor, torch.Tensor], None] | None = None,
) -> tuple[float, float]:
    """Train once over ``loader`` and return mean batch loss and accuracy."""
    model.train()
    loss_function = criterion or nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        if residual_callback is not None:
            residual_callback(outputs, targets)
        loss = loss_function(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct += (outputs.argmax(dim=1) == targets).sum().item()
        total += targets.size(0)

    if total == 0:
        raise ValueError("cannot train on an empty loader")
    return total_loss / len(loader), correct / total


def evaluate(model: nn.Module, loader, device: torch.device | str) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            predictions = model(inputs).argmax(dim=1)
            correct += (predictions == targets).sum().item()
            total += targets.size(0)
    if total == 0:
        raise ValueError("cannot evaluate an empty loader")
    return correct / total


def extract_hidden_states(model, loader, device, layer_idx: int, *, with_labels: bool = False):
    """Extract finite hidden states, optionally paired with their labels."""
    model.eval()
    state_batches: list[np.ndarray] = []
    label_batches: list[np.ndarray] = []
    with torch.no_grad():
        for inputs, targets in loader:
            states = model.get_hidden_states(inputs.to(device), layer_idx).cpu().numpy()
            if np.isfinite(states).all():
                state_batches.append(states)
                if with_labels:
                    label_batches.append(targets.cpu().numpy())

    if not state_batches:
        states = np.zeros((10, model.d_model))
        return (states, np.zeros(10, dtype=int)) if with_labels else states

    states = np.vstack(state_batches)
    if with_labels:
        return states, np.concatenate(label_batches)
    return states


def extract_all_hidden_states(model, loader, device, layer_idx):
    return extract_hidden_states(model, loader, device, layer_idx)


def extract_all_hidden_states_with_labels(model, loader, device, layer_idx):
    return extract_hidden_states(model, loader, device, layer_idx, with_labels=True)
