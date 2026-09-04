"""Pure helpers for inspecting saved experiment histories."""

from __future__ import annotations

import pickle
from pathlib import Path

def load_history(path: str | Path = "grokking_history.pkl") -> dict:
    with Path(path).open("rb") as stream:
        return pickle.load(stream)


def summarize_history(history: dict, *, grokking_threshold: float = 0.9) -> dict:
    """Return machine-readable summary data without printing or plotting."""
    if not history.get("epoch"):
        return {"checkpoints": 0, "grokking_epoch": None, "layers": 0}
    crossing = next(
        (index for index, accuracy in enumerate(history["val_acc"])
         if accuracy > grokking_threshold),
        None,
    )
    topology = history.get("topology", [])
    return {
        "checkpoints": len(history["epoch"]),
        "final_train_accuracy": history["train_acc"][-1],
        "final_validation_accuracy": history["val_acc"][-1],
        "grokking_epoch": history["epoch"][crossing] if crossing is not None else None,
        "layers": len(topology[0]) if topology else 0,
    }


def format_summary(summary: dict) -> str:
    if not summary["checkpoints"]:
        return "No checkpoints found."
    grokking = summary["grokking_epoch"]
    grokking_text = f"epoch {grokking}" if grokking is not None else "not detected"
    return "\n".join(
        (
            f"Checkpoints: {summary['checkpoints']}",
            f"Final train accuracy: {summary['final_train_accuracy']:.4f}",
            f"Final validation accuracy: {summary['final_validation_accuracy']:.4f}",
            f"Grokking: {grokking_text}",
            f"Layers tracked: {summary['layers']}",
        )
    )
