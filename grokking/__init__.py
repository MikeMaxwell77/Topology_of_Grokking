"""Reusable components for modular-arithmetic grokking experiments.

Public attributes are loaded lazily so lightweight analysis utilities can be
used without importing the machine-learning stack.
"""

from importlib import import_module

__all__ = [
    "ExperimentConfig",
    "ModularArithmeticDataset",
    "TinyTransformer",
    "evaluate",
    "make_dataloaders",
    "train_epoch",
]

_EXPORTS = {
    "ExperimentConfig": ("grokking.config", "ExperimentConfig"),
    "ModularArithmeticDataset": ("grokking.data", "ModularArithmeticDataset"),
    "TinyTransformer": ("grokking.network", "TinyTransformer"),
    "evaluate": ("grokking.training", "evaluate"),
    "make_dataloaders": ("grokking.data", "make_dataloaders"),
    "train_epoch": ("grokking.training", "train_epoch"),
}


def __getattr__(name):
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError as error:
        raise AttributeError(name) from error
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value
