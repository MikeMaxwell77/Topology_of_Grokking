"""Experiment configuration kept separate from execution logic."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ExperimentConfig:
    modulus: int = 113
    exponent: int = 1
    input_range: int = 300
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 2
    d_ff: int = 512
    train_fraction: float = 0.3
    batch_size: int = 512
    learning_rate: float = 1e-3
    weight_decay: float = 1.0
    num_epochs: int = 10_000
    log_interval: int = 1
    tda_interval: int = 10
    seed: int = 42

    def __post_init__(self) -> None:
        if self.modulus <= 0:
            raise ValueError("modulus must be positive")
        if self.input_range <= 0:
            raise ValueError("input_range must be positive")
        if not 0 < self.train_fraction < 1:
            raise ValueError("train_fraction must be between 0 and 1")
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if min(self.batch_size, self.num_epochs, self.log_interval, self.tda_interval) <= 0:
            raise ValueError("batch size, epochs, and intervals must be positive")
