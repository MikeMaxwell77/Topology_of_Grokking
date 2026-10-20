"""Experiment configuration kept separate from execution logic."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ExperimentConfig:
    task: str = "modular"
    agreement_data_dir: str = "data/simple_agreement"
    tinystories_data_dir: str = "data/tinystories"
    context_length: int = 64
    dyck_length: int = 32
    train_examples: int = 512
    validation_examples: int = 512
    entropy_interval: int = 10
    entropy_probe_size: int = 512
    modulus: int = 113
    exponent: int = 2
    input_range: int = 300
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 1
    d_ff: int = 512
    train_fraction: float = 0.3
    batch_size: int = 512
    learning_rate: float = 1e-3
    weight_decay: float = 1.0
    num_epochs: int = 10_000
    log_interval: int = 1
    tda_interval: int = 10
    tda_max_samples: int = 800
    seed: int = 42

    def __post_init__(self) -> None:
        if self.task not in ("modular", "simple_agreement", "dyck", "tinystories"):
            raise ValueError("unknown task")
        if self.task == "tinystories" and min(self.context_length, self.train_examples, self.validation_examples) <= 0:
            raise ValueError("context length and example counts must be positive")
        if self.task == "dyck":
            if self.dyck_length < 4 or self.dyck_length % 2:
                raise ValueError("Dyck length must be even and at least 4")
            if min(self.train_examples, self.validation_examples) < 2 or self.train_examples % 2 or self.validation_examples % 2:
                raise ValueError("Dyck split sizes must be even and at least 2")
        if self.entropy_interval <= 0 or self.entropy_probe_size <= 0 or self.tda_interval < 0 or self.tda_max_samples < 2:
            raise ValueError("entropy settings must be positive; tda_interval must be nonnegative")
        if self.modulus <= 0:
            raise ValueError("modulus must be positive")
        if self.input_range <= 0:
            raise ValueError("input_range must be positive")
        if not 0 < self.train_fraction < 1:
            raise ValueError("train_fraction must be between 0 and 1")
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if min(self.batch_size, self.num_epochs, self.log_interval) <= 0:
            raise ValueError("batch size, epochs, and intervals must be positive")

@dataclass(frozen=True)
class TopologyStudyConfig:
    """Fixed-task study; intentionally separate from historical experiments."""

    modulus: int = 113
    train_fraction: float = 0.3
    d_model: int = 128
    n_heads: int = 4
    d_ff: int = 512
    learning_rate: float = 0.001
    weight_decay: float = 1.0
    beta1: float = 0.9
    beta2: float = 0.98
    adam_eps: float = 1e-8
    steps: int = 40_000
    checkpoint_interval: int = 50
    evaluation_interval: int = 50
    evaluation_batch_size: int = 1024
    probe_size: int = 256
    seed: int = 42
    split_seed: int = 598
    probe_seed: int = 599
    analysis_seed: int = 600
    init_std: float = 0.1

    def __post_init__(self):
        if min(self.modulus, self.d_model, self.n_heads, self.d_ff,
               self.checkpoint_interval, self.evaluation_interval,
               self.evaluation_batch_size, self.probe_size) <= 0 or self.steps < 0:
            raise ValueError('dimensions and intervals must be positive; steps nonnegative')
        if self.modulus < 2 or not 0 < self.train_fraction < 1:
            raise ValueError('need modulus >= 2 and 0 < train_fraction < 1')
        if not 0 < int(self.modulus ** 2 * self.train_fraction) < self.modulus ** 2:
            raise ValueError('both splits must be nonempty')
        if self.d_model % self.n_heads or self.probe_size > self.modulus ** 2:
            raise ValueError('invalid head dimension or probe count')
        if self.learning_rate <= 0 or self.weight_decay < 0 or self.init_std <= 0 or self.adam_eps <= 0:
            raise ValueError('invalid optimizer or initialization setting')
        if not 0 <= self.beta1 < 1 or not 0 <= self.beta2 < 1:
            raise ValueError('Adam betas must lie in [0, 1)')
