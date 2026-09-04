"""Dataset construction for modular-arithmetic experiments."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class ModularArithmeticDataset(Dataset):
    """Samples ``[a, b, exponent, modulus] -> result``.

    A local random generator makes construction deterministic without changing
    NumPy's process-wide random state.
    """

    def __init__(
        self,
        p: int = 1,
        c: int = 1,
        r: int = 1,
        train: bool = True,
        train_fraction: float = 0.3,
        seed: int = 42,
    ) -> None:
        if p <= 0:
            raise ValueError("p must be positive")
        if r <= 0:
            raise ValueError("r must be positive")
        if not 0 <= train_fraction <= 1:
            raise ValueError("train_fraction must be between 0 and 1")

        self.p = p
        self.c = c
        self.r = r

        pairs = np.array(
            [(a, b, c, p, (a**c + b**c) % p) for a in range(r) for b in range(r)],
            dtype=np.int64,
        )
        np.random.default_rng(seed).shuffle(pairs)
        split_index = int(len(pairs) * train_fraction)
        self.data = pairs[:split_index] if train else pairs[split_index:]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        a, b, exponent, modulus, result = self.data[index]
        inputs = torch.tensor([a, b, exponent, modulus], dtype=torch.long)
        target = torch.tensor(result, dtype=torch.long)
        return inputs, target


def make_dataloaders(
    *,
    p: int,
    c: int,
    r: int,
    train_fraction: float,
    batch_size: int,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader]:
    """Build matching train/validation loaders from one deterministic split."""
    train_data = ModularArithmeticDataset(p, c, r, True, train_fraction, seed)
    validation_data = ModularArithmeticDataset(p, c, r, False, train_fraction, seed)
    generator = torch.Generator().manual_seed(seed)
    return (
        DataLoader(train_data, batch_size=batch_size, shuffle=True, generator=generator),
        DataLoader(validation_data, batch_size=batch_size, shuffle=False),
    )
