"""Seeded Dyck-2 classification with matched bracket-count negatives."""

import random

import torch
from torch.utils.data import TensorDataset

VOCABULARY = {"<readout>": 0, "(": 1, ")": 2, "[": 3, "]": 4}


def is_dyck(sequence):
    stack = []
    for token in sequence:
        if token in "([":
            stack.append(token)
        elif token in ")]":
            if not stack or stack.pop() != {")": "(", "]": "["}[token]:
                return False
        else:
            return False
    return not stack


def _valid_sequence(length, rng):
    stack, result = [], []
    opened = 0
    for _ in range(length):
        if opened < length // 2 and (not stack or rng.random() < 0.5):
            token = rng.choice("([")
            stack.append(token)
            opened += 1
        else:
            token = {"(": ")", "[": "]"}[stack.pop()]
        result.append(token)
    return "".join(result)


def build_dyck(length=32, train_examples=512, validation_examples=512, seed=42):
    """Negatives preserve per-type counts and balanced untyped nesting."""
    if length < 4 or length % 2:
        raise ValueError("Dyck length must be even and at least 4")
    if min(train_examples, validation_examples) < 2 or train_examples % 2 or validation_examples % 2:
        raise ValueError("Dyck split sizes must be even and at least 2")
    rng = random.Random(seed)
    seen = set()
    splits = []
    for count in (train_examples, validation_examples):
        examples = []
        for label in (0, 1):
            accepted = 0
            for _ in range(max(10000, count * 100)):
                sequence = _valid_sequence(length, rng)
                if not label:
                    round_closes = [i for i, token in enumerate(sequence) if token == ")"]
                    square_closes = [i for i, token in enumerate(sequence) if token == "]"]
                    if not round_closes or not square_closes:
                        continue
                    tokens = list(sequence)
                    a, b = rng.choice(round_closes), rng.choice(square_closes)
                    tokens[a], tokens[b] = tokens[b], tokens[a]
                    sequence = "".join(tokens)
                if sequence in seen:
                    continue
                seen.add(sequence)
                examples.append((sequence, label))
                accepted += 1
                if accepted == count // 2:
                    break
            else:
                raise ValueError("cannot generate enough unique Dyck strings; increase --dyck-length or reduce split sizes")
        rng.shuffle(examples)
        inputs = torch.tensor([[VOCABULARY[token] for token in text] + [0] for text, _ in examples])
        targets = torch.tensor([label for _, label in examples])
        splits.append(TensorDataset(inputs, targets))
    metadata = dict(task="dyck2", vocabulary=VOCABULARY, length=length, seed=seed,
                    train_examples=train_examples, validation_examples=validation_examples,
                    negatives="swap differently typed closing brackets; preserve counts and untyped nesting",
                    labels={0: "invalid", 1: "valid"})
    return splits[0], splits[1], metadata
