import unittest

import torch

from grokking.config import ExperimentConfig
from grokking.dyck import VOCABULARY, build_dyck, is_dyck
from grokking.experiment import config_from_args, parse_args, run_experiment


class DyckTests(unittest.TestCase):
    def test_labels_balance_counts_and_disjoint_reproducible_splits(self):
        train, validation, _ = build_dyck(16, 64, 64, 42)
        repeat, _, _ = build_dyck(16, 64, 64, 42)
        self.assertTrue(torch.equal(train.tensors[0], repeat.tensors[0]))
        decode = {value: key for key, value in VOCABULARY.items()}
        for dataset in (train, validation):
            self.assertEqual(dataset.tensors[1].sum().item(), 32)
            for inputs, label in dataset:
                text = "".join(decode[token] for token in inputs[:-1].tolist())
                self.assertEqual(is_dyck(text), bool(label))
                self.assertEqual(text.count("("), text.count(")"))
                self.assertEqual(text.count("["), text.count("]"))
                depth = 0
                for token in text:
                    depth += 1 if token in "([" else -1
                    self.assertGreaterEqual(depth, 0)
                self.assertEqual(depth, 0)
        self.assertFalse({tuple(row.tolist()) for row in train.tensors[0]} &
                         {tuple(row.tolist()) for row in validation.tensors[0]})

    def test_training_cli_and_language_stage_guard(self):
        config = config_from_args(parse_args(["--task", "dyck", "--dyck-length", "8",
            "--train-examples", "16", "--validation-examples", "16", "--d-model", "8",
            "--n-heads", "2", "--n-layers", "1", "--d-ff", "16", "--epochs", "2",
            "--batch-size", "8", "--tda-interval", "0", "--entropy-probe-size", "8"]))
        model, history = run_experiment(config, device="cpu")
        self.assertEqual(model.sequence_length, 9)
        self.assertEqual(len(history["val_loss"]), 2)
        self.assertTrue(all(value > 0 for value in history["val_loss"]))
        self.assertIsNone(history["ideal_topology"])
        with self.assertRaisesRegex(ValueError, "arithmetic stages"):
            run_experiment(config, stage_exponents=(1, 2))

    def test_rejects_invalid_settings_and_exhausted_space(self):
        with self.assertRaises(ValueError):
            ExperimentConfig(task="dyck", dyck_length=7)
        with self.assertRaises(ValueError):
            build_dyck(4, 64, 64)
