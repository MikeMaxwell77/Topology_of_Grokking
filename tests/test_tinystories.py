import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import torch

from grokking.config import ExperimentConfig
from grokking.experiment import run_experiment
from grokking.tinystories import load_tinystories, next_byte_dataset


class TinyStoriesTests(unittest.TestCase):
    def test_windows_are_deterministic_and_do_not_cross_boundaries(self):
        first = next_byte_dataset(["abcdef", "uvwxyz"], 3, 6, 42)
        second = next_byte_dataset(["abcdef", "uvwxyz"], 3, 6, 42)
        self.assertTrue(torch.equal(first.tensors[0], second.tensors[0]))
        expected = {b"abcd", b"bcde", b"cdef", b"uvwx", b"vwxy", b"wxyz"}
        actual = {bytes(row.tolist() + [target.item()]) for row, target in first}
        self.assertEqual(actual, expected)

    def test_offline_training_integrity_and_disjoint_contexts(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            manifest = {"files": {}}
            for split, text in (("train", "The cat played outside."),
                                ("validation", "The cat walked home.")):
                payload = (json.dumps(text) + "\n").encode()
                filename = f"{split}.jsonl"
                (directory / filename).write_bytes(payload)
                manifest["files"][filename] = {"sha256": hashlib.sha256(payload).hexdigest()}
            (directory / "manifest.json").write_text(json.dumps(manifest))
            config = ExperimentConfig(task="tinystories", tinystories_data_dir=temporary,
                                      context_length=4, train_examples=32, validation_examples=32,
                                      d_model=8, n_heads=2, n_layers=1, d_ff=16,
                                      batch_size=4, num_epochs=1, tda_interval=0, entropy_probe_size=8)
            model, history = run_experiment(config, device="cpu")
            self.assertEqual(model.vocab_size, 256)
            self.assertGreater(history["val_loss"][0], 0)
            train, validation, _ = load_tinystories(temporary, 4, 32, 32)
            self.assertFalse({tuple(row.tolist()) for row in train.tensors[0]} &
                             {tuple(row.tolist()) for row in validation.tensors[0]})
            (directory / "train.jsonl").write_text('"tampered"\n')
            with self.assertRaisesRegex(ValueError, "checksum"):
                load_tinystories(temporary)
