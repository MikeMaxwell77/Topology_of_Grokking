import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from grokking.agreement import GRAMMAR, REVISION, load_agreement
from grokking.config import ExperimentConfig
from grokking.experiment import run_experiment
from grokking.network import TinyTransformer
from grokking.plotting import save_tracking_plots


def write_fixture(directory):
    files = {
        "tag_token_map.txt": "det the\nn_s cat\nn_p cats\nn_s dog\nn_p dogs\nprep near\nv_s_intrans sleeps\nv_p_intrans sleep\n",
        "agreement_hr.gr": "fixture",
        "train": "the cat sleeps\nthe cats sleep\nthe cat sleeps\n",
        "val": "the cat sleeps\nthe dog sleeps\nthe dogs sleep\n",
        "g1_test": "the cat near the dogs sleeps\nthe cats near the dog sleep\n",
        "g2_test": "the cat near the dogs sleep\nthe cats near the dog sleeps\n",
    }
    manifest = {"revision": REVISION, "grammar": GRAMMAR, "files": {}}
    for name, text in files.items():
        payload = text.encode()
        (directory / name).write_bytes(payload)
        manifest["files"][name] = {"sha256": hashlib.sha256(payload).hexdigest()}
    (directory / "manifest.json").write_text(json.dumps(manifest))


class AgreementTests(unittest.TestCase):
    def test_adaptation_and_disjoint_splits(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            write_fixture(directory)
            datasets, metadata = load_agreement(directory)
            self.assertEqual(metadata["split_sizes"], dict(train=2, val=2, g1_test=2, g2_test=2))
            self.assertEqual(datasets["g1_test"][0][1].item(), 0)
            self.assertEqual(datasets["g2_test"][0][1].item(), 1)
            inputs, _ = datasets["train"][0]
            self.assertEqual(inputs.shape, (6,))
            self.assertEqual(inputs[-1].item(), metadata["vocabulary"]["<cls>"])
            (directory / "train").write_text("corrupted")
            with self.assertRaisesRegex(ValueError, "checksum"):
                load_agreement(directory)

    def test_padding_does_not_affect_logits_and_hidden_matches_readout(self):
        model = TinyTransformer(vocab_size=10, d_model=8, n_heads=2, n_layers=2,
                                d_ff=16, sequence_length=6, padding_idx=0, num_classes=2).eval()
        inputs = torch.tensor([[2, 3, 0, 0, 0, 1]])
        with torch.no_grad():
            expected = model(inputs)
            torch.testing.assert_close(expected, model.output(model.get_hidden_states(inputs, 1)))
            model.embed.weight[0].fill_(100)
            torch.testing.assert_close(expected, model(inputs))

    def test_experiment_records_entropy_and_generalization_without_arithmetic_topology(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            write_fixture(directory)
            config = ExperimentConfig(task="simple_agreement", agreement_data_dir=str(directory),
                                      d_model=8, n_heads=2, n_layers=1, d_ff=16,
                                      num_epochs=2, batch_size=2, tda_interval=0, entropy_interval=1)
            with patch("grokking.experiment.compute_dataset_topology", side_effect=AssertionError):
                _, history = run_experiment(config, device="cpu",
                                            dataset_topology_fn=lambda *a, **k: self.fail("arithmetic topology"))
            self.assertEqual(history["optimizer_step"], [1, 2])
            self.assertEqual([row["optimizer_step"] for row in history["entropy"]], [0, 1, 2])
            self.assertEqual(len(history["generalization_acc"]), 2)
            self.assertIsNone(history["ideal_topology"])
            self.assertTrue(np.isfinite(history["entropy"][-1]["layers"][0]["entropy"]))
            paths = save_tracking_plots(history, directory / "plots")
            self.assertEqual(len(paths), 2)
            self.assertTrue(all(path.exists() for path in paths))
