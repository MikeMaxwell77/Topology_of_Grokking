import unittest

from grokking.analysis import format_summary, summarize_history
from grokking.plotting import betti_curve, topology_sample_indices


class AnalysisTests(unittest.TestCase):
    def test_summary_detects_first_threshold_crossing(self):
        history = {
            "epoch": [0, 10, 20],
            "train_acc": [0.2, 0.8, 1.0],
            "val_acc": [0.1, 0.91, 0.95],
            "topology": [{0: {}}, {0: {}}, {0: {}}],
        }
        summary = summarize_history(history)
        self.assertEqual(summary["grokking_epoch"], 10)
        self.assertEqual(summary["layers"], 1)
        self.assertIn("epoch 10", format_summary(summary))

    def test_topology_indices_only_include_computed_checkpoints(self):
        history = {"topology": [{}, {}, {}], "tda_computed": [True, False, True]}
        self.assertEqual(topology_sample_indices(history), [0, 2])

    def test_betti_curve_counts_only_intervals_alive_at_scale(self):
        diagram = [(0.0, 1.0), (0.5, 2.0), (0.0, float("inf"))]
        self.assertEqual(betti_curve(diagram, [0.25, 0.75, 1.5]), [2, 3, 2])
