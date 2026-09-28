import unittest

import numpy as np

from grokking.entropy import persistent_entropy, spectral_entropy
from grokking.topology import compute_topology


class EntropyTests(unittest.TestCase):
    def test_isotropic_and_rank_one_covariance(self):
        isotropic = spectral_entropy([[1, 0], [-1, 0], [0, 1], [0, -1]])
        self.assertAlmostEqual(isotropic["normalized_entropy"], 1)
        self.assertAlmostEqual(isotropic["effective_rank"], 2)
        self.assertAlmostEqual(spectral_entropy([[1, 2], [-1, -2]])["entropy"], 0)

    def test_covariance_entropy_invariant_to_shift_scale_rotation(self):
        states = np.random.default_rng(0).normal(size=(20, 3)) * [1, 2, 4]
        rotation, _ = np.linalg.qr(np.random.default_rng(1).normal(size=(3, 3)))
        self.assertAlmostEqual(spectral_entropy(states)["entropy"],
                               spectral_entropy(7 * states @ rotation + 5)["entropy"])

    def test_degenerate_spectrum_is_undefined(self):
        result = spectral_entropy(np.ones((5, 3)))
        self.assertFalse(result["defined"])
        self.assertTrue(np.isnan(result["entropy"]))
        with self.assertRaises(ValueError):
            spectral_entropy([[float("nan"), 1]])

    def test_persistent_entropy_excludes_essential_and_zero_bars(self):
        result = persistent_entropy([[0, 1], [3, 4], [0, np.inf], [2, 2]])
        self.assertAlmostEqual(result["entropy"], np.log(2))
        self.assertAlmostEqual(result["normalized_entropy"], 1)
        self.assertEqual(result["positive_bar_count"], 2)
        self.assertEqual(result["excluded_bar_count"], 2)
        self.assertEqual(persistent_entropy([[0, 2]])["entropy"], 0)
        self.assertFalse(persistent_entropy([])["defined"])

    def test_wasserstein_detects_disappearing_features(self):
        from persim import wasserstein
        previous = [np.array([[0., 2.]])]
        metrics = compute_topology(np.array([[0., 1.], [1., 0.]]), maxdim=0,
                                   prev_diagrams=previous,
                                   persistence_fn=lambda *a, **k: {"dgms": [np.empty((0, 2))]},
                                   distance_fn=wasserstein, raise_on_error=True)
        self.assertAlmostEqual(metrics["wasserstein_shift_0"], np.sqrt(2))
        self.assertTrue(np.isnan(metrics["persistent_entropy_0"]))

    def test_topology_failure_is_marked(self):
        def fail(*args, **kwargs):
            raise RuntimeError("deliberate failure")
        metrics = compute_topology(np.eye(3), persistence_fn=fail)
        self.assertFalse(metrics["topology_valid"])
        self.assertTrue(np.isnan(metrics["wasserstein_shift_0"]))
        self.assertIn("deliberate failure", metrics["topology_error"])
