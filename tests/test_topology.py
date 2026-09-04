import unittest

import numpy as np

from grokking.topology import compute_topology, remove_infinite


def fake_persistence(points, *, maxdim):
    assert np.isfinite(points).all()
    return {"dgms": [
        np.array([[0.0, 0.2], [0.0, np.inf]]),
        np.array([[0.1, 0.4]]),
        np.empty((0, 2)),
    ][: maxdim + 1]}


class TopologyTests(unittest.TestCase):
    def test_remove_infinite_deaths(self):
        result = remove_infinite(np.array([[0.0, 1.0], [0.0, np.inf]]))
        np.testing.assert_array_equal(result, [[0.0, 1.0]])

    def test_compute_topology_supports_dependency_injection(self):
        states = np.array([[0.0, 0.0], [1.0, 0.5], [2.0, 1.5]])
        metrics = compute_topology(states, maxdim=1, persistence_fn=fake_persistence,
                                   distance_fn=lambda first, second: 3.0, raise_on_error=True)
        self.assertEqual(metrics["betti_0"], 2)
        self.assertAlmostEqual(metrics["total_persistence_0"], 0.2)
        self.assertAlmostEqual(metrics["total_persistence_1"], 0.3)
