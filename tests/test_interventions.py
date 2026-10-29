import unittest

import torch

from grokking.interventions import control_delta, displacement, fit_clusters, geometry


class InterventionTests(unittest.TestCase):
    def test_centers_collapse_and_controls_match(self):
        z = torch.tensor([[-3., -.1], [-3., .1], [3., -.1], [3., .1]])
        centers = fit_clusters(z, 2)
        delta, groups = displacement(z, centers, z.mean(0), 1.)
        altered = z + delta
        self.assertLess(geometry(altered, groups)['between_fraction'], 1e-5)
        for group in groups.unique():
            selected = groups == group
            torch.testing.assert_close(altered[selected] - altered[selected].mean(0),
                                       z[selected] - z[selected].mean(0))
        for kind in ('random', 'rotated'):
            control = control_delta(delta, kind, 77)
            torch.testing.assert_close(delta.norm(dim=1), control.norm(dim=1))
            if kind == 'rotated':
                torch.testing.assert_close(delta @ delta.T, control @ control.T)

    def test_identity_and_local_rng(self):
        z = torch.arange(24, dtype=torch.float32).reshape(8, 3)
        state = torch.get_rng_state().clone()
        centers = fit_clusters(z, 2)
        delta, _ = displacement(z, centers, z.mean(0), 0.)
        torch.testing.assert_close(z + delta, z)
        control_delta(delta, 'random', 3)
        control_delta(delta, 'rotated', 3)
        self.assertTrue(torch.equal(state, torch.get_rng_state()))
