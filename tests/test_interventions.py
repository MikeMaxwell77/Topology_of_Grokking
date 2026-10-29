import unittest

import torch

from grokking.interventions import control_delta, displacement, fit_clusters, geometry
from grokking.intervention_report import contrasts


class InterventionTests(unittest.TestCase):
    def test_paired_report_contrast(self):
        common = dict(checkpoint='example.pt', split='heldout', step=100,
                      accuracy=.8, between_fraction=.5, displacement_rms=1.)
        rows = [dict(common, strength=0., condition='targeted', loss=2.),
                dict(common, strength=.5, condition='targeted', loss=5.)]
        for kind in ('random', 'rotated'):
            rows.extend(dict(common, strength=.5, condition=kind, loss=loss)
                        for loss in (3., 4.))
        result = contrasts(rows)
        self.assertEqual(len(result), 2)
        for row in result:
            self.assertEqual(row['targeted_loss_increase'], 3.)
            self.assertEqual(row['control_mean_loss_increase'], 1.5)
            self.assertEqual(row['excess_targeted_loss'], 1.5)
            self.assertEqual(row['max_displacement_rms_mismatch'], 0.)

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
