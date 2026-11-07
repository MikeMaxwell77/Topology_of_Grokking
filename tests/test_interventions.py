import unittest
import json
import uuid
from dataclasses import asdict
from pathlib import Path

import torch

from grokking.interventions import control_delta, displacement, fit_clusters, geometry, sweep
from grokking.config import TopologyStudyConfig
from grokking.study import StudyTransformer, build_data
from grokking.intervention_report import contrasts


class InterventionTests(unittest.TestCase):
    def test_sufficiency_checkpoint_to_report(self):
        torch.set_num_threads(1)
        config = TopologyStudyConfig(modulus=7, d_model=8, n_heads=2, d_ff=16,
                                     probe_size=8)
        model = StudyTransformer(config)
        _, _, indices = build_data(config)
        root = Path(__file__).resolve().parents[1] / 'outputs' / ('intervention_test_' + uuid.uuid4().hex)
        root.mkdir(parents=True)
        directory = root
        checkpoint = Path(directory) / 'checkpoint.pt'
        torch.save(dict(config=asdict(config), model=model.state_dict(),
                        indices=indices, step=0), checkpoint)
        original = checkpoint.read_bytes()
        for mode in ('contract', 'tighten', 'separate'):
            output = Path(directory) / mode
            rows = sweep([checkpoint], output, clusters=2, strengths=(0., .5),
                         repeats=2, intervention=mode)
            self.assertEqual(len(rows), 12)
            self.assertEqual(json.loads((output / 'protocol.json').read_text())['intervention'], mode)
            self.assertIn('Experiment 1' if mode == 'contract' else 'Experiment 2',
                          (output / 'report.md').read_text())
            for contrast in contrasts(rows):
                self.assertLess(contrast['max_displacement_rms_mismatch'], 1e-5)
        self.assertEqual(checkpoint.read_bytes(), original)

    def test_sufficiency_geometry(self):
        z = torch.tensor([[-3., -1.], [-3., 1.], [3., -1.], [3., 1.]])
        centers = fit_clusters(z, 2)
        for mode in ('tighten', 'separate'):
            zero, _ = displacement(z, centers, z.mean(0), 0., mode)
            torch.testing.assert_close(zero, torch.zeros_like(z))
            delta, groups = displacement(z, centers, z.mean(0), .5, mode)
            altered = z + delta
            for group in groups.unique():
                selected = groups == group
                before = z[selected] - z[selected].mean(0)
                after = altered[selected] - altered[selected].mean(0)
                torch.testing.assert_close(after, before * (.5 if mode == 'tighten' else 1.))
                torch.testing.assert_close(altered[selected].mean(0),
                                           z[selected].mean(0) * (1.5 if mode == 'separate' else 1.))
            for kind in ('random', 'rotated'):
                torch.testing.assert_close(control_delta(delta, kind, 77).norm(dim=1),
                                           delta.norm(dim=1))
        heldout = torch.tensor([[-4., 2.], [4., -2.]])
        delta, assigned = displacement(heldout, centers, z.mean(0), 1., 'tighten')
        torch.testing.assert_close(heldout + delta, centers[assigned])

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
