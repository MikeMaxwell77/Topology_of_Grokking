import contextlib
import io
from pathlib import Path
import random
import uuid
import unittest
from unittest.mock import patch

import numpy as np
import torch

from grokking.config import TopologyStudyConfig
from grokking.study import StudyTransformer, build_data, load_checkpoint, run, rng_state
from grokking.study_analysis import (analyze, behavioral_timing, betti_curve, finite_diagram,
                                     geometry, measure, wasserstein)


class StudyTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.config = TopologyStudyConfig(modulus=7, d_model=16, n_heads=4, d_ff=32,
                                         steps=6, checkpoint_interval=2,
                                         evaluation_interval=2, probe_size=16)

    def test_pairs_splits_probes_and_rng(self):
        before = torch.get_rng_state().clone()
        x, y, indices = build_data(self.config)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        self.assertEqual(len(torch.unique(x[:, :2], dim=0)), 49)
        self.assertTrue(torch.equal(y, (x[:, 0] + x[:, 1]) % 7))
        self.assertTrue((x[:, 2] == 7).all())
        self.assertEqual(len(indices['train']), 14)
        self.assertEqual(set(indices['train'].tolist()) & set(indices['heldout'].tolist()), set())
        self.assertEqual(sorted(torch.cat((indices['train'], indices['heldout'])).tolist()), list(range(49)))
        self.assertTrue(torch.equal(indices['probes'], build_data(self.config)[2]['probes']))

    def test_architecture_and_token_extraction(self):
        model = StudyTransformer(self.config)
        self.assertFalse(any(isinstance(m, (torch.nn.LayerNorm, torch.nn.Dropout)) for m in model.modules()))
        x, _, _ = build_data(self.config)
        representations = model.representations(x[:5])
        self.assertEqual(representations['post_block_residual'].shape, (5, 16))
        self.assertEqual(representations['mlp_activation'].shape, (5, 32))
        self.assertTrue(torch.equal(model(x[:5]), model.unembed(representations['post_block_residual'])))
        captured = []
        handle = model.mlp_out.register_forward_hook(lambda m, a, out: captured.append(a[0]))
        model(x[:5])
        handle.remove()
        self.assertTrue(torch.equal(representations['mlp_activation'], captured[0][:, 2]))

    def test_geometry_scale_invariance_and_collapse(self):
        points = np.array([[0., 0.], [1., 0.], [0., 1.]])
        normalized, rms, rank = geometry(points)
        np.testing.assert_allclose(np.mean(np.sum(normalized ** 2, axis=1)), 1.)
        np.testing.assert_allclose(normalized, geometry(points * 7 + 3)[0])
        self.assertGreater(rank, 1)
        self.assertEqual(geometry(np.zeros((4, 2))), (None, 0., None))
        with self.assertRaises(ValueError):
            geometry(np.array([[np.nan], [1.]]))

    def test_known_cloud_and_diagonal_matching(self):
        angles = np.arange(32) * 2 * np.pi / 32
        circle = np.column_stack((np.cos(angles), np.sin(angles)))
        metrics, diagrams, _ = measure(circle, np.linspace(0, 3, 31))
        self.assertGreater(metrics['h1_max_finite_persistence'], 1.)
        _, line_diagrams, _ = measure(np.arange(8).reshape(-1, 1), np.linspace(0, 8, 31))
        self.assertEqual(len(line_diagrams[1]), 0)
        empty = np.empty((0, 2))
        diagram = np.array([[0., 2.], [0., np.inf]])
        self.assertEqual(len(finite_diagram(diagram)), 1)
        self.assertAlmostEqual(wasserstein(empty, diagram), np.sqrt(2))
        self.assertAlmostEqual(wasserstein(diagram, empty), np.sqrt(2))
        self.assertEqual(wasserstein(empty, empty), 0.)
        np.testing.assert_array_equal(betti_curve(diagram, np.array([0., 1., 2.])), [2, 2, 1])

    def test_behavioral_timing_and_absence(self):
        history = [dict(step=i * 50, train_accuracy=1. if i >= 2 else 0.,
                        heldout_accuracy=1. if i >= 15 else .1) for i in range(22)]
        timing = behavioral_timing(history)
        self.assertEqual(timing['memorization_step'], 100)
        self.assertEqual(timing['generalization_step'], 750)
        self.assertEqual(timing['delay_steps'], 650)
        self.assertTrue(timing['poor_plateau_detected'])
        self.assertIsNone(behavioral_timing(history[:10])['generalization_step'])

    def test_default_task_and_failed_topology_are_explicit(self):
        x, y, indices = build_data(TopologyStudyConfig())
        self.assertEqual(x.shape, (12769, 3))
        self.assertEqual(len(indices['train']), 3830)
        self.assertEqual(len(indices['probes']), 256)
        root = Path('outputs') / ('study_failure_test_' + uuid.uuid4().hex)
        with contextlib.redirect_stdout(io.StringIO()):
            run(self.config, root / 'train', stop_at=0)
            with patch('grokking.study_analysis.measure', side_effect=RuntimeError('synthetic failure')):
                rows, _ = analyze([root / 'train/checkpoint_00000000.pt'], root / 'analysis')
        for row in rows:
            self.assertFalse(row['valid'])
            self.assertIn('synthetic failure', row['error'])
            self.assertNotIn('h1_total_finite_persistence', row)

    def test_resume_equivalence_and_analysis_isolation(self):
        with contextlib.redirect_stdout(io.StringIO()):
            root = Path('outputs') / ('study_test_' + uuid.uuid4().hex)
            full, full_history = run(self.config, root / 'full')
            run(self.config, root / 'partial', stop_at=2)
            resumed, history = run(self.config, root / 'partial', resume=root / 'partial/checkpoint_00000002.pt')
            self.assertEqual(history, full_history)
            for key, value in full.state_dict().items():
                self.assertTrue(torch.equal(value, resumed.state_dict()[key]), key)
            left = load_checkpoint(root / 'full/checkpoint_00000006.pt')
            right = load_checkpoint(root / 'partial/checkpoint_00000006.pt')
            for key, state in left['optimizer']['state'].items():
                for name, value in state.items():
                    self.assertTrue(torch.equal(value, right['optimizer']['state'][key][name]))
            before = rng_state()
            rows, metadata = analyze([root / 'full/checkpoint_00000000.pt', root / 'full/checkpoint_00000006.pt'], root / 'analysis')
            after = rng_state()
            self.assertTrue(torch.equal(before['torch'], after['torch']))
            self.assertEqual(before['python'], after['python'])
            np.testing.assert_array_equal(before['numpy'][1], after['numpy'][1])
            self.assertTrue(all(r['valid'] for r in rows))
            self.assertEqual(rows[-1]['spacing_steps'], 6)
            self.assertTrue((root / 'analysis/aligned_metrics.png').is_file())
            arrays = np.load(root / 'analysis/representations_00000006.npz')
            self.assertEqual(int(arrays['token_position']), 2)
            self.assertEqual(arrays['post_block_residual'].shape, (16, 16))
            # Analysis between interruption and resume also leaves the resulting trajectory unchanged.
            run(self.config, root / 'analyzed_resume', resume=root / 'partial/checkpoint_00000002.pt')
            third = load_checkpoint(root / 'analyzed_resume/checkpoint_00000006.pt')
            for key, value in left['model'].items():
                self.assertTrue(torch.equal(value, third['model'][key]))


if __name__ == '__main__':
    unittest.main()
