import json
import unittest
import uuid
from dataclasses import asdict
from pathlib import Path

import torch

from grokking.config import TopologyStudyConfig
from grokking.rescue import sweep
from grokking.study import StudyTransformer, build_data, load_checkpoint


class RescueTests(unittest.TestCase):
    def test_checkpoint_rescue_and_saved_perturbations(self):
        torch.set_num_threads(1)
        config = TopologyStudyConfig(modulus=7, d_model=8, n_heads=2,
                                     d_ff=16, probe_size=8)
        model = StudyTransformer(config).eval()
        x, _, indices = build_data(config)
        root = Path('outputs') / ('rescue_test_' + uuid.uuid4().hex)
        root.mkdir(parents=True)
        checkpoint = root / 'checkpoint.pt'
        torch.save(dict(config=asdict(config), model=model.state_dict(),
                        indices=indices, step=0), checkpoint)
        original = checkpoint.read_bytes()
        rows = sweep([checkpoint], root / 'rescue', clusters=2,
                     strengths=(0., .5, 1.), repeats=2)
        self.assertEqual(len(rows), 66)
        self.assertEqual({r['mode'] for r in rows}, {'clean', 'disrupted', 'rescued'})
        for row in rows:
            if row['mode'] == 'rescued' or row['strength'] == 0:
                self.assertLess(row['max_activation_error'], 1e-6)
                self.assertLess(row['max_logit_error'], 1e-6)
                self.assertLess(abs(row['loss_difference_from_clean']), 1e-6)
        self.assertTrue(any(r['max_logit_error'] > 1e-7 for r in rows
                            if r['mode'] == 'disrupted' and r['strength'] > 0))
        artifact = load_checkpoint(root / 'rescue/perturbations_00000000.pt')
        with torch.no_grad():
            z = model.representations(x)['post_block_residual']
        for perturbation in artifact['perturbations']:
            torch.testing.assert_close((z + perturbation['delta']) - perturbation['delta'], z)
        self.assertEqual(checkpoint.read_bytes(), original)
        self.assertEqual(json.loads((root / 'rescue/results.json').read_text()), rows)
        self.assertIn('algebraic implementation control', (root / 'rescue/report.md').read_text())


if __name__ == '__main__':
    unittest.main()
