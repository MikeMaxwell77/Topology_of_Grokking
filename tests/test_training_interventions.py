import json
import unittest
import uuid
from pathlib import Path

import torch

from grokking.config import TopologyStudyConfig
from grokking.study import run, load_checkpoint
from grokking.training_interventions import branch


class TrainingInterventionTests(unittest.TestCase):
    def test_zero_strength_matches_uninterrupted_training(self):
        torch.set_num_threads(1)
        root = Path('outputs') / ('timing_test_' + uuid.uuid4().hex)
        config = TopologyStudyConfig(modulus=5, d_model=8, n_heads=2, d_ff=16,
                                     probe_size=5, steps=4, evaluation_interval=1,
                                     checkpoint_interval=2)
        model, _ = run(config, root / 'baseline')
        checkpoint = root / 'baseline/checkpoint_00000002.pt'
        original = checkpoint.read_bytes()
        branch(checkpoint, root / 'branches', duration=1, strength=0,
               clusters=2, repeats=1)
        for arm in ('clean', 'targeted', 'random_0', 'rotated_0'):
            saved = load_checkpoint(root / 'branches' / arm / 'checkpoint_00000004.pt')
            for name, value in model.state_dict().items():
                self.assertTrue(torch.equal(value, saved['model'][name]), name)
            rows = json.loads((root / 'branches' / arm / 'training.json').read_text())
            self.assertEqual({r['step'] for r in rows}, {2, 3, 4})
            self.assertEqual({r['mode'] for r in rows}, {'clean', 'active'})
        self.assertEqual(original, checkpoint.read_bytes())
        branch(checkpoint, root / 'nonzero', duration=1, strength=.5,
               clusters=2, repeats=1)
        clean = load_checkpoint(root / 'nonzero/clean/checkpoint_00000004.pt')
        targeted = load_checkpoint(root / 'nonzero/targeted/checkpoint_00000004.pt')
        self.assertTrue(all(torch.equal(value, clean['model'][name])
                            for name, value in model.state_dict().items()))
        self.assertTrue(any(not torch.equal(value, targeted['model'][name])
                            for name, value in model.state_dict().items()))
        self.assertTrue(all(not r['intervention_on_next_update']
                            for r in targeted['history'] if r['step'] >= 3))
        changes = dict(load_checkpoint(root / 'nonzero/interventions.pt')['conditions'])
        for name in ('random_0', 'rotated_0'):
            self.assertTrue(torch.allclose(changes[name].norm(dim=1),
                                           changes['targeted'].norm(dim=1), atol=1e-7))

    def test_fixed_translation_has_identity_backward(self):
        residual = torch.randn(5, 8, requires_grad=True)
        change = torch.randn_like(residual)
        (residual + change).sum().backward()
        self.assertTrue(torch.equal(residual.grad, torch.ones_like(residual)))
