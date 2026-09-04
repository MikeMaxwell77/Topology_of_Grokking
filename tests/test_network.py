import unittest
import torch

from grokking.network import TinyTransformer


class NetworkTests(unittest.TestCase):
    def test_forward_and_hidden_state_shapes(self):
        model = TinyTransformer(vocab_size=12, d_model=8, n_heads=2, n_layers=2, d_ff=16)
        inputs = torch.tensor([[1, 2, 1, 7], [2, 3, 1, 7]])
        self.assertEqual(model(inputs).shape, (2, 12))
        self.assertEqual(model.get_hidden_states(inputs, 0).shape, (2, 8))

    def test_hidden_state_rejects_invalid_layer(self):
        model = TinyTransformer(vocab_size=12, d_model=8, n_heads=2, n_layers=1, d_ff=16)
        with self.assertRaises(IndexError):
            model.get_hidden_states(torch.tensor([[1, 2, 1, 7]]), 1)
