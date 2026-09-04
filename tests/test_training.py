import unittest

import numpy as np
import torch

from grokking.training import classification_residuals, evaluate


class FixedClassifier(torch.nn.Module):
    def forward(self, inputs):
        return inputs.float()


class TrainingTests(unittest.TestCase):
    def test_evaluate_computes_accuracy(self):
        loader = [(torch.tensor([[5, 1], [1, 5], [4, 2]]), torch.tensor([0, 1, 1]))]
        self.assertEqual(evaluate(FixedClassifier(), loader, "cpu"), 2 / 3)

    def test_classification_residuals_uses_one_hot_targets(self):
        outputs = torch.tensor([[0.2, 0.8], [0.7, 0.3]])
        residuals = classification_residuals(outputs, torch.tensor([1, 0]))
        np.testing.assert_allclose(residuals, [[0.2, -0.2], [-0.3, 0.3]], atol=1e-6)
