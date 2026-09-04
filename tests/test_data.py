import unittest

import numpy as np

from grokking.data import ModularArithmeticDataset


class DatasetTests(unittest.TestCase):
    def test_train_and_validation_are_complementary_and_deterministic(self):
        train = ModularArithmeticDataset(p=5, c=2, r=4, train=True, train_fraction=0.25, seed=7)
        validation = ModularArithmeticDataset(p=5, c=2, r=4, train=False, train_fraction=0.25, seed=7)
        repeated = ModularArithmeticDataset(p=5, c=2, r=4, train=True, train_fraction=0.25, seed=7)

        self.assertEqual(len(train), 4)
        self.assertEqual(len(validation), 12)
        np.testing.assert_array_equal(train.data, repeated.data)
        self.assertTrue({tuple(row) for row in train.data}.isdisjoint(
            {tuple(row) for row in validation.data}
        ))

    def test_dataset_returns_expected_tokens_and_target(self):
        dataset = ModularArithmeticDataset(p=7, c=2, r=3, train=True, train_fraction=1, seed=1)
        inputs, target = dataset[0]
        a, b, exponent, modulus = inputs.tolist()
        self.assertEqual(exponent, 2)
        self.assertEqual(modulus, 7)
        self.assertEqual(target.item(), (a**2 + b**2) % 7)
