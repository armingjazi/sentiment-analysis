import math
import unittest

import numpy as np

from trading_sentiment_analysis.train.cost import cross_entropy_loss, gradient_descent


class CostTestCase(unittest.TestCase):
    def test_cross_entropy(self):
        y_true = np.array([1, 0, 1])
        y_pred = np.array([0.9, 0.1, 0.8])
        expected_cost = -1 / 3 * (math.log(0.9) + math.log(0.8) + math.log(0.9))
        self.assertAlmostEqual(cross_entropy_loss(y_true, y_pred), expected_cost)
    
    def test_cross_entropy_zero(self):
        y_true = np.array([1, 0, 1])
        y_pred = np.array([0.0, 0.0, 0.0])
        
        self.assertGreater(cross_entropy_loss(y_true, y_pred), 20)

    def test_cross_entropy_one(self):
        y_true = np.array([1, 0, 1])
        y_pred = np.array([float(1.0), float(1.0), float(1.0)])
        
        self.assertGreater(cross_entropy_loss(y_true, y_pred), 10)


class GradientDescentTestCase(unittest.TestCase):
    def test_gradient_descent(self):
        inputs = np.array([[1, 2], [3, 4], [5, 6]])
        targets = np.array([[0], [1], [0]])
        weights = np.array([[0.1], [0.2]])
        alpha = 0.01
        iterations = 1000
        
        cost, new_weights = gradient_descent(inputs, targets, weights, alpha, iterations)
        
        self.assertIsInstance(cost, float)
        self.assertEqual(new_weights.shape, (2, 1))
    
    def test_gradient_descent_zero(self):
        inputs = np.array([[1, 2], [3, 4], [5, 6]])
        targets = np.array([[0], [1], [0]])
        weights = np.array([[0.0], [0.0]])
        alpha = 0.01
        iterations = 1000
        
        cost, new_weights = gradient_descent(inputs, targets, weights, alpha, iterations)
        
        self.assertIsInstance(cost, float)
        self.assertEqual(new_weights.shape, (2, 1))

    def test_gradient_descent_values(self):
        inputs = np.array(
                        [
                            [1.00000000e00, 8.34044009e02, 1.44064899e03],
                            [1.00000000e00, 2.28749635e-01, 6.04665145e02],
                            [1.00000000e00, 2.93511782e02, 1.84677190e02],
                            [1.00000000e00, 3.72520423e02, 6.91121454e02],
                            [1.00000000e00, 7.93534948e02, 1.07763347e03],
                            [1.00000000e00, 8.38389029e02, 1.37043900e03],
                            [1.00000000e00, 4.08904499e02, 1.75623487e03],
                            [1.00000000e00, 5.47751864e01, 1.34093502e03],
                            [1.00000000e00, 8.34609605e02, 1.11737966e03],
                            [1.00000000e00, 2.80773877e02, 3.96202978e02],
                        ])
        targets = np.array([[1], [1], [0], [1], [1], [1], [0], [0], [0], [1]])
        weights = np.zeros((3, 1))
        alpha = 1e-8
        iterations = 10
        cost, new_weights = gradient_descent(inputs, targets, weights, alpha, iterations)

        self.assertAlmostEqual(cost, 0.6931471805599453)
        self.assertEqual(new_weights.shape, (3, 1))
        self.assertEqual(new_weights.tolist(), [[1.0000000000000002e-08], [7.638449816175002e-06], [5.907421485000002e-06]])

