import unittest
import numpy as np
from trading_sentiment_analysis.train.activation import sigmoid


class SigmoidTestCase(unittest.TestCase):
    def test_sigmoid(self):
        self.assertEqual(sigmoid(0), 0.5)
        self.assertEqual(sigmoid(1), 0.7310585786300049)
        self.assertEqual(sigmoid(-1), 0.2689414213699951)
        self.assertEqual(sigmoid(100), 1.0)
        self.assertEqual(sigmoid(4.92), 0.9927537604041685)

    def test_sigmoid_array(self):

        arr = np.array([0, 1, -1, 100, 4.92])
        expected = np.array([0.5, 0.7310585786300049, 0.2689414213699951, 1.0, 0.9927537604041685])
        np.testing.assert_array_almost_equal(sigmoid(arr), expected)

    def test_sigmoid_large_array(self):
        arr = np.array([0, 1, -1, 100, 4.92] * 1000)
        expected = np.array([0.5, 0.7310585786300049, 0.2689414213699951, 1.0, 0.9927537604041685] * 1000)
        np.testing.assert_array_almost_equal(sigmoid(arr), expected)

    def test_sigmoid_small_input(self):
        arr = np.array([0.0001, 0.0002, 0.0003])
        expected = np.array([0.500025, 0.50005, 0.5000749999999999])
        np.testing.assert_array_almost_equal(sigmoid(arr), expected)

    def test_sigmoid_zero(self):
        arr = np.array([0])
        expected = np.array([0.5])
        np.testing.assert_array_almost_equal(sigmoid(arr), expected)