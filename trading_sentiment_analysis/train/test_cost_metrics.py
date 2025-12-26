import unittest
import numpy as np
from trading_sentiment_analysis.train.cost import cross_entropy_loss, calculate_metrics


class CostMetricsTestCase(unittest.TestCase):
    def test_cross_entropy_loss(self):
        """Test binary cross-entropy loss calculation."""
        y_true = np.array([[1], [0], [1], [0]])
        y_pred = np.array([[0.9], [0.1], [0.8], [0.2]])

        loss = cross_entropy_loss(y_true, y_pred)

        self.assertIsInstance(float(loss), float)
        self.assertGreater(loss, 0)
        self.assertLess(loss, 1)

    def test_cross_entropy_loss_perfect(self):
        """Test loss with perfect predictions."""
        y_true = np.array([[1], [0], [1], [0]])
        y_pred = np.array([[1.0], [0.0], [1.0], [0.0]])

        loss = cross_entropy_loss(y_true, y_pred)

        self.assertAlmostEqual(float(loss), 0.0, places=5)

    def test_calculate_metrics_perfect(self):
        """Test metrics with perfect predictions."""
        y_true = np.array([1, 0, 1, 0])
        y_pred = np.array([[1.0], [0.0], [1.0], [0.0]])

        metrics = calculate_metrics(y_true, y_pred)

        self.assertAlmostEqual(metrics['accuracy'], 1.0, places=2)
        self.assertAlmostEqual(metrics['precision'], 1.0, places=2)
        self.assertAlmostEqual(metrics['recall'], 1.0, places=2)
        self.assertAlmostEqual(metrics['f1'], 1.0, places=2)
        self.assertAlmostEqual(metrics['loss'], 0.0, places=2)

    def test_calculate_metrics_mixed(self):
        """Test metrics with mixed predictions."""
        y_true = np.array([1, 0, 1, 0, 1, 0])
        y_pred = np.array([[0.9], [0.1], [0.6], [0.4], [0.8], [0.2]])

        metrics = calculate_metrics(y_true, y_pred)

        self.assertGreater(metrics['accuracy'], 0.5)
        self.assertLessEqual(metrics['accuracy'], 1.0)
        self.assertGreaterEqual(metrics['precision'], 0.0)
        self.assertLessEqual(metrics['precision'], 1.0)
        self.assertGreaterEqual(metrics['recall'], 0.0)
        self.assertLessEqual(metrics['recall'], 1.0)
        self.assertGreaterEqual(metrics['f1'], 0.0)
        self.assertLessEqual(metrics['f1'], 1.0)
        self.assertGreater(metrics['loss'], 0.0)

    def test_calculate_metrics_all_keys(self):
        """Test that all expected metrics are returned."""
        y_true = np.array([1, 0, 1, 0])
        y_pred = np.array([[0.9], [0.1], [0.8], [0.2]])

        metrics = calculate_metrics(y_true, y_pred)

        self.assertIn('accuracy', metrics)
        self.assertIn('precision', metrics)
        self.assertIn('recall', metrics)
        self.assertIn('f1', metrics)
        self.assertIn('loss', metrics)


if __name__ == '__main__':
    unittest.main()
