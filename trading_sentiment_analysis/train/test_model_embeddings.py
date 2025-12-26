import unittest
import tempfile
import os
import numpy as np
from trading_sentiment_analysis.embeddings.glove import GloVeEmbeddings
from trading_sentiment_analysis.train.model import RegressionModel


class ModelEmbeddingsTestCase(unittest.TestCase):
    def setUp(self):
        """Create mock GloVe file and test data."""
        self.temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt')

        words = ['stock', 'price', 'rose', 'fell', 'market', 'high', 'low', 'profit', 'loss', 'gain']
        for i, word in enumerate(words):
            vec = [str((i+1) * 0.1 * j) for j in range(100)]
            self.temp_file.write(f"{word} {' '.join(vec)}\n")

        self.temp_file.close()
        self.glove = GloVeEmbeddings(self.temp_file.name)

        self.train_headlines = [
            'stock price rose high',
            'market fell low',
            'profit gain',
            'loss fell'
        ]
        self.train_labels = np.array([1, 0, 1, 0])

    def tearDown(self):
        os.unlink(self.temp_file.name)

    def test_train_with_embeddings(self):
        """Test training model with embeddings."""
        model = RegressionModel(shape=(100, 1))

        costs, weights = model.train_with_embeddings(
            self.train_headlines,
            self.train_labels,
            self.glove,
            learning_rate=0.01,
            iterations=100,
            batch_size=2
        )

        self.assertEqual(weights.shape, (100, 1))
        self.assertIsNotNone(model.mean)
        self.assertIsNotNone(model.var)
        self.assertEqual(model.mean.shape, (1, 100))
        self.assertEqual(model.var.shape, (1, 100))
        self.assertIsInstance(costs, list)
        self.assertGreater(len(costs), 0)

    def test_predict_sentiments_with_embeddings(self):
        """Test prediction with embeddings."""
        model = RegressionModel(shape=(100, 1))

        model.train_with_embeddings(
            self.train_headlines,
            self.train_labels,
            self.glove,
            learning_rate=0.01,
            iterations=100,
            batch_size=2
        )

        test_headlines = ['stock rose', 'market fell']
        predictions, excluded = model.predict_sentiments_with_embeddings(test_headlines, self.glove)

        self.assertEqual(predictions.shape, (2, 1))
        self.assertEqual(len(excluded), 0)
        self.assertTrue(np.all(predictions >= 0))
        self.assertTrue(np.all(predictions <= 1))

    def test_save_load_embeddings_model(self):
        """Test save and load embedding model."""
        model = RegressionModel(shape=(100, 1))

        model.train_with_embeddings(
            self.train_headlines,
            self.train_labels,
            self.glove,
            learning_rate=0.01,
            iterations=50,
            batch_size=2
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            model_file = os.path.join(tmpdir, 'test_model.npy')
            mean_var_file = os.path.join(tmpdir, 'test_mean_var.npy')

            model.save_embeddings_model(model_file, mean_var_file)

            new_model = RegressionModel(shape=(100, 1))
            new_model.load_embeddings_model(model_file, mean_var_file)

            np.testing.assert_array_equal(new_model.weights, model.weights)
            np.testing.assert_array_equal(new_model.mean, model.mean)
            np.testing.assert_array_equal(new_model.var, model.var)

    def test_train_with_exclusions(self):
        """Test training with zero-match headlines."""
        headlines_with_unknown = self.train_headlines + ['unknown xyzabc qwerty']
        labels_with_unknown = np.append(self.train_labels, 1)

        model = RegressionModel(shape=(100, 1))

        costs, weights = model.train_with_embeddings(
            headlines_with_unknown,
            labels_with_unknown,
            self.glove,
            learning_rate=0.01,
            iterations=50,
            batch_size=2
        )

        self.assertEqual(weights.shape, (100, 1))


if __name__ == '__main__':
    unittest.main()
