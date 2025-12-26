import unittest
import tempfile
import os
import numpy as np
from trading_sentiment_analysis.embeddings.glove import GloVeEmbeddings
from trading_sentiment_analysis.embeddings.embedding_feature import (
    extract_features_glove,
    extract_features_glove_batch,
    compute_coverage
)


class EmbeddingFeatureTestCase(unittest.TestCase):
    def setUp(self):
        """Create mock GloVe file."""
        self.temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt')

        words = ['stock', 'price', 'rose', 'fell', 'market']
        for i, word in enumerate(words):
            vec = [str((i+1) * 0.1 * j) for j in range(100)]
            self.temp_file.write(f"{word} {' '.join(vec)}\n")

        self.temp_file.close()
        self.glove = GloVeEmbeddings(self.temp_file.name)

    def tearDown(self):
        os.unlink(self.temp_file.name)

    def test_extract_features_normal(self):
        """Test feature extraction with known words."""
        features = extract_features_glove('Stock prices rose', self.glove)

        self.assertIsNotNone(features)
        self.assertEqual(features.shape, (1, 100))
        self.assertEqual(features.dtype, np.float32)
        self.assertFalse(np.all(features == 0))

    def test_extract_features_empty(self):
        """Test empty headline."""
        features = extract_features_glove('', self.glove)

        self.assertIsNone(features)

    def test_extract_features_unknown_words(self):
        """Test headline with all unknown words."""
        features = extract_features_glove('xyzabc qwerty unknown', self.glove)

        self.assertIsNone(features)

    def test_extract_features_partial_matches(self):
        """Test headline with some unknown words."""
        features = extract_features_glove('stock unknown xyzabc', self.glove)

        self.assertIsNotNone(features)
        self.assertEqual(features.shape, (1, 100))
        self.assertFalse(np.all(features == 0))

    def test_extract_features_batch(self):
        """Test batch extraction."""
        headlines = ['stock rose', 'price fell', 'market']
        features, excluded = extract_features_glove_batch(headlines, self.glove)

        self.assertEqual(features.shape, (3, 100))
        self.assertEqual(features.dtype, np.float32)
        self.assertEqual(len(excluded), 0)

    def test_extract_features_batch_with_exclusions(self):
        """Test batch extraction with zero-match headlines."""
        headlines = ['stock rose', 'unknown xyzabc', 'market']
        features, excluded = extract_features_glove_batch(headlines, self.glove)

        self.assertEqual(features.shape, (2, 100))
        self.assertEqual(excluded, [1])

    def test_compute_coverage_full(self):
        """Test coverage computation with all words matched."""
        matched, total, ratio = compute_coverage('stock price market', self.glove)

        self.assertEqual(total, 3)
        self.assertEqual(matched, 3)
        self.assertAlmostEqual(ratio, 1.0, places=2)

    def test_compute_coverage_partial(self):
        """Test coverage computation with partial matches."""
        matched, total, ratio = compute_coverage('stock price unknown', self.glove)

        self.assertEqual(total, 3)
        self.assertEqual(matched, 2)
        self.assertAlmostEqual(ratio, 2/3, places=2)

    def test_compute_coverage_zero(self):
        """Test coverage computation with no matches."""
        matched, total, ratio = compute_coverage('unknown xyzabc qwerty', self.glove)

        self.assertEqual(total, 3)
        self.assertEqual(matched, 0)
        self.assertAlmostEqual(ratio, 0.0, places=2)


if __name__ == '__main__':
    unittest.main()
