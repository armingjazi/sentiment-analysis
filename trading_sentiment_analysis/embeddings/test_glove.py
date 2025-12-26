import unittest
import tempfile
import os
import numpy as np
from trading_sentiment_analysis.embeddings.glove import GloVeEmbeddings


class GloVeEmbeddingsTestCase(unittest.TestCase):
    def setUp(self):
        """Create mock GloVe file for testing."""
        self.temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt')

        # Write mock embeddings (word + 100 random floats)
        mock_data = [
            ['the'] + [str(0.1 * i) for i in range(100)],
            ['stock'] + [str(0.2 * i) for i in range(100)],
            ['price'] + [str(0.3 * i) for i in range(100)],
            ['market'] + [str(0.4 * i) for i in range(100)],
        ]

        for line_data in mock_data:
            self.temp_file.write(' '.join(line_data) + '\n')

        self.temp_file.close()

    def tearDown(self):
        """Clean up temp file."""
        os.unlink(self.temp_file.name)

    def test_load_all_words(self):
        """Test loading without vocabulary filter."""
        glove = GloVeEmbeddings(self.temp_file.name)

        self.assertEqual(glove.vocabulary_size(), 4)
        self.assertTrue(glove.has_word('stock'))
        self.assertTrue(glove.has_word('STOCK'))

    def test_load_with_vocab_filter(self):
        """Test vocabulary filtering."""
        vocab = {'stock', 'price'}
        glove = GloVeEmbeddings(self.temp_file.name, vocab)

        self.assertEqual(glove.vocabulary_size(), 2)
        self.assertTrue(glove.has_word('stock'))
        self.assertFalse(glove.has_word('the'))

    def test_get_vector(self):
        """Test vector retrieval."""
        glove = GloVeEmbeddings(self.temp_file.name)

        vec = glove.get_vector('stock')
        self.assertIsNotNone(vec)
        self.assertEqual(vec.shape, (100,))
        self.assertEqual(vec.dtype, np.float32)

        self.assertIsNone(glove.get_vector('unknownword123'))

    def test_case_insensitivity(self):
        """Test case-insensitive lookup."""
        glove = GloVeEmbeddings(self.temp_file.name)

        vec_lower = glove.get_vector('stock')
        vec_upper = glove.get_vector('STOCK')
        vec_mixed = glove.get_vector('StOcK')

        np.testing.assert_array_equal(vec_lower, vec_upper)
        np.testing.assert_array_equal(vec_lower, vec_mixed)


if __name__ == '__main__':
    unittest.main()
