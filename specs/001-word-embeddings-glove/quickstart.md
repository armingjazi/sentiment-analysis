# Quickstart: Implementing Word Embeddings with GloVe

**Feature**: 001-word-embeddings-glove
**Date**: 2025-12-25
**Purpose**: Step-by-step implementation guide for developers

## Prerequisites

- Python 3.12 installed
- Poetry dependency manager configured
- Existing codebase (TF-IDF pipeline) functional
- ~500MB free disk space (for GloVe download)
- ~50MB free memory (for filtered vocabulary)

## Implementation Roadmap

### Phase 1: Setup & Download (30 minutes)

**Step 1.1: Create directory structure**

```bash
# Create embeddings module
mkdir -p trading_sentiment_analysis/train/embeddings
touch trading_sentiment_analysis/train/embeddings/__init__.py
touch trading_sentiment_analysis/train/embeddings/glove.py
touch trading_sentiment_analysis/train/embeddings/embedding_feature.py
touch trading_sentiment_analysis/train/embeddings/test_glove.py

# Create data directory
mkdir -p data/embeddings
```

**Step 1.2: Download GloVe embeddings**

```bash
# Option A: Direct download (recommended)
cd data/embeddings
wget https://nlp.stanford.edu/data/glove.6B.zip
unzip -p glove.6B.zip glove.6B.100d.txt > glove.6B.100d.txt
rm glove.6B.zip

# Option B: Full download with all dimensions
wget https://nlp.stanford.edu/data/glove.6B.zip
unzip glove.6B.zip
# Keep glove.6B.100d.txt, optionally delete 50d/200d/300d

# Verify download
ls -lh glove.6B.100d.txt
# Expected: ~350MB file
head -n 3 glove.6B.100d.txt
# Should show: word 0.123 0.456 ... (100 numbers per line)
```

**Step 1.3: Update .gitignore**

```bash
# Add to .gitignore (GloVe files are large, don't commit)
echo "data/embeddings/*.txt" >> .gitignore
echo "data/embeddings/*.zip" >> .gitignore
echo "embedding_model_weights.npy" >> .gitignore
echo "embedding_mean_var.npy" >> .gitignore
```

---

### Phase 2: Implement GloVeEmbeddings Class (1 hour)

**Step 2.1: Create glove.py skeleton**

File: `trading_sentiment_analysis/train/embeddings/glove.py`

```python
from typing import Optional, Set
import numpy as np


class GloVeEmbeddings:
    """Manages pre-trained GloVe word vectors."""

    def __init__(self, file_path: str, vocab: Optional[Set[str]] = None) -> None:
        """Load GloVe embeddings with optional vocabulary filtering."""
        if vocab is not None:
            vocab = set(word.lower() for word in vocab)  # Normalize

        self.embeddings = self._load_embeddings(file_path, vocab)
        self.dimension = 100
        self.vocabulary = set(self.embeddings.keys())

    def _load_embeddings(
        self, file_path: str, vocab: Optional[Set[str]]
    ) -> dict[str, np.ndarray]:
        """Load embeddings from GloVe .txt file."""
        embeddings = {}

        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                values = line.split()

                if len(values) != 101:  # word + 100 dimensions
                    print(f"Warning: Line {line_num} has {len(values)} values, expected 101")
                    continue

                word = values[0].lower()  # Normalize to lowercase

                # Filter by vocabulary if provided
                if vocab is not None and word not in vocab:
                    continue

                # Parse vector values
                try:
                    vector = np.array(values[1:], dtype=np.float32)
                    embeddings[word] = vector
                except ValueError as e:
                    print(f"Warning: Could not parse line {line_num}: {e}")
                    continue

        print(f"Loaded {len(embeddings)} word embeddings")
        return embeddings

    def get_vector(self, word: str) -> Optional[np.ndarray]:
        """Get embedding vector for word (case-insensitive)."""
        return self.embeddings.get(word.lower(), None)

    def has_word(self, word: str) -> bool:
        """Check if word exists in vocabulary."""
        return word.lower() in self.embeddings

    def vocabulary_size(self) -> int:
        """Get number of loaded words."""
        return len(self.embeddings)
```

**Step 2.2: Write unit tests**

File: `trading_sentiment_analysis/train/embeddings/test_glove.py`

```python
import unittest
import tempfile
import os
import numpy as np
from trading_sentiment_analysis.train.embeddings.glove import GloVeEmbeddings


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
        self.assertTrue(glove.has_word('STOCK'))  # Case-insensitive

    def test_load_with_vocab_filter(self):
        """Test vocabulary filtering."""
        vocab = {'stock', 'price'}
        glove = GloVeEmbeddings(self.temp_file.name, vocab)

        self.assertEqual(glove.vocabulary_size(), 2)
        self.assertTrue(glove.has_word('stock'))
        self.assertFalse(glove.has_word('the'))  # Filtered out

    def test_get_vector(self):
        """Test vector retrieval."""
        glove = GloVeEmbeddings(self.temp_file.name)

        vec = glove.get_vector('stock')
        self.assertIsNotNone(vec)
        self.assertEqual(vec.shape, (100,))
        self.assertEqual(vec.dtype, np.float32)

        # Unknown word
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
```

**Step 2.3: Run tests**

```bash
# Run GloVe tests
python -m unittest trading_sentiment_analysis.train.embeddings.test_glove

# Expected output:
# ....
# Ran 4 tests in 0.XXXs
# OK
```

---

### Phase 3: Implement Feature Extraction (1.5 hours)

**Step 3.1: Create embedding_feature.py**

File: `trading_sentiment_analysis/train/embeddings/embedding_feature.py`

```python
import numpy as np
from typing import Tuple
from trading_sentiment_analysis.train.embeddings.glove import GloVeEmbeddings
from trading_sentiment_analysis.train.process import process_text_to_words


def extract_features_glove(headline: str, glove: GloVeEmbeddings) -> np.ndarray:
    """Extract mean embedding feature from headline.

    Args:
        headline: News headline text
        glove: Loaded GloVe embeddings

    Returns:
        np.ndarray of shape (1, 100) - mean of word embeddings
    """
    # Preprocess text (reuse existing pipeline)
    words = process_text_to_words(headline)

    # Collect embeddings for matched words
    embeddings = []
    for word in words:
        vec = glove.get_vector(word)
        if vec is not None:
            embeddings.append(vec)

    # Compute mean or return zero vector
    if len(embeddings) > 0:
        mean_vector = np.mean(embeddings, axis=0)
    else:
        mean_vector = np.zeros(100, dtype=np.float32)
        if len(words) > 0:  # Warn if we had words but no matches
            print(f"Warning: No embeddings found for headline: {headline[:50]}...")

    # Add batch dimension
    return mean_vector[np.newaxis, :]


def extract_features_glove_batch(
    headlines: list[str],
    glove: GloVeEmbeddings
) -> np.ndarray:
    """Extract features for multiple headlines.

    Args:
        headlines: List of headline strings
        glove: Loaded GloVe embeddings

    Returns:
        np.ndarray of shape (N, 100) - batch of features
    """
    features = np.zeros((len(headlines), 100), dtype=np.float32)

    for i, headline in enumerate(headlines):
        feature = extract_features_glove(headline, glove)
        features[i, :] = feature

    return features


def compute_coverage(
    headline: str,
    glove: GloVeEmbeddings
) -> Tuple[int, int, float]:
    """Compute vocabulary coverage for headline.

    Args:
        headline: Headline text
        glove: Loaded GloVe embeddings

    Returns:
        (matched_words, total_words, coverage_ratio)
    """
    words = process_text_to_words(headline)
    total_words = len(words)

    if total_words == 0:
        return (0, 0, 0.0)

    matched_words = sum(1 for word in words if glove.has_word(word))
    coverage = matched_words / total_words

    return (matched_words, total_words, coverage)
```

**Step 3.2: Add tests for feature extraction**

File: `trading_sentiment_analysis/train/test_embedding_feature.py`

```python
import unittest
import tempfile
import os
import numpy as np
from trading_sentiment_analysis.train.embeddings.glove import GloVeEmbeddings
from trading_sentiment_analysis.train.embeddings.embedding_feature import (
    extract_features_glove,
    extract_features_glove_batch,
    compute_coverage
)


class EmbeddingFeatureTestCase(unittest.TestCase):
    def setUp(self):
        """Create mock GloVe file."""
        self.temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt')

        # Mock embeddings for testing
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

        self.assertEqual(features.shape, (1, 100))
        self.assertEqual(features.dtype, np.float32)
        self.assertFalse(np.all(features == 0))  # Not zero vector

    def test_extract_features_empty(self):
        """Test empty headline."""
        features = extract_features_glove('', self.glove)

        self.assertEqual(features.shape, (1, 100))
        np.testing.assert_array_equal(features, np.zeros((1, 100), dtype=np.float32))

    def test_extract_features_unknown_words(self):
        """Test headline with all unknown words."""
        features = extract_features_glove('xyzabc qwerty unknown', self.glove)

        self.assertEqual(features.shape, (1, 100))
        np.testing.assert_array_equal(features, np.zeros((1, 100), dtype=np.float32))

    def test_extract_features_batch(self):
        """Test batch extraction."""
        headlines = ['stock rose', 'price fell', 'market']
        features = extract_features_glove_batch(headlines, self.glove)

        self.assertEqual(features.shape, (3, 100))
        self.assertEqual(features.dtype, np.float32)

    def test_compute_coverage(self):
        """Test coverage computation."""
        matched, total, ratio = compute_coverage('stock price unknown', self.glove)

        self.assertEqual(total, 3)
        self.assertEqual(matched, 2)  # stock, price known; unknown not
        self.assertAlmostEqual(ratio, 2/3, places=2)


if __name__ == '__main__':
    unittest.main()
```

**Step 3.3: Run tests**

```bash
python -m unittest trading_sentiment_analysis.train.test_embedding_feature
```

---

### Phase 4: Integrate with RegressionModel (1 hour)

**Step 4.1: Add methods to model.py**

File: `trading_sentiment_analysis/train/model.py` (add new methods)

```python
# Add import at top of file
from trading_sentiment_analysis.train.embeddings.glove import GloVeEmbeddings
from trading_sentiment_analysis.train.embeddings.embedding_feature import extract_features_glove_batch

class RegressionModel:
    # ... existing methods ...

    def train_with_embeddings(
        self,
        train_x: list[str],
        train_y: np.ndarray,
        glove: GloVeEmbeddings,
        learning_rate: float,
        iterations: int,
        batch_size: int = 1024
    ):
        """Train model using embedding features."""
        print("Extracting embedding features from training data...")

        # Extract features for all headlines
        train_input = extract_features_glove_batch(train_x, glove)

        # Normalize features (no bias term for embeddings)
        train_input, self.mean, self.var = normalize(train_input)

        # Labels
        train_target = train_y.reshape(-1, 1)

        print(f"Training input shape: {train_input.shape}")
        print(f"Training label shape: {train_target.shape}")

        # Train using existing optimizer
        costs, weights = batch_gradient_descent(
            train_input, train_target, self.weights,
            learning_rate, iterations, batch_size
        )

        self.weights = weights
        return costs, weights

    def predict_sentiments_with_embeddings(
        self,
        headlines: list[str],
        glove: GloVeEmbeddings
    ):
        """Predict sentiments using embedding features."""
        # Extract features
        features = extract_features_glove_batch(headlines, glove)

        # Normalize
        features = normalize_with_mean_var(features, self.mean, self.var)

        # Predict
        return self.predict(features)

    def save_embeddings_model(self, model_file: str, mean_var_file: str):
        """Save embedding model (separate from TF-IDF)."""
        np.save(model_file, self.weights)
        np.save(mean_var_file, {'mean': self.mean, 'var': self.var})
        print(f"Embedding model saved to {model_file}")

    def load_embeddings_model(self, model_file: str, mean_var_file: str):
        """Load embedding model."""
        self.weights = np.load(model_file, allow_pickle=True)
        mean_var = np.load(mean_var_file, allow_pickle=True).item()
        self.mean = mean_var['mean']
        self.var = mean_var['var']
        print(f"Embedding model loaded from {model_file}")
```

---

### Phase 5: Update CLI (30 minutes)

**Step 5.1: Modify train/main.py**

File: `trading_sentiment_analysis/train/main.py`

```python
# Add imports
from trading_sentiment_analysis.train.embeddings.glove import GloVeEmbeddings

def main():
    parser = argparse.ArgumentParser(description='train model')
    parser.add_argument('data_file', type=str, nargs='?', default='data/labeled/cleaned_news.csv')

    # Existing flags
    parser.add_argument('--tfidf', action='store_true', help='Use TF-IDF for feature extraction')

    # NEW: Embedding flags
    parser.add_argument('--embeddings', action='store_true', help='Use GloVe embeddings for features')
    parser.add_argument('--glove-path', type=str, default='data/embeddings/glove.6B.100d.txt',
                       help='Path to GloVe embeddings file')

    # Training hyperparameters (existing)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--learning_rate', type=float, default=5e-3)
    parser.add_argument('--iterations', type=int, default=1000)

    # Output files
    parser.add_argument('--model_file', type=str, default='model_weights.npy')
    parser.add_argument('--embedding_model_file', type=str, default='embedding_model_weights.npy')

    args = parser.parse_args()

    # Load data (existing)
    train_data, test_data, train_y, test_y = train_test_split(args.data_file)
    train_x = train_data['title'].values
    test_x = test_data['title'].values

    # NEW: Embedding mode
    if args.embeddings:
        print("=== Training with GloVe Embeddings ===")
        print(f"Loading GloVe embeddings from {args.glove_path}...")

        # Load embeddings (optionally filter to training vocab)
        glove = GloVeEmbeddings(args.glove_path)

        # Initialize model with 100D input
        model = RegressionModel(shape=(100, 1))

        # Train
        costs, weights = model.train_with_embeddings(
            train_x, train_y, glove,
            learning_rate=args.learning_rate,
            iterations=args.iterations,
            batch_size=args.batch_size
        )

        # Evaluate
        test_probs = model.predict_sentiments_with_embeddings(test_x, glove)
        test_preds = (test_probs >= 0.5).astype(int)
        accuracy = np.mean(test_preds.flatten() == test_y)

        print(f"Test Accuracy: {accuracy * 100:.2f}%")

        # Save
        model.save_embeddings_model(args.embedding_model_file, 'embedding_mean_var.npy')

    # Existing TF-IDF mode
    elif args.tfidf:
        # ... existing TF-IDF training code ...
        pass

    else:
        print("Please specify --tfidf or --embeddings")
```

---

### Phase 6: Test End-to-End (30 minutes)

**Step 6.1: Verify GloVe file**

```bash
# Check file exists and format is correct
head -n 1 data/embeddings/glove.6B.100d.txt | wc -w
# Expected: 101 (word + 100 dimensions)
```

**Step 6.2: Train embedding model**

```bash
# Train with embeddings
poetry run train --embeddings --iterations 500

# Expected output:
# === Training with GloVe Embeddings ===
# Loading GloVe embeddings from data/embeddings/glove.6B.100d.txt...
# Loaded 400000 word embeddings
# Extracting embedding features from training data...
# Training input shape: (XXXX, 100)
# Training model...
# Test Accuracy: XX.XX%
# Embedding model saved to embedding_model_weights.npy
```

**Step 6.3: Compare with TF-IDF**

```bash
# Train TF-IDF model (if not already trained)
poetry run train --tfidf --iterations 500

# Compare results manually
# - Check accuracy values
# - Both should be > 50% (better than random)
# - Embeddings may be higher (semantic understanding) or lower (less features)
```

---

## Verification Checklist

- [ ] GloVe file downloaded and in `data/embeddings/glove.6B.100d.txt`
- [ ] All unit tests pass (`test_glove.py`, `test_embedding_feature.py`)
- [ ] `GloVeEmbeddings` class loads vectors successfully
- [ ] Feature extraction produces (N, 100) arrays
- [ ] Model trains without errors
- [ ] Test accuracy > 50% (better than random)
- [ ] Artifacts saved: `embedding_model_weights.npy`, `embedding_mean_var.npy`
- [ ] CLI flag `--embeddings` works
- [ ] No regressions in TF-IDF mode

---

## Troubleshooting

**Issue**: `FileNotFoundError: glove.6B.100d.txt`
- **Fix**: Download GloVe file following Step 1.2

**Issue**: `ValueError: Line has wrong number of values`
- **Fix**: Verify GloVe file not corrupted, re-download if needed

**Issue**: Test accuracy < 50%
- **Fix**: Check feature extraction (should not be all zeros), verify normalization applied

**Issue**: Memory error during load
- **Fix**: Use vocabulary filtering: `GloVeEmbeddings(path, vocab=training_vocab)`

**Issue**: Warning "No embeddings found for headline"
- **Fix**: Normal for headlines with uncommon words, acceptable if < 10% of data

---

## Next Steps

After implementation complete:
1. Run comparison between TF-IDF and embeddings
2. Document results in feature spec
3. Optionally: Experiment with other pooling strategies (max, weighted)
4. Optionally: Try other embedding dimensions (50d, 200d)

## Success Criteria

✅ Feature extraction < 5 seconds per 1000 headlines
✅ Model accuracy > 50% (random baseline)
✅ Clear comparison with TF-IDF baseline
✅ All unit tests passing
✅ Documentation complete
