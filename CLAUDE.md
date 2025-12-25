# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Trading sentiment analysis system that predicts stock price movements by training machine learning models on financial news headlines. The pipeline downloads financial news, labels it based on subsequent stock price movements, and trains logistic regression classifiers using different feature extraction methods (TF-IDF, word embeddings).

## Development Commands

### Environment Setup
```bash
poetry install
```

### Data Pipeline (Run in Order)

1. **Download financial news data**:
   ```bash
   poetry run download
   ```

2. **Label news with stock movements** (batched processing, 1000 items per batch):
   ```bash
   poetry run label <starting_batch>
   # Example: poetry run label 0
   # Resumes from specific batch if interrupted
   ```

3. **Clean and merge labeled batches**:
   ```bash
   poetry run clean [output_file]
   # Default: data/labeled/cleaned_news.csv
   ```

4. **Train sentiment model**:
   ```bash
   # Basic training (without TF-IDF)
   poetry run train [data_file]

   # With TF-IDF features
   poetry run train --tfidf

   # Custom hyperparameters
   poetry run train --tfidf --batch_size 1024 --learning_rate 5e-3 --iterations 1000

   # Specify output files
   poetry run train --model_file model.npy --mean_var_file stats.npy --frequencies freq.npy --idf_scores idf.npy
   ```

5. **Make predictions**:
   ```bash
   poetry run predict
   ```

### Testing

Tests use Python's built-in `unittest` framework (pytest not installed):

```bash
# Run all tests
python -m unittest discover -s trading_sentiment_analysis -p "test_*.py"

# Run specific test file
python -m unittest trading_sentiment_analysis.train.test_feature

# Run specific test case
python -m unittest trading_sentiment_analysis.train.test_feature.FeatureTestCase.test_feature_extraction
```

## Architecture

### Pipeline Flow

```
Download (Kaggle) → Label (Stock API) → Clean (Merge) → Train (Feature Extraction + Model) → Predict
```

### Module Structure

- **`label/`**: Downloads financial news from Kaggle and labels headlines based on stock price movements
  - Batched processing (1000 items/batch) for memory efficiency
  - Uses Twelve Data API to fetch stock prices
  - Labels: 1 (positive >1% movement), 0 (negative <-1%), 2 (neutral)
  - Caches stock data locally to avoid redundant API calls

- **`clean/`**: Merges labeled batches and removes rows with missing labels
  - Concatenates all `news_batch_*.csv` files from `data/labeled/`
  - Drops unnamed index columns and NaN labels

- **`train/`**: Feature extraction and model training
  - **Feature extraction**: Converts headlines to numerical vectors
    - `extract_features()`: Basic frequency-based (word, label) → positive/negative scores
    - `extract_features_idf()`: TF-IDF weighted features
    - Both produce 3D vectors: `[bias=1, positive_score, negative_score]`
  - **Text preprocessing**: `process_text_to_words()` handles tokenization, lowercasing, stopword removal
  - **Model**: Logistic regression with sigmoid activation
    - Batch gradient descent optimizer
    - Feature normalization (mean/variance stored for inference)
    - Outputs saved: weights, mean/var, frequencies, IDF scores

- **`predict/`**: Loads trained model and makes predictions on new headlines

- **`stock_api/`**: Stock price data fetching with caching
  - **Cache system**: Saves downloaded data to `data/stock/{ticker}.csv`
  - **404 tracking**: Marks failed symbols with `.404` sentinel files to avoid retries
  - **Retry logic**: Handles API rate limits with exponential backoff

### Key Data Flows

**Labeling**: `headline + timestamp` → Stock API → `price_change` → `label (0/1/2)`

**Training**: `headlines + labels` → `build_freqs_docs()` → `(word, label)` frequencies → `extract_features()` → normalized vectors → gradient descent → weights

**Prediction**: `headline` → `extract_features(freqs, idf)` → normalize with saved mean/var → `sigmoid(weights · features)` → probability

### Important Conventions

1. **Batching**: News labeling processes 1000 items at a time to manage memory and allow resumption. Each batch saved as `news_batch_{n}.csv`.

2. **Feature vectors**: Always 3-dimensional `[bias, positive_score, negative_score]` where bias=1. The shape `(3, 1)` is used for weight matrices.

3. **Normalization**: Features (excluding bias term) are normalized using mean/variance from training data. These statistics must be saved and reused during inference.

4. **Stock price labeling**: Uses price before/after news timestamp with configurable threshold (default 1.0%). Handles timezone-aware dates (GMT+4).

5. **TF-IDF**: Optional. When enabled, word frequencies are weighted by inverse document frequency. Empty dict `{}` passed when disabled.

6. **Model persistence**: Training saves 4 artifacts:
   - `model_weights.npy`: Logistic regression weights (3×1)
   - `mean_var.npy`: Dict with 'mean' and 'var' for normalization
   - `frequencies.npy`: Word-label frequency dictionary
   - `idf_scores.npy`: IDF scores (only when --tfidf used)

7. **API credentials**: Twelve Data API key stored in `.env` file (see `.env.example`). Not committed to repo.

8. **Data directories**:
   - `data/labeled/`: Labeled news batches
   - `data/stock/`: Cached stock price data
   - Root: Model artifacts (weights, frequencies, etc.)

## SpecKit Integration

This repository uses SpecKit for feature specification and planning. Feature branches follow the pattern `###-feature-name` with corresponding specs in `specs/###-feature-name/`:

- `spec.md`: Feature specification (what/why, no implementation details)
- `plan.md`: Technical implementation plan
- `tasks.md`: Actionable task list
- `checklists/`: Quality validation checklists

When adding new features, follow the SpecKit workflow: specify → plan → implement.

## Active Technologies
- Python 3.12 (001-word-embeddings-glove)
- Local files (GloVe vectors ~350MB, model weights .npy) (001-word-embeddings-glove)

## Recent Changes
- 001-word-embeddings-glove: Added Python 3.12
