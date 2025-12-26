# Data Model: Word Embeddings with GloVe Vectors

**Phase**: 1 (Design & Contracts)
**Date**: 2025-12-25
**Purpose**: Define entities, data structures, and relationships

## Entity Definitions

### 1. GloVeEmbeddings

**Purpose**: Manages pre-trained GloVe word vectors with vocabulary filtering and efficient lookup.

**Attributes**:
| Attribute | Type | Description | Constraints |
|-----------|------|-------------|-------------|
| `embeddings` | `Dict[str, np.ndarray]` | Word → vector mapping | Keys lowercase, values shape (100,) float32 |
| `dimension` | `int` | Embedding dimensionality | Fixed at 100 |
| `vocabulary` | `Set[str]` | Loaded words (lowercase) | Subset of 400k GloVe vocab |
| `file_path` | `str` | Path to GloVe .txt file | Must exist, readable |

**Methods**:
- `__init__(file_path: str, vocab: Optional[Set[str]] = None)` → None
  - Loads GloVe embeddings from file
  - Filters to vocab if provided, else loads all
  - Raises: `FileNotFoundError`, `ValueError` (invalid format)

- `get_vector(word: str)` → `Optional[np.ndarray]`
  - Returns 100d vector for word (lowercase normalized)
  - Returns None if word not in vocabulary
  - Never raises (graceful unknown word handling)

- `has_word(word: str)` → `bool`
  - Checks if word exists in loaded vocabulary
  - Lowercase normalized

- `vocabulary_size()` → `int`
  - Returns number of loaded words

**Invariants**:
- All keys in `embeddings` are lowercase
- All vectors are shape (100,) and dtype float32
- `dimension` always equals 100 (GloVe 100d constraint)
- `vocabulary` and `embeddings.keys()` are synchronized

**State Diagram**:
```
[Uninitialized] --__init__()--> [Loading] --success--> [Ready]
                                    |
                                    +--error--> [Error] (raises exception)

[Ready] --get_vector()--> [Ready] (stateless lookups)
```

---

### 2. EmbeddingFeature

**Purpose**: Represents a headline converted to a mean embedding vector for model input.

**Attributes**:
| Attribute | Type | Description | Constraints |
|-----------|------|-------------|-------------|
| `vector` | `np.ndarray` | Mean embedding (100d) | Shape (100,), float32 |
| `matched_words` | `int` | Count of words with embeddings | ≥ 0 |
| `total_words` | `int` | Total words in headline | ≥ 0 |
| `coverage` | `float` | Fraction of words matched | Range [0.0, 1.0] |

**Derived Properties**:
- `coverage = matched_words / total_words` (if total_words > 0, else 0.0)
- `is_valid = matched_words > 0` (at least one word had embedding)

**Construction**:
Created by `extract_features_glove()` function, not directly instantiated.

**Lifecycle**:
1. Headline text → tokenization
2. Tokens → embedding lookup
3. Found vectors → mean pooling
4. Result → EmbeddingFeature

**Validation Rules**:
- If `total_words == 0`: Empty headline → zero vector
- If `matched_words == 0`: All unknown words → zero vector (warn user)
- If `matched_words > 0`: Valid feature (normal case)

---

### 3. HeadlineEmbedding (Processing Unit)

**Purpose**: Intermediate representation during feature extraction (not a persistent entity).

**Attributes**:
| Attribute | Type | Description |
|-----------|------|-------------|
| `headline` | `str` | Original headline text |
| `tokens` | `List[str]` | Preprocessed words (lowercase, no stopwords) |
| `vectors` | `List[np.ndarray]` | Embeddings for matched tokens |
| `mean_vector` | `np.ndarray` | Averaged embedding (100d) |

**Transformation Flow**:
```
headline (str)
  ↓ process_text_to_words()
tokens (List[str])
  ↓ GloVeEmbeddings.get_vector() for each token
vectors (List[np.ndarray]) - filtered for non-None
  ↓ np.mean(vectors, axis=0) if len(vectors) > 0 else np.zeros(100)
mean_vector (np.ndarray)
```

---

### 4. TrainingData (Extended)

**Purpose**: Existing entity, extended to support embedding features.

**New Attributes**:
| Attribute | Type | Description |
|-----------|------|-------------|
| `embedding_features` | `np.ndarray` | Shape (N, 100) embedding features | Optional, alternative to TF-IDF features |

**Relationships**:
- **Existing**: `headlines` → `tfidf_features` (3D: bias, pos, neg)
- **New**: `headlines` → `embedding_features` (100D mean vectors)
- **Mutual exclusion**: Use TF-IDF OR embeddings, not both simultaneously

**Training Modes**:
1. **TF-IDF mode** (existing): `extract_features_idf()` → shape (N, 3)
2. **Embedding mode** (new): `extract_features_glove()` → shape (N, 100)

---

### 5. RegressionModel (Extended)

**Purpose**: Existing logistic regression model, extended to support 100D embedding input.

**Modified Attributes**:
| Attribute | Type | Description | Change |
|-----------|------|-------------|--------|
| `weights` | `np.ndarray` | Model parameters | Shape (3, 1) → **(100, 1)** for embeddings |
| `mean` | `np.ndarray` | Normalization mean | Shape (2,) → **(100,)** for embeddings |
| `var` | `np.ndarray` | Normalization variance | Shape (2,) → **(100,)** for embeddings |

**Initialization**:
- **TF-IDF mode**: `shape=(3, 1)` (bias + pos + neg)
- **Embedding mode**: `shape=(100, 1)` (no bias, all embedding dims)

**Impact**:
- Same training algorithm (gradient descent)
- Same prediction logic (sigmoid)
- Only input dimension changes (3 → 100)

---

## Data Relationships

### Entity Relationship Diagram

```
┌─────────────────────┐
│  GloVeEmbeddings    │
│  - embeddings: Dict │
│  - dimension: 100   │
└──────────┬──────────┘
           │ uses
           ↓
┌─────────────────────────┐
│ extract_features_glove()│ (function)
│ Input: headline (str)   │
│ Output: np.ndarray(100) │
└──────────┬──────────────┘
           │ produces
           ↓
┌─────────────────────┐       ┌──────────────────┐
│ EmbeddingFeature    │------>│ TrainingData     │
│ - vector: (100,)    │       │ - features: (N,100) │
│ - coverage: float   │       └────────┬─────────┘
└─────────────────────┘                │ trains
                                       ↓
                            ┌──────────────────┐
                            │ RegressionModel  │
                            │ - weights: (100,1)│
                            │ - mean: (100,)   │
                            │ - var: (100,)    │
                            └──────────────────┘
```

### Data Flow

**Training Pipeline**:
```
1. Load GloVe embeddings
   Input: glove.6B.100d.txt (~350MB)
   Output: GloVeEmbeddings (4MB in memory, 10k vocab)

2. Extract features
   Input: headlines (List[str]), GloVeEmbeddings
   Process: For each headline → tokens → lookup → mean
   Output: features (N, 100) np.ndarray

3. Normalize features
   Input: features (N, 100)
   Output: normalized (N, 100), mean (100,), var (100,)

4. Train model
   Input: normalized features, labels
   Output: weights (100, 1)

5. Save artifacts
   - embedding_model_weights.npy: weights (100, 1)
   - embedding_mean_var.npy: {mean: (100,), var: (100,)}
```

**Prediction Pipeline**:
```
1. Load artifacts
   - GloVeEmbeddings from .txt
   - Model weights, mean, var from .npy

2. Extract features
   Input: new headline
   Output: feature vector (100,)

3. Normalize
   Input: feature vector, saved mean/var
   Output: normalized vector (100,)

4. Predict
   Input: normalized vector, weights
   Output: probability (sigmoid)
```

---

## Data Validation

### GloVe File Format
```
# Expected format (space-separated)
the 0.418 0.24968 -0.41242 ... (100 dimensions)
of -0.15164 0.25341 0.29135 ...
and -0.26818 0.14346 0.39348 ...

# Validation rules:
- Each line: word + 100 floats
- Separator: single space
- Encoding: UTF-8
- Words: lowercase (normalized during load)
```

### Feature Vector Validation
```python
def validate_embedding_feature(vector: np.ndarray) -> bool:
    """Validate embedding feature vector."""
    assert vector.shape == (100,), f"Wrong shape: {vector.shape}"
    assert vector.dtype == np.float32, f"Wrong dtype: {vector.dtype}"
    assert not np.isnan(vector).any(), "Contains NaN"
    assert not np.isinf(vector).any(), "Contains Inf"
    return True
```

### Coverage Thresholds
- **Good coverage**: ≥80% words matched (typical for financial news)
- **Acceptable**: 50-80% matched (some domain-specific terms)
- **Poor**: <50% matched (warn user, results may be unreliable)

---

## Storage Format

### Serialization Strategy

**GloVe Embeddings** (read-only):
- **Format**: Plain text (.txt)
- **Location**: `data/embeddings/glove.6B.100d.txt`
- **Size**: ~350MB on disk, ~4MB loaded (filtered)
- **Persistence**: Manual download (not in git)

**Model Artifacts**:
- **embedding_model_weights.npy**: `np.save(weights)` - shape (100, 1)
- **embedding_mean_var.npy**: `np.save({mean, var})` - dict
- **Format**: NumPy binary (.npy)
- **Location**: Repository root (alongside TF-IDF artifacts)

**Naming Convention**:
- TF-IDF: `model_weights.npy`, `mean_var.npy`
- Embeddings: `embedding_model_weights.npy`, `embedding_mean_var.npy`
- Separate files prevent confusion, enable comparison

---

## Memory Footprint

### Runtime Memory Usage

| Component | Size | Notes |
|-----------|------|-------|
| GloVe embeddings (10k vocab) | ~4 MB | 10k × 100 × 4 bytes |
| GloVe embeddings (400k full) | ~160 MB | 400k × 100 × 4 bytes |
| Training features (10k samples) | ~4 MB | 10k × 100 × 4 bytes |
| Model weights | ~400 bytes | 100 × 1 × 4 bytes |
| Normalization stats | ~800 bytes | 100 + 100 floats |

**Total (filtered vocab)**: ~8-12 MB (minimal)
**Total (full vocab)**: ~165 MB (acceptable)

**Recommendation**: Use filtered vocab for training, full vocab only if needed for deployment.

---

## Backward Compatibility

### Coexistence with TF-IDF

**Design principle**: Embeddings are additive, not replacement.

**File separation**:
- TF-IDF uses: `model_weights.npy`, `frequencies.npy`, `idf_scores.npy`
- Embeddings use: `embedding_model_weights.npy`, `glove_frequencies.npy`
- No conflicts, both can exist simultaneously

**CLI flags**:
- Default (TF-IDF): `poetry run train --tfidf`
- New (Embeddings): `poetry run train --embeddings`
- Mutual exclusion enforced in argument parser

**Model class compatibility**:
- `RegressionModel.__init__(shape=(3, 1))` for TF-IDF
- `RegressionModel.__init__(shape=(100, 1))` for embeddings
- Same class, different initialization

---

## Edge Cases & Error Handling

### Unknown Words
- **Scenario**: Headline contains words not in GloVe (e.g., "COVID-19", "cryptocurrency")
- **Handling**: Skip unknown words, compute mean from matched words only
- **Fallback**: If zero matches, return zero vector (warn user)

### Empty Headlines
- **Scenario**: Headline is empty string or only stopwords
- **Handling**: Return zero vector (100,)
- **Validation**: Flag as low coverage (0.0)

### Corrupted GloVe File
- **Scenario**: File truncated, wrong format, encoding issues
- **Handling**: Raise `ValueError` during load with clear error message
- **Recovery**: User must re-download GloVe file

### Memory Pressure
- **Scenario**: System has <500MB free memory
- **Handling**: Use vocabulary filtering (auto-detect or user flag)
- **Fallback**: Lazy loading not implemented (would require redesign)

---

## Testing Strategy

### Unit Tests

**GloVeEmbeddings**:
- Load small mock .txt file (10 words)
- Lookup existing/non-existing words
- Vocabulary filtering
- Error handling (missing file, invalid format)

**Feature Extraction**:
- Normal case: "Stock prices rose sharply" → 100d vector
- Empty headline: "" → zero vector
- All unknown words: "xyzabc qwerty" → zero vector
- Partial matches: "Apple announced iPhone xyzabc" → mean of 3 vectors

**Integration**:
- End-to-end: Load embeddings → extract features → train model → predict
- Comparison: Verify embeddings vs TF-IDF produce different but valid results

### Test Data

**Mock GloVe file** (for tests):
```
the 0.1 0.2 ... (100 values)
stock 0.5 -0.3 ...
price -0.2 0.4 ...
rose 0.3 0.1 ...
fell -0.4 -0.2 ...
```

**Test headlines**:
- "Stock prices rose" → Should match all 3 words
- "Unknown words only" → Should return zero vector
- "" → Should handle empty gracefully

---

## Next Steps

With data model defined, proceed to:
1. `contracts/`: Specify exact class interfaces and method signatures
2. `quickstart.md`: Step-by-step implementation guide
3. Constitution re-check: Verify design compliance
