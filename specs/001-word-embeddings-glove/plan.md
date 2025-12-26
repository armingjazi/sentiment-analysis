# Implementation Plan: Word Embeddings with GloVe Vectors

**Branch**: `001-word-embeddings-glove` | **Date**: 2025-12-25 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/001-word-embeddings-glove/spec.md`

## Summary

Implement semantic feature extraction using pre-trained GloVe 100d word embeddings as an alternative to TF-IDF for financial news sentiment classification. Each headline will be converted to a 100-dimensional vector by averaging word embeddings, then fed into the existing logistic regression pipeline. The system must demonstrate significant improvement in binary cross-entropy loss compared to the TF-IDF baseline of 0.6915. Headlines with zero matching words will be excluded from the dataset and logged for tracking.

## Technical Context

**Language/Version**: Python 3.12
**Primary Dependencies**: numpy, pandas, nltk (existing), **no new dependencies**
**Storage**: Local files (GloVe vectors ~350MB, model weights .npy)
**Testing**: unittest (existing framework)
**Target Platform**: Local development (macOS/Linux)
**Project Type**: Single project (Python package)
**Performance Goals**: Process 1000 headlines in <5 seconds, achieve binary cross-entropy loss significantly better than 0.6915
**Constraints**: GloVe 100d specifically (not 50d, 200d, 300d), mean pooling strategy (not max/weighted), ≥90% headline inclusion rate
**Scale/Scope**: ~10k+ headlines in dataset, 400k GloVe vocabulary, 100-dimensional embeddings

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

### Principle I: Bare-Bone First
**Status**: ✅ PASS

- **Manual implementation required**: Word embedding lookup and mean pooling (simple array operations)
- **Library usage justified**:
  - Manual GloVe file parsing (no gensim dependency needed)
  - Existing numpy/pandas for array operations (already in use)
- **Rationale**: Core logic (averaging embeddings, file parsing) will be hand-written. Educational value in understanding embedding formats.

### Principle II: Class-Based Modularity
**Status**: ✅ PASS (pending design)

- **Planned classes**:
  - `GloVeEmbeddings`: Loads and manages word vector lookups
  - `EmbeddingFeatureExtractor`: Converts headlines to mean embedding vectors
  - Integration with existing `RegressionModel` class
- **Single responsibility**: Each class handles one aspect (loading, feature extraction, modeling)
- **Testability**: All classes independently testable with mock embeddings

### Principle III: Unit Testing
**Status**: ✅ PASS (pending implementation)

- **Required tests**:
  - `test_glove_embeddings.py`: Vector loading, word lookup, unknown word handling
  - `test_embedding_feature.py`: Mean pooling, empty headlines, partial matches, zero-match exclusion
  - Integration tests with existing `RegressionModel`
- **Coverage**: Normal cases, edge cases (empty headlines, all unknown words), error conditions

### Principle IV: Caching & Batching
**Status**: ✅ PASS

- **Caching strategy**: Load GloVe vectors once at startup, keep in memory
- **Batching**: Reuse existing training batch size (1024), feature extraction vectorized
- **Rationale**: GloVe loading expensive (~2-3s), must cache. Training batching already implemented.

### Principle V: Poetry Dependency Management
**Status**: ✅ PASS

- **New dependencies**: **NONE** - manual GloVe parsing eliminates gensim
- **Process**: No changes to pyproject.toml needed
- **Justification**: Manual parsing is simple and educational

### Principle VI: Self-Documenting Code
**Status**: ✅ PASS (pending implementation)

- **Naming conventions**: `extract_features_glove()`, `GloVeEmbeddings`, `mean_embedding_vector()`
- **Comments policy**: Only for GloVe file format parsing if non-obvious, mathematical formula (mean pooling) with citation
- **Code clarity**: Extract multi-step operations into named helper methods

### Summary
**Gate Result**: ✅ PASS - All principles satisfied. Proceed to Phase 0 research.

**Open Questions for Research** (resolved in research.md):
1. ✅ Gensim vs manual parsing for GloVe loading → Manual parsing (no dependencies)
2. ✅ GloVe download location and format → Stanford NLP glove.6B.100d.txt
3. ✅ Memory optimization strategies → Vocabulary filtering (4MB vs 160MB)

## Project Structure

### Documentation (this feature)

```text
specs/001-word-embeddings-glove/
├── plan.md              # This file
├── research.md          # Phase 0 output (GloVe loading strategy)
├── data-model.md        # Phase 1 output (embedding entities)
├── quickstart.md        # Phase 1 output (implementation guide)
├── contracts/           # Phase 1 output (module interfaces)
│   ├── glove_embeddings.py      # GloVeEmbeddings class contract
│   ├── embedding_feature.py     # Feature extraction contract
│   └── model_integration.py     # RegressionModel updates
└── tasks.md             # Phase 2 output (NOT created by /speckit.plan)
```

### Source Code (repository root)

```text
trading_sentiment_analysis/
├── embeddings/                    # New module (this feature)
│   │   ├── __init__.py
│   │   ├── glove.py                   # GloVeEmbeddings class
│   │   ├── embedding_feature.py       # EmbeddingFeatureExtractor class
│   │   └── test_glove.py              # Unit tests for embeddings
├── train/
│   ├── feature.py                     # Existing (add extract_features_glove)
│   ├── model.py                       # Existing (add embedding support)
│   ├── test_embedding_feature.py      # New tests for embedding features
│   └── main.py                        # Update to support --embeddings flag
├── data/
│   └── embeddings/                    # GloVe vectors storage
│       └── glove.6B.100d.txt          # Downloaded GloVe file (~350MB)
└── pyproject.toml                     # No changes needed

Root artifacts:
├── embedding_model_weights.npy        # Separate weights for embedding model
├── embedding_mean_var.npy             # Normalization stats for embeddings
└── glove_frequencies.npy              # Word counts for this feature (optional)
```

**Structure Decision**: Single project structure maintained. New `embeddings/` submodule under `train/` follows existing pattern (`frequency.py`, `feature.py` siblings). Embeddings isolated in dedicated module for maintainability and testing.

## Complexity Tracking

> No violations to justify. All constitution principles satisfied.

## Phase 0: Research ✅ COMPLETED

All research questions resolved. See [research.md](./research.md) for detailed findings.

**Key Decisions**:
1. **GloVe Loading**: Manual parsing (no new dependencies, educational value)
2. **GloVe Source**: Stanford NLP glove.6B.100d.txt (official, standard)
3. **Memory Optimization**: Vocabulary filtering (4MB vs 160MB full load)

**Technical Context Updated**:
- Primary Dependencies: numpy, pandas, nltk (no additions)
- Performance: <5s per 1000 headlines (confirmed achievable)
- Memory: 4MB loaded (10k vocab) vs 160MB (400k full)

---

## Phase 1: Design & Contracts ✅ COMPLETED

**Deliverables**:
- [data-model.md](./data-model.md): Entity definitions and relationships
- [contracts/](./contracts/): Class interfaces and method signatures
  - `glove_embeddings.py`: GloVeEmbeddings class contract
  - `embedding_feature.py`: Feature extraction functions
  - `model_integration.py`: RegressionModel extensions
- [quickstart.md](./quickstart.md): Step-by-step implementation guide

**Design Summary**:
- **GloVeEmbeddings**: Loads and manages word vectors (100d)
- **extract_features_glove()**: Converts headlines to mean embeddings
- **Zero-match handling**: Excludes headlines with no word matches, logs for tracking
- **RegressionModel extensions**: Support for 100D input (vs 3D TF-IDF)
- **CLI integration**: `--embeddings` flag alongside existing `--tfidf`
- **Loss tracking**: Binary cross-entropy loss calculation and reporting (baseline: 0.6915)

---

## Phase 2: Constitution Re-Check (Post-Design)

*GATE: Verify design adheres to all principles after detailed planning.*

### Principle I: Bare-Bone First
**Status**: ✅ PASS - CONFIRMED

- **Manual implementations**:
  - GloVe file parsing (~20 lines, no gensim dependency)
  - Mean pooling calculation (numpy.mean)
  - Feature extraction logic
  - Zero-match detection and exclusion
- **Library usage justified**:
  - NumPy: Already in use (array operations)
  - No new dependencies added
- **Educational value**: Understanding embedding file formats, mean pooling, and data filtering

### Principle II: Class-Based Modularity
**Status**: ✅ PASS - CONFIRMED

- **Classes defined** (see contracts/):
  - `GloVeEmbeddings`: Single responsibility (vector management)
  - `EmbeddingFeatureExtractor`: Single responsibility (feature extraction with filtering)
  - Integration with existing `RegressionModel`
- **Independence**: Each class testable in isolation with mocks
- **Interfaces**: Clear public APIs documented in contracts/

### Principle III: Unit Testing
**Status**: ✅ PASS - CONFIRMED

- **Test coverage planned** (see quickstart.md):
  - `test_glove.py`: Loading, lookup, filtering, error handling
  - `test_embedding_feature.py`: Feature extraction, edge cases, batch processing, zero-match exclusion
  - Integration tests: End-to-end pipeline with loss calculation
- **Test cases**: Normal, empty headlines, unknown words, partial matches, zero-match filtering
- **Deterministic**: Mock embeddings for reproducible tests

### Principle IV: Caching & Batching
**Status**: ✅ PASS - CONFIRMED

- **Caching**: GloVe vectors loaded once at startup (2-3s), kept in memory
- **Batching**: Reuses existing batch_size=1024 for training
- **Vectorization**: `extract_features_glove_batch()` for efficient processing
- **Resumability**: Same batch processing as TF-IDF pipeline

### Principle V: Poetry Dependency Management
**Status**: ✅ PASS - CONFIRMED

- **Zero new dependencies**: Manual GloVe parsing eliminates need for gensim
- **Existing dependencies**: numpy, pandas, nltk (already managed via Poetry)
- **No pip installs**: All dependencies in pyproject.toml

### Principle VI: Self-Documenting Code
**Status**: ✅ PASS - CONFIRMED

- **Clear naming** (see contracts/):
  - `GloVeEmbeddings`, `extract_features_glove()`, `compute_coverage()`, `exclude_zero_match_headlines()`
  - `mean_embedding_vector()`, `vocabulary_filtering()`
- **Comment policy**:
  - GloVe file format parsing (rare case: external format)
  - Mean pooling formula (mathematical operation with citation)
  - Binary cross-entropy loss formula (with reference)
  - No inline what/how comments
- **Docstrings**: Type hints and clear descriptions in contracts/

### Final Gate Result
**Status**: ✅ PASS ALL CHECKS

All six principles satisfied after detailed design. No violations, no complexity tracking needed.

**Design Quality**:
- Zero new dependencies (bare-bone principle)
- Modular class structure
- Comprehensive test coverage planned
- Efficient caching/batching strategy
- Self-documenting interfaces
- Explicit zero-match exclusion with logging
- Loss metric tracking and comparison
- Ready for implementation

---

## Implementation Readiness

**Status**: ✅ READY FOR TASKS GENERATION

All planning phases complete:
- [x] Technical Context defined
- [x] Constitution gates passed (initial + post-design)
- [x] Research completed (all unknowns resolved)
- [x] Data model designed
- [x] Contracts specified
- [x] Implementation guide written
- [x] Agent context updated
- [x] Clarifications integrated (zero-match exclusion, loss threshold)

**Clarifications Incorporated**:
1. **Zero-match headlines**: Excluded from training/test set and logged for tracking
2. **Success threshold**: Must demonstrate significant improvement in binary cross-entropy loss vs TF-IDF baseline of 0.6915

**Next Command**: `/speckit.tasks` to generate actionable task list

**Estimated Implementation Time**: 4-5 hours
- Phase 1 (Setup & GloVe download): 30 min
- Phase 2 (GloVeEmbeddings class): 1 hour
- Phase 3 (Feature extraction with filtering): 1.5 hours
- Phase 4 (Model integration with loss tracking): 1 hour
- Phase 5 (CLI updates): 30 min
- Phase 6 (Testing & validation): 30 min
