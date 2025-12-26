# Feature Specification: Word Embeddings with GloVe Vectors

**Feature Branch**: `001-word-embeddings-glove`
**Created**: 2025-12-25
**Status**: Draft
**Input**: User description: "Implement word embeddings for the financial news dataset using pre-trained GloVe 100d vectors, mapping each headline to its mean embedding vector, then train the same logistic regression model and compare results to TF-IDF."

## Clarifications

### Session 2025-12-25

- Q: When a headline has zero words with GloVe embeddings (all unknown), what should the system do? → A: Skip the headline entirely and exclude it from the training/test set
- Q: What is the minimum success threshold for the GloVe embedding approach relative to TF-IDF? → A: Must demonstrate significant improvement in loss compared to TF-IDF baseline of 0.6915 (using binary cross-entropy loss L(y,ŷ) = -1/n Σ [y_i log(ŷ_i) + (1-y_i) log(1-ŷ_i)])

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Feature Extraction with Word Embeddings (Priority: P1)

A data scientist wants to represent financial news headlines using semantic word embeddings instead of traditional frequency-based approaches, to capture word meaning and relationships in the vector space.

**Why this priority**: This is the core functionality - converting headlines to embedding vectors. Without this, the feature cannot deliver any value. This provides the foundation for semantic analysis.

**Independent Test**: Can be fully tested by providing a sample headline and verifying that it returns a 100-dimensional mean embedding vector computed from GloVe vectors, and delivers immediate value by enabling semantic representation of text.

**Acceptance Scenarios**:

1. **Given** a financial news headline and GloVe 100d vectors are loaded, **When** the system processes the headline, **Then** it returns a 100-dimensional vector representing the mean of all word embeddings in the headline
2. **Given** a headline containing words not in the GloVe vocabulary, **When** the system processes the headline, **Then** it handles unknown words gracefully and computes the mean from available word vectors
3. **Given** multiple headlines in a dataset, **When** the system extracts features, **Then** each headline is represented as a 100-dimensional vector suitable for model training

---

### User Story 2 - Model Training with Embedding Features (Priority: P2)

A data scientist wants to train a logistic regression sentiment classifier using word embedding features, maintaining the same model architecture as the existing TF-IDF approach for fair comparison.

**Why this priority**: This enables the actual predictive capability using the new feature representation. It depends on P1 (feature extraction) being complete and delivers value by producing a trainable model.

**Independent Test**: Can be fully tested by training a logistic regression model on embedding-based features and verifying that it produces predictions, delivers value by enabling sentiment prediction using semantic embeddings.

**Acceptance Scenarios**:

1. **Given** a training dataset with embedding-based features, **When** the logistic regression model is trained, **Then** the model converges and produces sentiment predictions
2. **Given** trained model parameters, **When** new headlines are processed, **Then** the model outputs sentiment classifications (positive/negative)
3. **Given** the same training data used for TF-IDF model, **When** training with embedding features, **Then** the model uses identical hyperparameters and training procedure to ensure fair comparison

---

### User Story 3 - Performance Comparison with TF-IDF (Priority: P3)

A data scientist wants to compare the performance of the GloVe embedding-based model against the existing TF-IDF baseline, to determine which feature representation produces better sentiment predictions on financial news.

**Why this priority**: This provides insights and justifies the new approach. It depends on both P1 and P2 being complete. Delivers value by quantifying the improvement or trade-offs of the new approach.

**Independent Test**: Can be fully tested by running both models on the same test set and comparing accuracy, precision, recall, and F1 scores, delivers value by providing actionable insights on model performance.

**Acceptance Scenarios**:

1. **Given** both TF-IDF and GloVe embedding models trained on the same data, **When** evaluated on the same test set, **Then** performance metrics (accuracy, precision, recall, F1, binary cross-entropy loss) are calculated and compared side-by-side
2. **Given** performance comparison results, **When** presented to the user, **Then** results clearly show which approach performs better on financial news sentiment classification, with GloVe embeddings demonstrating significant loss improvement from TF-IDF baseline of 0.6915
3. **Given** evaluation results, **When** analyzing performance, **Then** any trade-offs (e.g., training time, memory usage, interpretability) are documented alongside accuracy and loss metrics

---

### Edge Cases

- When a headline contains only words not present in the GloVe vocabulary (all unknown words), the headline is excluded from the training/test set entirely and logged for tracking
- When a headline is empty or contains only stopwords after preprocessing, it is treated as having zero matched words and excluded from the dataset
- What happens if the GloVe vectors file cannot be loaded or is corrupted?
- How does performance compare when headlines vary significantly in length (very short vs. very long)?
- What happens when the dataset contains special characters, numbers, or non-English words that may not have GloVe representations?

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST load pre-trained GloVe 100d word vectors from a standard format file
- **FR-002**: System MUST convert each word in a financial news headline to its corresponding 100-dimensional GloVe vector
- **FR-003**: System MUST compute the mean embedding vector across all words in a headline to produce a single 100-dimensional feature vector
- **FR-004**: System MUST handle words not present in the GloVe vocabulary by skipping them in the mean calculation. If a headline has zero words with GloVe embeddings after preprocessing, the system MUST exclude that headline entirely from the training/test set and log it for tracking
- **FR-005**: System MUST apply the same text preprocessing steps used in the existing TF-IDF pipeline (tokenization, lowercasing, etc.) before looking up word vectors
- **FR-006**: System MUST train a logistic regression classifier using embedding-based features with the same hyperparameters as the existing TF-IDF model
- **FR-007**: System MUST evaluate both GloVe embedding and TF-IDF models on the same test dataset split
- **FR-008**: System MUST calculate and report accuracy, precision, recall, F1 score, and binary cross-entropy loss for both approaches
- **FR-009**: System MUST present a side-by-side comparison showing performance differences between TF-IDF and GloVe embeddings
- **FR-010**: System MUST persist the trained embedding-based model for future predictions

### Key Entities *(include if feature involves data)*

- **GloVe Vectors**: Pre-trained 100-dimensional word embeddings mapping vocabulary words to dense vector representations, capturing semantic relationships
- **Headline Embedding**: A 100-dimensional vector representing the mean of all word embeddings in a financial news headline, serving as the feature representation
- **Sentiment Label**: Binary classification target (positive/negative) associated with each financial news headline, used for training and evaluation
- **Feature Vector**: The processed representation of a headline (either TF-IDF or mean GloVe embedding) used as input to the logistic regression model
- **Performance Metrics**: Quantitative measures (accuracy, precision, recall, F1, binary cross-entropy loss) comparing TF-IDF and GloVe embedding approaches. The TF-IDF baseline loss is 0.6915, and embeddings must demonstrate significant improvement

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Researchers can extract 100-dimensional embedding features from financial news headlines in under 5 seconds per 1000 headlines
- **SC-002**: The embedding-based model demonstrates significant improvement in binary cross-entropy loss compared to TF-IDF baseline of 0.6915, indicating better predictive performance on financial news sentiment classification
- **SC-003**: Performance comparison between TF-IDF and GloVe embeddings is presented with clear metrics showing differences in accuracy, precision, recall, F1 score, and binary cross-entropy loss (formula: L(y,ŷ) = -1/n Σ [y_i log(ŷ_i) + (1-y_i) log(1-ŷ_i)])
- **SC-004**: The system successfully processes headlines containing unknown words without failing, maintaining at least 90% inclusion rate across the dataset (i.e., no more than 10% of headlines excluded due to zero word matches)
- **SC-005**: Documentation clearly shows which approach (TF-IDF or GloVe) performs better, enabling researchers to make informed decisions about feature representation choices
