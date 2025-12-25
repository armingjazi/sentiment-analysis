<!--
Sync Impact Report:
Version change: 1.0.0 → 1.1.0
Modified principles: N/A
Added sections:
  - VI. Self-Documenting Code (inline comments principle)
Removed sections: N/A
Templates requiring updates:
  ✅ plan-template.md - No changes needed
  ✅ spec-template.md - No changes needed
  ✅ tasks-template.md - No changes needed
Follow-up TODOs: None

Previous changes (v1.0.0):
  - Initial constitution with 5 core principles
  - Additional Constraints section (Dataset & ML Specific)
  - Development Workflow section (Quality Gates)
-->

# Trading Sentiment Analysis Constitution

## Core Principles

### I. Bare-Bone First (Learning-Oriented Implementation)

This is a learning project focused on understanding machine learning fundamentals. Implementation MUST prioritize writing code from scratch over using pre-built libraries. Library usage is permitted ONLY when implementation complexity would significantly detract from learning objectives or is unreasonably time-consuming.

**Rationale**: The primary goal is educational—understanding how machine learning algorithms work internally, not just how to use them. Every library introduced must be justified against the learning value of implementing the feature manually.

**Examples**:
- MUST implement: Custom gradient descent, basic neural network layers, loss functions, data normalization
- MAY use libraries for: Complex optimization algorithms (L-BFGS), specialized data formats, production-grade infrastructure

### II. Class-Based Modularity (NON-NEGOTIABLE)

All functionality MUST be organized into classes with clear, single responsibilities. Each module MUST be independently importable and testable. Code organization MUST follow object-oriented design principles with proper encapsulation.

**Rationale**: Class-based design enforces modularity, enables testing in isolation, and creates reusable components. This structure is critical for managing complexity in machine learning pipelines where data flows through multiple transformation stages.

**Requirements**:
- Each class has a single, well-defined purpose
- Dependencies are injected, not hardcoded
- Public interfaces are documented and stable
- Internal implementation details are private
- Classes can be instantiated and tested independently

### III. Unit Testing (NON-NEGOTIABLE)

Every class and public method MUST have corresponding unit tests. Tests MUST be written before or alongside implementation. Test coverage is mandatory—no code may be merged without tests.

**Rationale**: Machine learning projects are particularly prone to silent failures where code runs but produces incorrect results. Unit tests validate correctness at each step, catch regressions, and serve as executable documentation.

**Requirements**:
- Test files mirror source structure (tests/unit/ follows trading_sentiment_analysis/)
- Each class has a dedicated test file
- Tests cover normal cases, edge cases, and error conditions
- Tests are isolated (no shared state, no external dependencies)
- Test data is deterministic and minimal

### IV. Caching & Batching When Possible

Data fetching and processing operations SHOULD implement caching to avoid redundant API calls and computation. Large datasets MUST be processed in batches to manage memory and enable resumability.

**Rationale**: Working with financial APIs (Twelve Data) and large Kaggle datasets requires respecting rate limits and managing resources efficiently. Caching reduces API costs and speeds up iteration. Batching prevents out-of-memory errors and allows resuming interrupted processing.

**Implementation**:
- API responses cached locally with appropriate invalidation
- Batch size tunable (currently 1000 items for news labeling)
- Progress saved per batch to enable resume from interruption
- Cache hit/miss logged for transparency

### V. Poetry Dependency Management

All dependencies MUST be managed through Poetry (pyproject.toml). No manual pip installations. Dependency versions MUST be pinned to compatible ranges (^version) to ensure reproducibility.

**Rationale**: Poetry provides deterministic dependency resolution, virtual environment management, and clear separation between production and development dependencies. This ensures the project builds consistently across environments.

**Requirements**:
- All new dependencies added via `poetry add`
- Development dependencies use `poetry add --group dev`
- Lock file (poetry.lock) committed to version control
- Scripts defined in [tool.poetry.scripts] section

### VI. Self-Documenting Code

Code MUST be self-explanatory through clear naming and structure. Inline comments MUST be avoided except in rare cases where they add significant value. The code itself should communicate intent.

**Rationale**: Inline comments often become outdated, create noise, and indicate unclear code. Well-named classes, methods, and variables eliminate the need for most comments. Comments should explain "why" only when the reason is non-obvious, never "what" the code does.

**Requirements**:
- Use descriptive variable names (e.g., `daily_return_percentage` not `drp` or `x`)
- Method names should be verbs describing actions (e.g., `calculate_moving_average()`)
- Class names should be nouns describing entities (e.g., `StockPriceNormalizer`)
- Break complex expressions into named intermediate variables
- Extract complex logic into well-named private methods

**When comments ARE acceptable** (rare cases):
- Non-obvious mathematical formulas with citation to source
- Workarounds for external library bugs with issue tracker links
- Performance optimizations that sacrifice readability (with benchmark justification)
- Domain-specific business rules that require external context

**When comments are NOT acceptable**:
- Explaining what a line of code does (code should be self-evident)
- Describing function parameters (use type hints and docstrings instead)
- Commented-out code (delete it, use version control)
- TODO/FIXME markers without associated issue tracker links

## Additional Constraints

### Dataset & ML Specific

**Data Sources**:
- Kaggle datasets for sentiment data (requires kagglehub)
- Twelve Data API for stock prices (requires API key in .env)
- Data NOT included in repository—users must provide their own credentials

**Processing Requirements**:
- News articles processed in batches of 1000
- Stock price data cached locally after first fetch
- Automatic retry logic for API rate limits
- Label format: next-day bullish/bearish binary classification

**Model Constraints**:
- Focus on interpretability over black-box performance
- Feature engineering must be explicit and traceable
- Model evaluation includes both train/test metrics

## Development Workflow

### Quality Gates

**Before Commit**:
- All unit tests pass
- New code has corresponding tests
- Classes follow single-responsibility principle
- Dependencies justified (bare-bone first check)
- Code is self-documenting with no unnecessary inline comments

**Code Review Focus**:
- Is this implemented manually when it should be?
- Are classes modular and independently testable?
- Do tests cover edge cases and failure modes?
- Is caching/batching applied where appropriate?
- Are inline comments justified (only rare, high-value cases)?
- Do variable/method/class names clearly express intent?

**Integration Testing** (when applicable):
- End-to-end pipeline tests for multi-stage workflows
- Contract tests for external APIs (Twelve Data, Kaggle)
- Data validation tests (schema, ranges, consistency)

## Governance

This constitution supersedes all other practices. Any violation of NON-NEGOTIABLE principles requires documented approval and a migration plan to eventual compliance.

**Amendment Process**:
- Proposed changes must document rationale and impact
- Breaking changes require MAJOR version bump
- New principles/sections require MINOR version bump
- Clarifications/wording require PATCH version bump

**Compliance Review**:
- All PRs must verify compliance with this constitution
- Constitution Check section in plan-template.md enforces gate reviews
- Complexity violations must be explicitly justified in plan.md

**Version**: 1.1.0 | **Ratified**: 2025-12-25 | **Last Amended**: 2025-12-25
