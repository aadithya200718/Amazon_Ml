# Business Entity Resolution — Amazon ML Challenge 2026

## Overview
This package implements an end-to-end Machine Learning pipeline for large-scale Business Entity Resolution (ER) across heterogeneous, noisy data sources (Source 1, Source 2, Source 3).

### Key Pipeline Stages:
1. **Normalization & Canonicalization (`src/normalize.py`)**:
   - NFKD Unicode normalization handling transliterated, accented, and non-Latin texts.
   - Comprehensive legal suffix stripping across US, India, and France (`Inc`, `Corp`, `LLC`, `Pvt Ltd`, `SARL`, `SASU`, `SCI`, etc.).
   - Landmark and generic address component normalization.
2. **Multi-Strategy Inverted-Index Blocking (`src/blocking.py`)**:
   - Ultra-fast Polars-based vectorized key generation running on native multithreaded engine.
   - Multi-tier blocking keys: compressed alphanumeric names, bigrams, name prefix + house numbers, address numbers + street words, and locality tokens.
   - Frequency-bounded inverted index with pruning to prevent Cartesian product blow-ups.
3. **Pairwise Feature Engineering (`src/features.py`)**:
   - High-performance C++/SIMD RapidFuzz string similarities (ratio, token sort ratio, token set ratio, partial ratio).
   - Address numeric overlap and street number verification.
   - Relative candidate pool signals and length disparity metrics.
4. **Calibrated GBDT Matching (`src/model.py`)**:
   - LightGBM binary classifier trained on hard negatives derived directly from the blocking distribution.
5. **Collective Many-to-One Resolution (`src/postprocess.py`)**:
   - Greedy assignment enforcing the domain constraint that each target record describes at most one real-world business entity.
   - Threshold calibrated specifically to maximize the macro F_0.5 metric (precision-weighted 2x over recall).

---

## Environment Setup

Create and activate a virtual environment (Python 3.10+ recommended), then install the pinned dependencies:

```bash
pip install -r requirements.txt
```

Core dependencies:
- `polars==1.37.1`
- `pandas==2.2.3`
- `scikit-learn==1.7.2`
- `lightgbm==4.6.0`
- `rapidfuzz==3.14.5`
- `numpy==2.2.6`

---

## Reproduction Instructions

### 1. Training the Model
To train the LightGBM classifier on training pairs:

```bash
python -m src.train --train-dir path/to/dataset/train
```
This saves the trained model to `models/lgbm_matcher.pkl`.

### 2. Running Inference on Test Dataset
To run the full end-to-end pipeline and generate both `matching_results.tsv` and `candidate_pairs.tsv`:

```bash
python -m src.predict --test-dir path/to/dataset/test --output-dir ../../output
```

Both outputs will be generated:
- `output/matching_results.tsv`: Scored final matches (one row per test S1 entity).
- `output/candidate_pairs.tsv`: Candidate pairs from blocking (matches ⊆ candidates).

### 3. Validating Outputs
Run the official challenge validator:

```bash
python path/to/validate_submission.py \
    --matching ../../output/matching_results.tsv \
    --candidate ../../output/candidate_pairs.tsv \
    --test-dir path/to/dataset/test
```
