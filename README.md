# Amazon ML Challenge 2026: Business Entity Resolution

[![Metric](https://img.shields.io/badge/Validation%20Macro%20F0.5-0.7741-brightgreen.svg)]()
[![License](https://img.shields.io/badge/License-Apache%202.0%20%2F%20MIT-blue.svg)]()
[![Model](https://img.shields.io/badge/Model-LightGBM%20GBDT-orange.svg)]()
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)]()

Production Machine Learning solution for the **Amazon ML Challenge 2026: Business Entity Resolution** track. Resolves multi-source fragmented entity records across Source 1 (reference), Source 2, and Source 3 into canonical business entities under a precision-weighted macro $F_{0.5}$ metric.

---

## 🏆 Key Results

- **Validation Macro $F_{0.5}$:** **`0.7741`** at calibrated threshold $\tau = 0.55$.
- **US Blocking Recall:** **`96.83%`** against 6.18M candidates.
- **India Blocking Recall:** **`90.79%`** against 4.13M candidates across diverse scripts.
- **Test Set Coverage:** **1,732,544 / 1,732,544 (100%)** entities evaluated across France, India, and US.
- **Official Validator:** **`PASS`** (0 blocking issues, 0 malformed rows, 100% valid test set IDs).

---

## 📂 Repository Structure

```
├── team_submission/
│   ├── output/
│   │   └── matching_results.tsv      # Final scored entity matches for the leaderboard
│   ├── code/
│   │   └── business_entity_resolution/
│   │       ├── src/
│   │       │   ├── config.py         # Pipeline configuration and paths
│   │       │   ├── normalize.py      # Multi-country suffix & unicode normalization
│   │       │   ├── blocking.py       # Polars Rust-vectorized inverted index blocking
│   │       │   ├── features.py       # RapidFuzz SIMD similarity feature extraction
│   │       │   ├── model.py          # LightGBM training & model serialization
│   │       │   ├── postprocess.py    # Collective many-to-one conflict resolution
│   │       │   ├── evaluate.py       # Local macro F_0.5 metric evaluation
│   │       │   ├── predict.py        # Streaming end-to-end test inference engine
│   │       │   └── train.py          # Training CLI entrypoint
│   │       ├── models/
│   │       │   └── lgbm_matcher.pkl  # Serialized trained LightGBM model
│   │       ├── README.md             # Code reproduction guide
│   │       └── requirements.txt      # Exact pinned dependencies
│   ├── Documentation_template.md     # Official methodology documentation
│   └── submission_log.md             # Experiment and version tracking log
├── guidelines.pdf                    # Competition guidelines
├── prb statement.pdf                 # Problem statement specifications
└── .gitignore
```

---

## ⚡ Reproduction Guide

### 1. Environment Setup
```bash
cd team_submission/code/business_entity_resolution
pip install -r requirements.txt
```

### 2. Model Training
```bash
python -m src.train --train-dir path/to/dataset/train
```

### 3. Test Inference
```bash
python -m src.predict --test-dir path/to/dataset/test --output-dir ../../output
```

### 4. Submission Validation
```bash
python path/to/validate_submission.py \
    --matching ../../output/matching_results.tsv \
    --candidate ../../output/candidate_pairs.tsv \
    --test-dir path/to/dataset/test
```
