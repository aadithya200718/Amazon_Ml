# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** EntityResolvers  
**Team Members:** Pair-Programming Team  
**Submission Date:** 25 September 2026  

---

## 1. Executive Summary

This solution presents a high-performance, two-stage machine learning system for large-scale Business Entity Resolution (ER) across heterogeneous and noisy data sources (Source 1 reference, Source 2, and Source 3). Our architecture pairs an ultra-fast, multi-strategy Polars-powered inverted-index blocking engine (achieving 96.8% recall on US and 90.8% on India) with a gradient-boosted decision tree (LightGBM) trained on hard-negative candidate distributions. Crucially, we enforce a collective many-to-one greedy assignment post-processing constraint and calibrate decision thresholds specifically for the precision-weighted macro $F_{0.5}$ metric, achieving a validation macro $F_{0.5}$ score of **0.7741**.

---

## 2. Methodology

### 2.1 Problem Analysis
Exploratory data analysis across the 2.2 million training records and 1.73 million test records revealed several foundational domain properties:
1. **Source Disparities:** Source 1 serves as the deduplicated reference truth. Matches from Source 2 and Source 3 contain substantial real-world noise: legal suffix variations (`Inc` vs `Incorporated`, `Pvt Ltd` vs `Private Limited`, French `SARL`, `SASU`, `EURL`), address component omissions (missing states, street types), landmark inclusions (`Opposite SBI ATM`, `Behind Oxford School`), transliteration and native script variations (Hindi, Tamil, Odia, Gujarati, French accents), and alphanumeric building/suite identifiers (`Wz-187C`, `6(29)`, `Af-684`).
2. **Strict Country Isolation:** Extensive validation confirmed 0 cross-country matches in ground truth (0 out of 693,069 inspected links). Thus, candidate search is strictly partitioned by country (US, India, France).
3. **Strict Target Uniqueness (Many-to-One):** Audit of 7,638,365 matched instances revealed that exactly 0 target records match multiple Source 1 entities. Every Source 2/3 entity corresponds to at most one real-world Source 1 business, though an S1 business may legitimately link to multiple Source 2/3 fragments.
4. **Metric Asymmetry ($F_{0.5}$):** With $\beta = 0.5$, precision is weighted 2× over recall. Merging two distinct entities drops the entity score from 1.0 to 0.0. Conservative, high-confidence decision boundaries are mathematically required.

### 2.2 Solution Strategy
**Approach Type:** Multi-Strategy High-Recall Blocking + Feature-Engineered GBDT Classifier + Collective Many-to-One Conflict Resolution.

**Core Innovation:**
1. **Multi-Strategy Inverted-Index Blocking with Rust-Vectorized Extraction:** Combining alphanumeric compressed name tokens, bigram tokens, composite name-number keys, address-locality bigrams, and street number pairings.
2. **Frequency-Bounded Index Pruning:** Eliminating frequent stop-keys (frequency > 1200) to maintain sub-second query latency and zero OOM risk across 10 million target records.
3. **Collective Many-to-One Resolution:** Globally sorting candidate pair probabilities and greedily assigning each target entity to its single highest-probability Source 1 claim, eliminating double-assignment false positives entirely.

---

## 3. Candidate Generation (Blocking)

To reduce the $1.73 \times 10^6 \times 9.97 \times 10^6 \approx 1.7 \times 10^{13}$ Cartesian comparison space to a tractable candidate set:
- **Blocking keys used:**
  - `k_comp`: Alphanumeric compressed name prefix (length 5-10, removing all punctuation, legal suffixes, spaces, and web domains).
  - `k_n2_01`, `k_n2_12`: Name token bigrams (e.g. `scott_eagle`, `crystal_lending`).
  - `k_nn`, `k_nn1`: Name prefix + address numbers (e.g. `dahl_630`, `payn_3315`).
  - `k_na0`, `k_na`: Name prefix + first/last address words.
  - `k_numa0`, `k_numa1`, `k_numa`: Street number + significant address words (e.g. `1400_wolf`, `1712_montebello`).
  - `k_num2`: Dual address numbers combination.
  - `k_a2`: Address word bigrams (e.g. `prathna_greens`, `kanhaiya_plaza`).
  - `k_nw0`, `k_nw1`, `k_nw2`: Significant name tokens.
- **Candidate pairs generated:** Capped at the top 30 candidates per Source-1 entity, producing an average of ~25 candidates per entity.
- **How true matches were preserved:**
  - Evaluated on 20,000 held-out S1 records against 10M+ targets:
    - **US Recall:** **96.83%**
    - **India Recall:** **90.79%**
  - Robust Unicode NFKD normalization decomposes accents (`é` $\to$ `e`), making French and Latin-based names seamlessly matchable.

---

## 4. Matching Model

### Features used:
- **Name Features:**
  - Levenshtein ratio (`n_ratio`)
  - Token sort ratio (`n_tsort`)
  - Token set ratio (`n_tset`)
  - Partial string ratio (`n_partial`)
  - Relative character length disparity (`len_diff`)
- **Address Features:**
  - Address Levenshtein ratio (`a_ratio`)
  - Address token sort ratio (`a_tsort`)
  - Address token set ratio (`a_tset`)
  - Address missingness / "None" indicator (`addr_none`)
  - Number of matching street/house numbers (`num_match`)
  - Boolean flag for any number match (`has_num_match`)
- **Composite & Metadata Features:**
  - Combined name + address token set ratio (`combo_tset`)
  - Total blocking keys shared (`n_keys`)
  - Target source indicator (`is_s2`: 1.0 for Source 2, 0.0 for Source 3)

### Model Architecture:
- **Model Type:** LightGBM Gradient Boosted Decision Trees (`LGBMClassifier`), max_depth=6, num_leaves=31, learning_rate=0.08, n_estimators=200.
- **Negative Sample Construction:** Extracted directly from blocking hard negatives (candidates surfaced by blocking that are not in ground truth).
- **Threshold Selection:** Evaluated across thresholds $0.30 \le \tau \le 0.70$ on 10,000 held-out validation entities using exact macro $F_{0.5}$. The optimal operating point was determined at **$\tau = 0.55$**.

---

## 5. Results & Error Analysis

- **Macro $F_{0.5}$ Score:** **0.7741** on 10,000 held-out validation entities across US and India.
- **Common False Positives:**
  - Multi-location chain stores or franchises with identical brand names in adjacent localities where street numbers differ subtly.
  - Very short generic business names sharing high token overlap.
- **Common False Negatives:**
  - Severe cross-lingual transliteration where business names are purely in regional Indian scripts and address descriptions omit street numbers.
  - Entities where both name and address were altered by heavy abbreviations simultaneously.

---

## 6. Conclusion

Our solution achieves state-of-the-art business entity resolution by combining high-speed Polars-based inverted index blocking, SIMD string similarity feature engineering, and a calibrated LightGBM classifier. By integrating domain-driven collective many-to-one conflict resolution and optimizing specifically for macro $F_{0.5}$, the pipeline delivers high precision while scaling effortlessly to the full 1.73M test set in minutes.

---

## Appendix

### A. Code Artefacts
- `code/business_entity_resolution/src/`:
  - `config.py`: Centralized configuration, hyperparameter tokens, and paths.
  - `normalize.py`: Unicode normalization, legal suffix & address stopword dictionaries.
  - `blocking.py`: Rust-vectorized key generator & in-memory inverted index.
  - `features.py`: Pairwise similarity extraction using RapidFuzz.
  - `model.py`: LightGBM training and model serialization.
  - `postprocess.py`: Collective many-to-one resolution logic.
  - `evaluate.py`: Standalone macro $F_{0.5}$ scorer with verification tests.
  - `predict.py`: End-to-end test inference generating `matching_results.tsv` and `candidate_pairs.tsv`.
  - `train.py`: Model training CLI entrypoint.
- `requirements.txt`: Exact pinned dependencies.
- `README.md`: Complete reproduction instructions.
