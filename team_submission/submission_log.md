# Submission Log — Amazon ML Challenge 2026
**Team:** Amazon ML ER Champions  
**Problem:** Business Entity Resolution across Multi-Source Records (Source 1, Source 2, Source 3)  
**Evaluation Metric:** Macro F_0.5 Score  

---

## Experiment & Model History

| Date/Time (IST) | Version / Tag | Local Validation Macro F_0.5 | Threshold | Description & Key Innovations |
|---|---|---|---|---|
| 2026-09-25 16:30 | `v0.1-baseline` | 0.4120 | 0.50 | Basic string similarity threshold without collective resolution |
| 2026-09-25 16:55 | `v0.2-polars-blocking` | 0.6580 | 0.50 | Polars vectorized inverted index blocking (US: 96.8% recall, India: 90.8% recall) |
| 2026-09-25 17:05 | `v0.3-gbdt-many2one` | **0.7741** | **0.55** | LightGBM classifier with RapidFuzz features + Many-to-One collective resolution |
| 2026-09-25 17:15 | `v1.0-final-submission` | **0.7741** | **0.55** | Full multi-country pipeline (US, India, France) producing `matching_results.tsv` and `candidate_pairs.tsv` |

---

## Validation Summary
- **US Blocking Recall:** 96.83%
- **India Blocking Recall:** 90.79%
- **Combined Train Pairs:** 614,251 pairs across US & India
- **Best Macro F_0.5 Score:** **0.7741** at threshold = **0.55**
- **Many-to-One Resolution:** Enforces uniqueness of S2/S3 entity assignment, eliminating duplicate false merges and maximizing precision under F_0.5.
