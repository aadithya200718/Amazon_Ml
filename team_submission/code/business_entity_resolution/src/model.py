"""
Model training and inference management module.
Amazon ML Challenge 2026.
"""

import os
import pickle
import time
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
import polars as pl
import lightgbm as lgb

from .config import (
    DEFAULT_TRAIN_DIR,
    DEFAULT_MODEL_DIR,
    MODEL_PATH,
    MAX_KEY_FREQUENCY,
    MATCH_THRESHOLD
)
from .blocking import generate_keys_df, build_inverted_index
from .features import extract_pair_features, FEATURE_NAMES


def train_matching_model(train_dir: Path = DEFAULT_TRAIN_DIR, model_path: Path = MODEL_PATH, sample_size: int = 40000):
    """
    Train LightGBM matching model on hard negative candidates from blocking.
    """
    print(f"Loading training datasets from {train_dir}...")
    t0 = time.time()
    s1 = pl.read_csv(train_dir / "train_source1.tsv", separator="\t")
    s2 = pl.read_csv(train_dir / "train_source2.tsv", separator="\t")
    s3 = pl.read_csv(train_dir / "train_source3.tsv", separator="\t")
    gt = pl.read_csv(train_dir / "train_ground_truth.tsv", separator="\t")

    half_sample = sample_size // 2
    sampled_s1 = pl.concat([
        s1.filter(pl.col("country") == "US").head(half_sample),
        s1.filter(pl.col("country") == "India").head(half_sample)
    ])
    sampled_s1_ids = set(sampled_s1["entity_id"])
    sampled_gt = gt.filter(pl.col("source1_entity_id").is_in(sampled_s1_ids))
    truth_dict = {row[0]: set(row[1].split(",")) if row[1] else set() for row in sampled_gt.iter_rows()}

    s1_meta = {row[0]: (row[1], row[2]) for row in sampled_s1.iter_rows()}

    X_train = []
    y_train = []

    for country in ["US", "India"]:
        print(f"Processing training country: {country}...")
        country_targets = pl.concat([
            s2.filter(pl.col("country") == country),
            s3.filter(pl.col("country") == country)
        ])
        target_meta = {row[0]: (row[1], row[2]) for row in country_targets.iter_rows()}

        target_keys_df = generate_keys_df(country_targets)
        inverted_index = build_inverted_index(target_keys_df, max_frequency=MAX_KEY_FREQUENCY)

        country_s1 = sampled_s1.filter(pl.col("country") == country)
        s1_keys_df = generate_keys_df(country_s1)

        for row in s1_keys_df.iter_rows():
            s1_id = row[0]
            s1_n, s1_a = s1_meta[s1_id]
            true_set = truth_dict.get(s1_id, set())

            cand_counter = Counter()
            for k in row[2:]:
                if k and k in inverted_index:
                    for tid in inverted_index[k]:
                        cand_counter[tid] += 1

            top_cands = [tid for tid, _ in cand_counter.most_common(20)]
            for tid in true_set:
                if tid not in top_cands and tid in target_meta:
                    top_cands.append(tid)

            for tid in top_cands:
                t_n, t_a = target_meta.get(tid, ("", ""))
                feat = extract_pair_features(s1_n, s1_a, t_n, t_a, cand_counter[tid], tid.startswith("S2-"))
                X_train.append(feat)
                y_train.append(1 if tid in true_set else 0)

    X = np.array(X_train, dtype=np.float32)
    y = np.array(y_train, dtype=np.int32)
    print(f"Training pairs built: {len(X)} (Positives: {y.sum()}, Negatives: {len(y)-y.sum()}) in {time.time()-t0:.2f}s")

    clf = lgb.LGBMClassifier(
        n_estimators=200,
        learning_rate=0.08,
        num_leaves=31,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
    clf.fit(X, y)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump(clf, f)
    print(f"Model saved successfully to {model_path}")
    return clf


def load_matching_model(model_path: Path = MODEL_PATH):
    """Load pretrained matching model from disk."""
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at {model_path}. Train model first.")
    with open(model_path, "rb") as f:
        clf = pickle.load(f)
    return clf
