"""
End-to-end prediction pipeline for test dataset.
Generates matching_results.tsv and candidate_pairs.tsv.
Amazon ML Challenge 2026.
"""

import os
import sys
import time
import argparse
from pathlib import Path
from collections import Counter
import polars as pl
import numpy as np

from .config import (
    DEFAULT_TEST_DIR,
    DEFAULT_OUTPUT_DIR,
    MATCHING_OUTPUT_PATH,
    CANDIDATE_OUTPUT_PATH,
    MODEL_PATH,
    MAX_KEY_FREQUENCY,
    MAX_CANDIDATES_PER_S1,
    MATCH_THRESHOLD
)
from .blocking import generate_keys_df, build_inverted_index
from .features import extract_pair_features
from .model import load_matching_model, train_matching_model
from .postprocess import resolve_many_to_one
from .normalize import canonical_country


def run_inference(test_dir: Path = DEFAULT_TEST_DIR, output_dir: Path = DEFAULT_OUTPUT_DIR, threshold: float = MATCH_THRESHOLD):
    """
    Run complete streaming end-to-end inference on the test set.
    Writes:
      - output_dir / matching_results.tsv
      - output_dir / candidate_pairs.tsv
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    matching_file = output_dir / "matching_results.tsv"
    candidate_file = output_dir / "candidate_pairs.tsv"

    # Ensure model is available
    if not MODEL_PATH.exists():
        print(f"Model not found at {MODEL_PATH}. Training model now...", flush=True)
        train_matching_model()

    print(f"Loading trained matching model from {MODEL_PATH}...", flush=True)
    model = load_matching_model(MODEL_PATH)

    print(f"\nLoading test sources from {test_dir}...", flush=True)
    t0 = time.time()
    s1_df = pl.read_csv(test_dir / "test_source1.tsv", separator="\t")
    s2_df = pl.read_csv(test_dir / "test_source2.tsv", separator="\t")
    s3_df = pl.read_csv(test_dir / "test_source3.tsv", separator="\t")
    print(f"Loaded {len(s1_df)} S1, {len(s2_df)} S2, {len(s3_df)} S3 records in {time.time()-t0:.2f}s", flush=True)

    # Initialize output files with headers
    with open(matching_file, "w", encoding="utf-8") as f_m:
        f_m.write("source1_entity_id\tmatched_entity_ids\n")

    with open(candidate_file, "w", encoding="utf-8") as f_c:
        f_c.write("source1_entity_id\tcandidate_entity_ids\n")

    # Group records by country
    countries = list(s1_df["country"].unique())
    print(f"Countries present in test set: {countries}", flush=True)

    total_candidates_count = 0
    total_matches_count = 0
    total_s1_processed = 0

    for country in countries:
        c_name = canonical_country(country)
        print(f"\n==========================================", flush=True)
        print(f"PROCESSING COUNTRY: {country} (canonical: {c_name})", flush=True)
        print(f"==========================================", flush=True)

        country_s1 = s1_df.filter(pl.col("country") == country)
        country_targets = pl.concat([
            s2_df.filter(pl.col("country") == country),
            s3_df.filter(pl.col("country") == country)
        ])
        n_s1 = len(country_s1)
        print(f"Country {country}: {n_s1} S1 records, {len(country_targets)} Targets.", flush=True)

        s1_meta = {row[0]: (row[1], row[2]) for row in country_s1.iter_rows()}
        target_meta = {row[0]: (row[1], row[2]) for row in country_targets.iter_rows()}

        t_k = time.time()
        print("Generating blocking keys for targets...", flush=True)
        target_keys_df = generate_keys_df(country_targets)
        print(f"Target keys generated in {time.time()-t_k:.2f}s", flush=True)

        t_idx = time.time()
        print("Building inverted index...", flush=True)
        inverted_index = build_inverted_index(target_keys_df, max_frequency=MAX_KEY_FREQUENCY)
        print(f"Index built: {len(inverted_index)} keys kept in {time.time()-t_idx:.2f}s", flush=True)

        t_s1 = time.time()
        print("Generating keys for S1 entities...", flush=True)
        s1_keys_df = generate_keys_df(country_s1)
        print(f"S1 keys generated in {time.time()-t_s1:.2f}s", flush=True)

        # Batch processing: stream candidates to file, collect scored pairs for many-to-one
        BATCH_SIZE = 50000
        country_scored_pairs = []

        print(f"Scoring S1 candidates in batches of {BATCH_SIZE}...", flush=True)

        # Open candidate file in append mode for streaming
        with open(candidate_file, "a", encoding="utf-8") as f_c:
            for b_start in range(0, n_s1, BATCH_SIZE):
                b_end = min(b_start + BATCH_SIZE, n_s1)
                batch_s1_keys = s1_keys_df.slice(b_start, b_end - b_start)

                batch_pairs = []
                batch_features = []

                for row in batch_s1_keys.iter_rows():
                    s1_id = row[0]
                    s1_n, s1_a = s1_meta[s1_id]

                    cand_counter = Counter()
                    for k in row[2:]:
                        if k and k in inverted_index:
                            for tid in inverted_index[k]:
                                cand_counter[tid] += 1

                    top_cands = cand_counter.most_common(MAX_CANDIDATES_PER_S1)
                    cand_ids = [tid for tid, _ in top_cands]
                    
                    # Stream candidate pairs to file
                    f_c.write(f"{s1_id}\t{','.join(cand_ids)}\n")
                    total_candidates_count += len(cand_ids)

                    for tid, n_k in top_cands:
                        t_n, t_a = target_meta.get(tid, ("", ""))
                        feat = extract_pair_features(s1_n, s1_a, t_n, t_a, n_k, tid.startswith("S2-"))
                        batch_pairs.append((s1_id, tid))
                        batch_features.append(feat)

                if batch_features:
                    X_batch = np.array(batch_features, dtype=np.float32)
                    probs = model.predict_proba(X_batch)[:, 1]
                    for (s1_id, tid), prob in zip(batch_pairs, probs):
                        if prob >= (threshold - 0.10):  # Keep candidates near threshold
                            country_scored_pairs.append((s1_id, tid, float(prob)))

                print(f"  Processed {b_end}/{n_s1} S1 entities (Candidate pairs scored: {len(country_scored_pairs)})...", flush=True)

        # Collective Many-to-One resolution for this country
        print(f"Applying many-to-one resolution on {len(country_scored_pairs)} scored candidate pairs...", flush=True)
        country_matches = resolve_many_to_one(country_scored_pairs, threshold=threshold)

        # Stream matches to matching_results.tsv
        with open(matching_file, "a", encoding="utf-8") as f_m:
            for s1_id in country_s1["entity_id"]:
                m_list = country_matches.get(s1_id, [])
                f_m.write(f"{s1_id}\t{','.join(m_list)}\n")
                total_matches_count += len(m_list)

        total_s1_processed += n_s1
        print(f"Completed {country}: {n_s1} entities processed, {len(country_matches)} matched entities.", flush=True)

    print(f"\n==========================================", flush=True)
    print(f"INFERENCE SUMMARY", flush=True)
    print(f"==========================================", flush=True)
    print(f"Total S1 entities processed: {total_s1_processed} / {len(s1_df)}", flush=True)
    print(f"Total candidates written: {total_candidates_count}", flush=True)
    print(f"Total matches predicted: {total_matches_count}", flush=True)
    print(f"Total inference runtime: {time.time()-t0:.2f}s", flush=True)
    print(f"Matching file: {matching_file}", flush=True)
    print(f"Candidate file: {candidate_file}", flush=True)
    return matching_file, candidate_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Entity Resolution Inference")
    parser.add_argument("--test-dir", type=Path, default=DEFAULT_TEST_DIR, help="Path to test dataset directory")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Path to output directory")
    parser.add_argument("--threshold", type=float, default=MATCH_THRESHOLD, help="Classification probability threshold")
    args = parser.parse_args()

    run_inference(test_dir=args.test_dir, output_dir=args.output_dir, threshold=args.threshold)
