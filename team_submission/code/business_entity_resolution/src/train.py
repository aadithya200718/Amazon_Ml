"""
Training entrypoint for Business Entity Resolution.
Amazon ML Challenge 2026.
"""

import argparse
from pathlib import Path
from .config import DEFAULT_TRAIN_DIR, MODEL_PATH
from .model import train_matching_model


def main():
    parser = argparse.ArgumentParser(description="Train LightGBM Matching Model")
    parser.add_argument("--train-dir", type=Path, default=DEFAULT_TRAIN_DIR, help="Path to train data directory")
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH, help="Path to save trained model")
    parser.add_argument("--sample-size", type=int, default=50000, help="Number of S1 entities to train on")
    args = parser.parse_args()

    print("Starting model training pipeline...")
    train_matching_model(train_dir=args.train_dir, model_path=args.model_path, sample_size=args.sample_size)
    print("Training complete.")


if __name__ == "__main__":
    main()
