"""
Configuration and constants for Business Entity Resolution pipeline.
Amazon ML Challenge 2026.
"""

import os
from pathlib import Path

# Paths
SRC_DIR = Path(__file__).resolve().parent
CODE_DIR = SRC_DIR.parent
PKG_DIR = CODE_DIR.parent.parent  # team_submission root
WORKSPACE_DIR = PKG_DIR.parent

# Default dataset paths
STUDENT_RESOURCE_DIR = WORKSPACE_DIR / "6ab10eb3b23ba_student_resource" / "student_resource"
DEFAULT_TRAIN_DIR = STUDENT_RESOURCE_DIR / "dataset" / "train"
DEFAULT_TEST_DIR = STUDENT_RESOURCE_DIR / "dataset" / "test"

# Output paths
DEFAULT_OUTPUT_DIR = PKG_DIR / "output"
MATCHING_OUTPUT_PATH = DEFAULT_OUTPUT_DIR / "matching_results.tsv"
CANDIDATE_OUTPUT_PATH = DEFAULT_OUTPUT_DIR / "candidate_pairs.tsv"

# Model paths
DEFAULT_MODEL_DIR = CODE_DIR / "models"
MODEL_PATH = DEFAULT_MODEL_DIR / "lgbm_matcher.pkl"

# Blocking hyperparameters
MAX_KEY_FREQUENCY = 1200      # Prune keys occurring in more than N target records
MAX_CANDIDATES_PER_S1 = 30    # Top candidates evaluated per S1 entity
NAME_MIN_WORD_LEN = 2
ADDR_MIN_WORD_LEN = 3

# Classification hyperparameters
MATCH_THRESHOLD = 0.55        # Tuned on local validation for macro F_0.5
BETA = 0.5
