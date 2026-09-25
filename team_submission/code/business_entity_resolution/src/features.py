"""
Pairwise feature engineering module.
Amazon ML Challenge 2026.
"""

import re
from rapidfuzz import fuzz

FEATURE_NAMES = [
    "n_ratio",
    "n_tsort",
    "n_tset",
    "n_partial",
    "a_ratio",
    "a_tsort",
    "a_tset",
    "combo_tset",
    "num_match",
    "has_num_match",
    "addr_none",
    "len_diff",
    "n_keys",
    "is_s2"
]


def extract_pair_features(s1_name: str, s1_addr: str, t_name: str, t_addr: str, n_keys: int, is_s2: bool) -> list:
    """
    Compute similarity features for a pair of entities (S1, Target).
    """
    s1_n = s1_name or ""
    s1_a = s1_addr or ""
    t_n = t_name or ""
    t_a = t_addr or ""

    # Name similarity metrics
    n_ratio = fuzz.ratio(s1_n, t_n) / 100.0
    n_tsort = fuzz.token_sort_ratio(s1_n, t_n) / 100.0
    n_tset = fuzz.token_set_ratio(s1_n, t_n) / 100.0
    n_partial = fuzz.partial_ratio(s1_n, t_n) / 100.0

    # Address similarity metrics
    a_ratio = fuzz.ratio(s1_a, t_a) / 100.0
    a_tsort = fuzz.token_sort_ratio(s1_a, t_a) / 100.0
    a_tset = fuzz.token_set_ratio(s1_a, t_a) / 100.0

    # Combined full text similarity
    combo_tset = fuzz.token_set_ratio(f"{s1_n} {s1_a}", f"{t_n} {t_a}") / 100.0

    # Address numbers overlap
    s1_nums = set(re.findall(r"\b\d{2,6}\b", s1_a))
    t_nums = set(re.findall(r"\b\d{2,6}\b", t_a))
    num_match = float(len(s1_nums & t_nums)) if s1_nums else 0.0
    has_num_match = 1.0 if num_match > 0.0 else 0.0

    # Address missingness flag
    addr_none = 1.0 if (not t_a or t_a.lower() in ("none", "null")) else 0.0

    # Relative name length difference
    len_diff = abs(len(s1_n) - len(t_n)) / max(len(s1_n), len(t_n), 1)

    return [
        n_ratio, n_tsort, n_tset, n_partial,
        a_ratio, a_tsort, a_tset,
        combo_tset,
        num_match, has_num_match, addr_none,
        len_diff,
        float(n_keys),
        1.0 if is_s2 else 0.0
    ]
