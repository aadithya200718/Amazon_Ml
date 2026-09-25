"""
Post-processing module: collective many-to-one resolution.
Amazon ML Challenge 2026.
"""

from typing import List, Tuple, Dict, Set


def resolve_many_to_one(scored_pairs: List[Tuple[str, str, float]], threshold: float) -> Dict[str, List[str]]:
    """
    Apply many-to-one collective resolution:
    - Filter pairs by threshold (probability >= threshold).
    - Sort candidate pairs by confidence (probability) descending.
    - An S1 entity may keep multiple target matches.
    - Each target record (S2/S3) is assigned to at most ONE S1 entity (its highest-confidence claim).
    
    Returns: dict mapping source1_entity_id -> list of matched target IDs.
    """
    kept = [p for p in scored_pairs if p[2] >= threshold]
    kept.sort(key=lambda p: p[2], reverse=True)

    claimed: Set[str] = set()
    matches: Dict[str, List[str]] = {}

    for s1_id, cand_id, _ in kept:
        if cand_id in claimed:
            continue
        matches.setdefault(s1_id, []).append(cand_id)
        claimed.add(cand_id)

    return matches
