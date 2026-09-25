"""
Evaluation module implementing exact macro F_0.5 score.
Amazon ML Challenge 2026.
"""


def entity_f_beta(predicted: set, truth: set, beta: float = 0.5) -> float:
    """
    Compute F_beta for one Source-1 entity according to the challenge rules:
    - Singletons: If truth is empty, predicting empty scores 1.0; predicting matches scores 0.0.
    - If truth is non-empty, predicting empty scores 0.0.
    - Precision is weighted 2x over recall (beta = 0.5).
    """
    if not truth:
        return 1.0 if not predicted else 0.0
    if not predicted:
        return 0.0
    tp = len(predicted & truth)
    precision = tp / len(predicted)
    recall = tp / len(truth)
    if precision == 0.0 and recall == 0.0:
        return 0.0
    beta2 = beta ** 2
    return (1.0 + beta2) * precision * recall / (beta2 * precision + recall)


def macro_f_beta(pred_dict: dict, truth_dict: dict, beta: float = 0.5) -> float:
    """
    Compute macro-averaged F_beta across all Source-1 entities.
    pred_dict: {source1_entity_id: set(matched_entity_ids)}
    truth_dict: {source1_entity_id: set(matched_entity_ids)}
    """
    scores = [
        entity_f_beta(pred_dict.get(s1_id, set()), truth_set, beta)
        for s1_id, truth_set in truth_dict.items()
    ]
    return sum(scores) / len(scores) if scores else 0.0


if __name__ == "__main__":
    # Self-test using the problem statement worked example:
    test_pred = {"S2-00047", "S2-00193", "S3-00812"}
    test_truth = {"S2-00047", "S3-00812"}
    score = entity_f_beta(test_pred, test_truth, beta=0.5)
    print(f"Worked example check (expected 0.714): {score:.3f}")
    assert abs(score - 0.714) < 0.001, "F_beta formula check failed!"
    print("Evaluation self-test passed.")
