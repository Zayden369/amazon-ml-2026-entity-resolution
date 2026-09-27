"""Official-style macro F0.5 evaluation, including singleton queries."""


def evaluate(truth: dict[str, set[str]], predictions: dict[str, set[str]]) -> dict[str, float]:
    scores: list[float] = []
    true_positive = false_positive = false_negative = 0
    for query_id, expected in truth.items():
        predicted = predictions.get(query_id, set())
        tp = len(expected & predicted)
        fp = len(predicted - expected)
        fn = len(expected - predicted)
        true_positive += tp
        false_positive += fp
        false_negative += fn
        if not expected and not predicted:
            scores.append(1.0)
            continue
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        denominator = 0.25 * precision + recall
        scores.append(1.25 * precision * recall / denominator if denominator else 0.0)
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    return {
        "macro_f05": sum(scores) / len(scores) if scores else 0.0,
        "pair_precision": precision,
        "pair_recall": recall,
    }

