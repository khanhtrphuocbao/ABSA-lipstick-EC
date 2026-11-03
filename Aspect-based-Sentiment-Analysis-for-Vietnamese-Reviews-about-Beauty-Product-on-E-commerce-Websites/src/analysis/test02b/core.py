"""Pure helpers for TEST-02B metric checks and fixed-21 decomposition."""

import math
import statistics

SENTIMENTS = ("positive", "neutral", "negative")
CANDIDATES = ("TFIDF-02", "BIGRU-02", "PHOBERT-02")
SEEDS = (42, 2026, 3407)


def validate_seed_matrix(keys):
    keys = list(keys)
    expected = {(candidate, seed) for candidate in CANDIDATES for seed in SEEDS}
    if set(keys) != expected or len(keys) != len(expected):
        raise ValueError("candidate/seed matrix must contain each of the nine runs exactly once")


def sample_summary(values):
    """Return arithmetic mean and sample SD (ddof=1), or nulls if unsupported."""
    values = [value for value in values if value is not None]
    if not values:
        return None, None
    return statistics.mean(values), (statistics.stdev(values) if len(values) > 1 else None)


def validate_pair_counts(metric):
    """Enforce evaluator TP/FP/FN identities and absent-support recall convention."""
    support, predicted = metric["support"], metric["predicted_positive"]
    tp, fp, fn = metric["tp"], metric["fp"], metric["fn"]
    if tp + fn != support or tp + fp != predicted:
        raise ValueError("pair TP/FP/FN counts do not reconcile with support")
    if support == 0 and metric["recall"] is not None:
        raise ValueError("zero-support recall must be null")
    if support > 0 and metric["recall"] is None:
        raise ValueError("supported pair recall cannot be null")
    if predicted == 0 and metric["precision"] is not None:
        raise ValueError("zero-prediction precision must be null")
    if predicted > 0 and metric["precision"] is None:
        raise ValueError("predicted pair precision cannot be null")


def decompose_contributions(left, right, labels, comparison, seed):
    """Decompose a fixed-21 Macro-F1 gap into the three seven-pair groups."""
    if len(labels) != 21 or len(set(labels)) != 21:
        raise ValueError("fixed-21 decomposition requires 21 unique labels")
    rows = []
    for sentiment in SENTIMENTS:
        differences = [
            left[label]["f1"] - right[label]["f1"]
            for label in labels
            if label.endswith(":" + sentiment)
        ]
        if len(differences) != 7:
            raise ValueError(f"{sentiment} must contain exactly seven aspect pairs")
        mean_difference = statistics.mean(differences)
        rows.append(
            {
                "comparison": comparison,
                "seed": seed,
                "sentiment": sentiment,
                "mean_pair_f1_difference": mean_difference,
                "fixed21_contribution": sum(differences) / 21,
            }
        )
    return rows


def verify_decomposition(rows, left_macro_f1, right_macro_f1, *, tolerance=1e-12):
    total = sum(row["fixed21_contribution"] for row in rows)
    if not math.isclose(total, left_macro_f1 - right_macro_f1, rel_tol=1e-10, abs_tol=tolerance):
        raise ValueError("sentiment contributions do not reconcile to fixed-21 Macro-F1 gap")
