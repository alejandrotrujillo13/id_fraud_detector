"""Metrics for a score on an unbalanced dataset (numpy only).

The main one is the recall at a fixed false-positive rate: how many forged IDs are caught when
only that share of the genuine ones may be sent to manual review.
"""
import numpy as np
import pandas as pd


def _arrays(labels, scores) -> tuple[np.ndarray, np.ndarray]:
    return np.asarray(labels, dtype=bool), np.asarray(scores, dtype=float)


def roc_auc(labels, scores) -> float:
    """Probability that a positive scores higher than a negative (ties count half)."""
    y, s = _arrays(labels, scores)
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if not n_pos or not n_neg:
        return float("nan")
    _, inverse, counts = np.unique(s, return_inverse=True, return_counts=True)
    ranks = (np.cumsum(counts) - (counts - 1) / 2)[inverse]  # average rank, 1-based
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def average_precision(labels, scores) -> float:
    """Area under the precision-recall curve (step-wise, as scikit-learn computes it)."""
    y, s = _arrays(labels, scores)
    n_pos = int(y.sum())
    if not n_pos:
        return float("nan")
    order = np.argsort(-s, kind="mergesort")
    y_sorted, s_sorted = y[order], s[order]
    last_of_tie = np.r_[np.nonzero(np.diff(s_sorted))[0], len(s) - 1]
    true_positives = np.cumsum(y_sorted)[last_of_tie]
    precision = true_positives / (last_of_tie + 1)
    recall = true_positives / n_pos
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def threshold_at_fpr(labels, scores, max_fpr: float = 0.02) -> float:
    """Score above which at most `max_fpr` of the negatives fall. Flag with `score > threshold`."""
    y, s = _arrays(labels, scores)
    negatives = np.sort(s[~y])[::-1]
    if not len(negatives):
        return float("-inf")
    allowed = int(np.floor(max_fpr * len(negatives)))
    return float(negatives[min(allowed, len(negatives) - 1)])


def recall_at_fpr(labels, scores, max_fpr: float = 0.02) -> float:
    y, s = _arrays(labels, scores)
    if not y.any():
        return float("nan")
    return float((s[y] > threshold_at_fpr(y, s, max_fpr)).mean())


def summary(labels, scores, max_fpr: float = 0.02) -> dict:
    y, s = _arrays(labels, scores)
    return {"positives": int(y.sum()), "negatives": int((~y).sum()),
            "pr_auc": round(average_precision(y, s), 4), "roc_auc": round(roc_auc(y, s), 4),
            f"recall_at_{max_fpr:.0%}_fpr": round(recall_at_fpr(y, s, max_fpr), 4)}


def by_group(df: pd.DataFrame, column: str, threshold: float) -> pd.DataFrame:
    """Per value of `column` (channel, label source...): how the score separates the positives
    and negatives of that group, and what the global `threshold` flags inside it. `df` needs
    `positive` (bool) and `score`."""
    rows = []
    for value, group in df.groupby(column, dropna=False):
        y, s = _arrays(group["positive"], group["score"])
        flagged = s > threshold
        rows.append({column: value, "positives": int(y.sum()), "negatives": int((~y).sum()),
                     "positives_flagged": int(flagged[y].sum()),
                     "recall": round(float(flagged[y].mean()), 4) if y.any() else None,
                     "negatives_flagged": int(flagged[~y].sum()),
                     "false_positive_rate": round(float(flagged[~y].mean()), 4) if (~y).any() else None,
                     "roc_auc_inside_group": round(roc_auc(y, s), 4) if y.any() and (~y).any() else None})
    return pd.DataFrame(rows)
