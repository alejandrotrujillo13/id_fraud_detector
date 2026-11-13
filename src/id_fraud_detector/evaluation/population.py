"""From the people with photos to the whole labelled population.

Photos were downloaded for every positive but only for a sample of the negatives, and the
sample is not proportional: every negative of the app and every one registered since September
2026 is in it, the rest were drawn by month and channel. Anything that speaks about "the
legitimate customers" (how many go to review, what a score means for a new customer) has to
give each sampled negative the weight of the negatives it stands for.
"""
import numpy as np
import pandas as pd


def sampling_weights(manifest: pd.DataFrame, anon_ids) -> pd.Series:
    """Weight of each person in `anon_ids` (the ones with photos), indexed by anon_id.

    A negative weighs as many negatives of the manifest as there are per sampled negative in its
    cell (channel and month of registration). A positive weighs 1: all of them have photos.
    `manifest` needs anon_id, label, channel and registered_date."""
    table = manifest[["anon_id", "label", "channel", "registered_date"]].copy()
    table["cell"] = table["channel"].fillna("(none)") + "|" + table["registered_date"].astype(str).str[:7]
    table["sampled"] = table["anon_id"].isin(set(anon_ids))
    negatives = table[table["label"] == "NEGATIVE"]
    cells = negatives.groupby("cell")["sampled"].agg(population="size", sampled="sum")
    empty = cells[cells["sampled"] == 0]
    if len(empty):
        raise ValueError(f"{len(empty)} cells of negatives have nobody with photos ({int(empty['population'].sum())} people)")
    ratio = cells["population"] / cells["sampled"]
    chosen = table[table["sampled"]].set_index("anon_id")
    weights = chosen["cell"].map(ratio).where(chosen["label"] == "NEGATIVE", 1.0)
    return weights.reindex(list(anon_ids)).astype(float)


def weighted_share_above(scores, weights, threshold: float) -> float:
    """Share of the weight whose score is above the threshold."""
    scores, weights = np.asarray(scores, dtype=float), np.asarray(weights, dtype=float)
    return float(weights[scores > threshold].sum() / weights.sum())


def weighted_threshold(scores, weights, share: float) -> float:
    """The lowest score t such that at most `share` of the weight has a score above t."""
    scores, weights = np.asarray(scores, dtype=float), np.asarray(weights, dtype=float)
    order = np.argsort(-scores)
    scores, above = scores[order], np.cumsum(weights[order]) / weights.sum()  # above[i]: share with score >= scores[i]
    allowed = above <= share
    last = int(np.flatnonzero(allowed).max()) if allowed.any() else -1
    # The threshold is the next score down: only the scores strictly above it are counted.
    return float(scores[last + 1]) if last + 1 < len(scores) else float(scores[-1]) - 1.0
