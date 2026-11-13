"""Phase 3 · Audit of a training run: does the classifier follow the forgery, or the camera
and the date?

The channel decides the camera, and the positives are concentrated in some months, so a network
could score well by learning those instead of the document. This script reads the scores of a
run (runs/<name>/oof_predictions.csv, each given by a model that never saw that person) and the
photos on disk, connects to nothing, and prints:

  - how the model does inside groups where camera and date are the same for everybody
    (channel, photo size, registration period): if it still separates there, it is looking at
    the document;
  - what a "model" that never looks at the picture would score, knowing only channel, photo
    size and period;
  - recall and false positives by photo width and by period.

Only counts and rates are printed.

Usage:  python scripts/32_audit_run.py runs/<name>
"""
import json
import sys
from pathlib import Path

import pandas as pd
from PIL import Image

from id_fraud_detector.common import image_path, load_config, output_dir
from id_fraud_detector.evaluation import metrics

run = Path(sys.argv[1])
side = json.loads((run / "config.json").read_text(encoding="utf-8"))["side"]
if side == "both":  # a combined run (33_combine_sides.py): the camera frame is measured on the back photo
    side = "back"
out = output_dir(load_config())
pd.set_option("display.width", 250)
oof = pd.read_csv(run / "oof_predictions.csv")
manifest = pd.read_csv(out / "manifest.csv", usecols=["anon_id", "tenant", "registered_date"])
df = oof.merge(manifest, on="anon_id")
if (run / "history.csv").exists():  # a combined run has no training history of its own
    history = pd.read_csv(run / "history.csv")
    print("epochs per fold:", history.groupby("fold").size().to_dict(),
          " best validation PR-AUC per fold:", history.groupby("fold")["val_pr_auc"].max().round(3).to_dict())

sizes = []
for r in df.itertuples(index=False):
    path = image_path(out, r.tenant, r.label, r.anon_id, side)
    with Image.open(path) as image:
        sizes.append((image.size[0], image.size[1], path.stat().st_size / 1e6))
df[["width", "height", "megabytes"]] = sizes
df["size"] = df["width"].astype(str) + "x" + df["height"].astype(str)
df["width_band"] = pd.cut(df["width"], [0, 700, 1400, 100000], labels=["<=700", "701-1400", ">1400"])
df["period"] = pd.cut(pd.to_datetime(df["registered_date"]),
                      pd.to_datetime(["2024-12-31", "2025-06-30", "2025-12-31", "2026-08-31", "2027-01-01"]),
                      labels=["2025 H1", "2025 H2", "2026 Jan-Aug", "2026 Sep-Nov"])
threshold = metrics.threshold_at_fpr(df["positive"], df["score"], 0.02)
df["flagged"] = df["score"] > threshold
print(f"people: {len(df)}  threshold at 2% FPR: {threshold:.3f}  overall: {metrics.summary(df['positive'], df['score'])}")


def table(by):
    rows = []
    for key, g in df.groupby(by, observed=True):
        y = g["positive"].to_numpy()
        key = key if isinstance(key, tuple) else (key,)
        rows.append({**dict(zip(by, key)), "positives": int(y.sum()), "negatives": int((~y).sum()),
                     "share_positive": round(float(y.mean()), 3),
                     "recall": round(float(g.loc[g.positive, "flagged"].mean()), 3) if y.any() else None,
                     "fpr": round(float(g.loc[~g.positive, "flagged"].mean()), 3) if (~y).any() else None,
                     "auc_inside": round(metrics.roc_auc(y, g["score"]), 3) if y.sum() >= 5 and (~y).sum() >= 5 else None})
    return pd.DataFrame(rows)


print("\n== By channel and width of the photo: is the label predictable from the camera? does the model work inside each cell?")
print(table(["channel", "width_band"]).to_string(index=False))
print("\n== By registration period")
print(table(["period"]).to_string(index=False))
print("\n== By channel and period")
print(table(["channel", "period"]).to_string(index=False))


def metadata_baseline(columns):
    """Out-of-fold positive rate of each cell of `columns`: what a 'model' that only knows the
    camera and the date would score."""
    score = pd.Series(0.0, index=df.index)
    for fold in sorted(df["fold"].unique()):
        train, test = df[df["fold"] != fold], df[df["fold"] == fold]
        rate = train.groupby(columns, observed=True)["positive"].agg(["sum", "size"])
        smoothed = (rate["sum"] + train["positive"].mean() * 5) / (rate["size"] + 5)
        keys = pd.MultiIndex.from_frame(test[columns]) if len(columns) > 1 else test[columns[0]]
        score[test.index] = smoothed.reindex(keys).fillna(train["positive"].mean()).to_numpy()
    return metrics.summary(df["positive"], score)


print("\n== Baselines that never look at the picture (out-of-fold positive rate of each cell)")
for columns in (["channel"], ["channel", "size"], ["channel", "size", "period"]):
    print(f"   {' + '.join(columns):28s} {metadata_baseline(columns)}")
megabytes_auc = metrics.roc_auc(df["positive"], df["megabytes"])
print(f"   file size alone: roc_auc {megabytes_auc:.3f}")

print("\n== Model vs the best metadata cell, inside the cells with enough people (channel + exact size + period)")
cells = df.groupby(["channel", "size", "period"], observed=True)
rows = []
for key, g in cells:
    y = g["positive"].to_numpy()
    if y.sum() >= 8 and (~y).sum() >= 8:
        rows.append({"cell": " | ".join(map(str, key)), "positives": int(y.sum()), "negatives": int((~y).sum()),
                     "auc_inside": round(metrics.roc_auc(y, g["score"]), 3),
                     "recall": round(float(g.loc[g.positive, "flagged"].mean()), 3),
                     "fpr": round(float(g.loc[~g.positive, "flagged"].mean()), 3)})
print(pd.DataFrame(rows).to_string(index=False))
covered = sum(r["positives"] for r in rows)
weighted = sum(r["auc_inside"] * r["positives"] for r in rows) / max(1, covered)
print(f"positives in those cells: {covered} of {int(df.positive.sum())}   AUC inside cells, weighted by positives: {weighted:.3f}")

print("\n== Score of the negatives and positives: quantiles")
print(df.groupby("positive")["score"].quantile([0.1, 0.25, 0.5, 0.75, 0.9]).unstack().round(3).to_string())
