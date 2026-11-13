"""Phase 3 · Heat maps (Grad-CAM) of a training run: where the network looks.

It connects to nothing. For every person it uses the model of the fold that never saw them
(runs/<name>/fold<k>.pt) and the reduced photos already cached, and leaves:

  - runs/<name>/cam_mean.png and cam_summary.csv: the average heat map of each group
    (positives detected, positives missed, negatives flagged, negatives not flagged) and how
    much of the heat falls on the outer frame of the photo, where there is background and no
    card. These are averages over many people: no photo or personal data in them;
  - data/<stage>/cam/<name>/<group>/<anon_id>_<side>.png: for the people of one group, the
    photo next to the same photo with the heat map over it. These are ID photos, so they stay
    under data/ and are never versioned.

Groups use the run's own threshold (2% of the negatives flagged). With --queue the pictures are
those of the people listed in a CSV with an `anon_id` column, for example the positives the
combined score misses (runs/ensemble/missed_positives.csv).

Usage:  python scripts/35_grad_cam.py runs/<name> [--pictures missed] [--queue file.csv] [--limit 60] [--cpu]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from id_fraud_detector.common import load_config, output_dir
from id_fraud_detector.evaluation import metrics
from id_fraud_detector.model.cam import border_share, grad_cam, overlay
from id_fraud_detector.model.data import cache_paths
from id_fraud_detector.model.net import build
from id_fraud_detector.model.train import normalize, to_input

GROUPS = {"detected": "positives detected", "missed": "positives missed",
          "false_positives": "negatives flagged", "negatives": "negatives not flagged"}
SURFACE, HEAT = (252, 252, 251), (235, 104, 52)


def mean_panel(heat: np.ndarray, size: tuple[int, int]) -> Image.Image:
    """One average heat map as a picture: white where the group never looks, the heat hue where it
    always does. Scaled by its own maximum."""
    scaled = heat / heat.max() if heat.max() > 0 else heat
    large = np.asarray(Image.fromarray((scaled * 255).astype(np.uint8)).resize(size, Image.BICUBIC), dtype=np.float32)[..., None] / 255
    return Image.fromarray((np.array(SURFACE) * (1 - large) + np.array(HEAT) * large).round().astype(np.uint8))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--pictures", default="missed", choices=[*GROUPS, "none"], help="group whose pictures are saved")
    ap.add_argument("--queue", help="CSV with an anon_id column: save the pictures of these people instead")
    ap.add_argument("--limit", type=int, default=60, help="at most this many pictures (highest score first)")
    ap.add_argument("--cpu", action="store_true", help="do not use the GPU")
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()

    run = Path(args.run)
    config = json.loads((run / "config.json").read_text(encoding="utf-8"))
    if config["side"] not in ("front", "back"):
        raise SystemExit("ERROR: this needs a training run of one side (31_train.py), not a combination.")
    side, size = config["side"], tuple(config["size"])
    out = output_dir(load_config())
    array_path, index_path = cache_paths(out, side, size)
    if not array_path.exists():
        raise SystemExit(f"ERROR: {array_path} is missing: the cache of reduced photos of that run.")
    photos = np.load(array_path, mmap_mode="r")
    row_of = {anon: i for i, anon in enumerate(pd.read_csv(index_path)["anon_id"])}

    df = pd.read_csv(run / "oof_predictions.csv")
    threshold = metrics.threshold_at_fpr(df["positive"], df["score"], 0.02)
    flagged = df["score"] > threshold
    df["group"] = np.select([df["positive"] & flagged, df["positive"] & ~flagged, ~df["positive"] & flagged],
                            ["detected", "missed", "false_positives"], "negatives")
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"run: {run.name}  side: {side}  model: {config['arch']}  people: {len(df)}  device: {device}  "
          f"threshold at 2% FPR: {threshold:.3f}")

    if args.queue:
        wanted = df[df["anon_id"].isin(set(pd.read_csv(args.queue)["anon_id"]))]
        folder = Path(args.queue).stem
    else:
        wanted = df[df["group"] == args.pictures]
        folder = args.pictures
    wanted = set(wanted.sort_values("score", ascending=False)["anon_id"][:args.limit])
    picture_dir = out / "cam" / run.name / folder
    if wanted:
        picture_dir.mkdir(parents=True, exist_ok=True)

    sums = {group: None for group in GROUPS}
    shares, checked = {}, []
    for fold in sorted(df["fold"].unique()):
        model = build(config["arch"], pretrained=False)
        model.load_state_dict(torch.load(run / f"fold{fold}.pt", map_location="cpu"))
        model.to(device)
        part = df[df["fold"] == fold]
        for start in range(0, len(part), args.batch):
            batch = part.iloc[start:start + args.batch]
            rows = np.array([row_of[a] for a in batch["anon_id"]])
            order = np.argsort(rows)  # sorted: sequential reads from the memory map
            pixels = np.asarray(photos[rows[order]])
            scores, maps = grad_cam(model, normalize(to_input(pixels, device)))
            for j, i in enumerate(order):
                person = batch.iloc[i]
                checked.append(abs(float(scores[j]) - person["score"]))
                sums[person["group"]] = maps[j] if sums[person["group"]] is None else sums[person["group"]] + maps[j]
                shares[person["anon_id"]] = border_share(maps[j])
                if person["anon_id"] in wanted:
                    pair = np.concatenate([pixels[j], overlay(pixels[j], maps[j])], axis=1)
                    Image.fromarray(pair).save(picture_dir / f"{person['anon_id']}_{side}.png")
        print(f"   fold {fold}: {len(part)} people")

    # The score recomputed here must be the one saved by the training: same model, same photo.
    print(f"largest difference with the saved scores: {max(checked):.4f}")
    df["border_share"] = df["anon_id"].map(shares)
    counts = df["group"].value_counts()
    grid = np.zeros(next(m for m in sums.values() if m is not None).shape)
    frame_area = border_share(np.ones_like(grid))
    summary = pd.DataFrame([{"group": GROUPS[g], "people": int(counts.get(g, 0)),
                             "no_heat": int(df.loc[df["group"] == g, "border_share"].isna().sum()),
                             "heat_on_outer_frame_median": round(float(df.loc[df["group"] == g, "border_share"].median()), 3)}
                            for g in GROUPS])
    print(f"\n== Heat on the outer frame of the photo (evenly spread heat would give {frame_area:.3f})")
    print(summary.to_string(index=False))
    summary.to_csv(run / "cam_summary.csv", index=False)

    panel_size, gap = (size[0] // 2, size[1] // 2), 12
    sheet = Image.new("RGB", (panel_size[0] * len(GROUPS) + gap * (len(GROUPS) + 1), panel_size[1] + 2 * gap), SURFACE)
    for k, group in enumerate(GROUPS):
        if sums[group] is not None:
            sheet.paste(mean_panel(sums[group] / counts[group], panel_size), (gap + k * (panel_size[0] + gap), gap))
    sheet.save(run / "cam_mean.png")
    print(f"\naverage heat maps, left to right: {', '.join(GROUPS.values())} -> {run / 'cam_mean.png'}")
    if wanted:
        print(f"pictures of {len(wanted)} people ({folder}) -> {picture_dir}")


if __name__ == "__main__":
    main()
