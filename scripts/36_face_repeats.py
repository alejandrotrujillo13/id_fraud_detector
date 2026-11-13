"""Phase 3 · Repeated faces: is the same person on the front of more than one ID, and does the
classifier detect those people better because it saw their face in training?

It connects to nothing. It reads the front photos already on disk, finds and describes the face
of each one with two public OpenCV models (data/face_models/, see README) and compares every
pair of people. Two faces "match" when their similarity is above a threshold; the threshold is
chosen so that two different people (two negatives) match by chance only once in --chance pairs.

It prints counts only. It leaves, under data/<stage>/faces/ (biometric data, never versioned):
  - faces.csv, embeddings.npy, crops.npy: what was read, reused on the next run (--force redoes it);
  - face_groups.csv: anon_id and face_group of the people whose face matches someone else's;
  - pairs/: pictures of matching pairs of positives (the two aligned faces side by side), spread
    over the range of similarity, so a person can check that the matches are real.

With --run it also compares the detection of the positives whose face is on another positive of
the folds their model was trained on with the detection of the rest. The back score is the
control: a face cannot help the back model.

Usage:  python scripts/36_face_repeats.py [--run runs/ensemble] [--chance 1e-5] [--pairs 60] [--force]
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from id_fraud_detector.common import image_path, load_config, output_dir
from id_fraud_detector.evaluation import metrics
from id_fraud_detector.evaluation.faces import (CROP_SIZE, SFACE_THRESHOLD, FaceReader, chance_rate, chance_threshold,
                                                face_groups, pairs_above)

RING_SIZE = 50  # a group of the split this large is a ring kept whole in one fold


def read_faces(people: pd.DataFrame, out: Path, model_dir: Path, force: bool, workers: int):
    """Face of the front photo of every person, in the order of `people`. Cached on disk."""
    folder = out / "faces"
    index_path, embeddings_path, crops_path = folder / "faces.csv", folder / "embeddings.npy", folder / "crops.npy"
    if not force and index_path.exists() and embeddings_path.exists() and crops_path.exists():
        index = pd.read_csv(index_path)
        if list(index["anon_id"]) == list(people["anon_id"]):
            return index, np.load(embeddings_path), np.load(crops_path, mmap_mode="r")
    folder.mkdir(parents=True, exist_ok=True)
    reader = FaceReader(model_dir)
    paths = [image_path(out, r.tenant, r.label, r.anon_id, "front") for r in people.itertuples(index=False)]
    embeddings = np.zeros((len(people), 128), dtype=np.float32)
    crops = np.zeros((len(people), CROP_SIZE, CROP_SIZE, 3), dtype=np.uint8)
    rows = []
    with ThreadPoolExecutor(max_workers=workers) as pool:  # threads decode the PNG; one reader finds the faces
        photos = pool.map(lambda path: cv2.imread(str(path), cv2.IMREAD_COLOR), paths)
        for i, photo in enumerate(tqdm(photos, total=len(paths), desc="   reading faces")):
            face = reader.read(photo) if photo is not None else None
            if face is not None:
                embeddings[i], crops[i] = face["embedding"], face["crop"]
            rows.append({"anon_id": people["anon_id"].iloc[i], "found": face is not None,
                         "photo_width": 0 if photo is None else photo.shape[1],
                         "face_width": round(face["width"]) if face else 0,
                         "detector_score": round(face["score"], 3) if face else 0.0})
    index = pd.DataFrame(rows)
    np.save(embeddings_path, embeddings)
    np.save(crops_path, crops)
    index.to_csv(index_path, index=False)
    return index, embeddings, crops


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", help="run with out-of-fold scores (31_train.py or 33_combine_sides.py) to compare detection")
    ap.add_argument("--chance", type=float, default=1e-5, help="share of pairs of different people allowed to match")
    ap.add_argument("--pairs", type=int, default=60, help="pictures of matching pairs of positives to save")
    ap.add_argument("--force", action="store_true", help="read the faces again")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    cfg = load_config()
    out = output_dir(cfg)
    model_dir = out.parent / "face_models"
    manifest = pd.read_csv(out / "manifest.csv", usecols=["anon_id", "tenant", "label", "label_source", "channel"])
    people = pd.read_csv(out / "splits.csv").merge(manifest, on="anon_id", how="left", validate="one_to_one")
    people["positive"] = people["label"] == "POSITIVE"
    people["ring"] = people.groupby("group")["fold"].transform("size") >= RING_SIZE
    n_folds = int(people["fold"].max()) + 1
    print(f"people with photos: {len(people)}  positives: {int(people['positive'].sum())}")

    index, embeddings, crops = read_faces(people, out, model_dir, args.force, args.workers)
    people["found"], people["face_width"] = index["found"].to_numpy(), index["face_width"].to_numpy()

    print("\n== Faces found on the front photo")
    found = people.groupby(["label", "channel"], dropna=False).agg(
        people=("found", "size"), with_face=("found", "sum"),
        median_face_width_px=("face_width", lambda s: int(s[s > 0].median()) if (s > 0).any() else 0))
    found["share"] = (found["with_face"] / found["people"]).round(3)
    print(found.to_string())

    has = people["found"].to_numpy()
    pos, neg = has & people["positive"].to_numpy(), has & ~people["positive"].to_numpy()
    pos_pairs_total, cross_total = pos.sum() * (pos.sum() - 1) // 2, int(pos.sum()) * int(neg.sum())

    print("\n== Matches at several thresholds. 'by chance' is what the rate between negatives predicts.")
    print(f"   pairs: negatives {neg.sum() * (neg.sum() - 1) // 2:,}  positives {pos_pairs_total:,}  positive-negative {cross_total:,}")
    grid = {f"chance {rate:g}": chance_threshold(embeddings[neg], rate) for rate in (1e-3, 1e-4, 1e-5, 1e-6)}
    grid["SFace's own"] = SFACE_THRESHOLD
    similar = embeddings @ embeddings.T
    table = []
    for name, threshold in sorted(grid.items(), key=lambda item: item[1]):
        rate = chance_rate(embeddings[neg], threshold)
        above = np.triu(similar > threshold, k=1)
        table.append({"threshold": name, "similarity": round(threshold, 3), "rate_between_negatives": f"{rate:.2e}",
                      "positive_pairs": int(above[np.ix_(pos, pos)].sum()), "by_chance": round(rate * pos_pairs_total, 1),
                      "positive_negative_pairs": int((similar[np.ix_(pos, neg)] > threshold).sum()),
                      "by_chance_": round(rate * cross_total, 1)})
    print(pd.DataFrame(table).to_string(index=False))

    threshold = max(chance_threshold(embeddings[neg], args.chance), SFACE_THRESHOLD)
    print(f"\n== Threshold used: {threshold:.3f} (the stricter of chance {args.chance:g} and SFace's own {SFACE_THRESHOLD})")
    where = np.flatnonzero(has)
    pairs = where[pairs_above(embeddings[has], threshold)]
    people["face_group"] = face_groups(len(people), pairs)
    sizes = people.groupby("face_group")["anon_id"].transform("size")
    grouped = people[sizes > 1]
    summary = grouped.groupby("face_group").agg(people=("anon_id", "size"), positives=("positive", "sum"),
                                                folds=("fold", "nunique"))
    print(f"groups of people with the same face: {len(summary)}   people in them: {len(grouped)}")
    print(f"(about {chance_rate(embeddings[neg], threshold) * neg.sum() * (neg.sum() - 1) / 2:.0f} pairs of negatives match "
          "by chance at this threshold, so the groups of negatives are mostly not the same person)")
    print("groups by size and by how many of their people are positives:")
    summary["kind"] = np.select([summary["positives"] == summary["people"], summary["positives"] == 0],
                                ["all positives", "all negatives"], "mixed")
    print(summary.pivot_table(index="people", columns="kind", values="folds", aggfunc="size", fill_value=0).to_string())
    print(f"groups spread over more than one fold: {int((summary['folds'] > 1).sum())}")
    grouped[["anon_id", "face_group"]].to_csv(out / "faces" / "face_groups.csv", index=False)

    # For each positive: is their face on another positive, and was that one in the training folds
    # of the model that scored them? (fold k is scored by the model trained without k and k+1.)
    fold, positive = people["fold"].to_numpy(), people["positive"].to_numpy()
    twin_any = np.zeros(len(people), bool)
    twin_other_fold = np.zeros(len(people), bool)
    twin_trained = np.zeros(len(people), bool)
    twin_negative = np.zeros(len(people), bool)
    for i, j in pairs:
        for a, b in ((i, j), (j, i)):
            if positive[b]:
                twin_any[a] = True
                twin_other_fold[a] |= fold[b] != fold[a]
                twin_trained[a] |= fold[b] not in (fold[a], (fold[a] + 1) % n_folds)
            else:
                twin_negative[a] = True
    people["twin_any"], people["twin_other_fold"], people["twin_trained"] = twin_any, twin_other_fold, twin_trained
    people["twin_negative"] = twin_negative
    positives = people[people["positive"]]
    print(f"\n== Positives ({len(positives)}; with a face found: {int(positives['found'].sum())})")
    print(f"face on another positive: {int(positives['twin_any'].sum())}   on one of another fold: "
          f"{int(positives['twin_other_fold'].sum())}   on one their model trained on: {int(positives['twin_trained'].sum())}"
          f"   face on a negative: {int(positives['twin_negative'].sum())}")
    print(positives.groupby("label_source").agg(positives=("anon_id", "size"), face_found=("found", "sum"),
                                                on_another_positive=("twin_any", "sum"),
                                                on_one_their_model_trained_on=("twin_trained", "sum")).to_string())
    print(positives.groupby("fold").agg(positives=("anon_id", "size"), on_another_positive=("twin_any", "sum"),
                                        on_one_their_model_trained_on=("twin_trained", "sum")).to_string())

    if args.run:
        scores = pd.read_csv(Path(args.run) / "oof_predictions.csv")
        columns = [c for c in ("front_score", "back_score", "score") if c in scores]
        merged = people.merge(scores[["anon_id", *columns]], on="anon_id", how="inner", validate="one_to_one")
        print(f"\n== Detection with 2% of the negatives flagged, run {Path(args.run).name} ({len(merged)} people)")
        for column in columns:
            merged[f"hit_{column}"] = merged[column] > metrics.threshold_at_fpr(merged["positive"], merged[column], 0.02)
        hits = [f"hit_{c}" for c in columns]
        merged["face"] = np.select([~merged["found"], merged["twin_trained"], merged["twin_any"]],
                                   ["no face found", "face seen in training (another positive)",
                                    "face on another positive, not in training"], "face on no other positive")
        merged["origin"] = np.where(merged["ring"], "ring", "rest")
        scored = merged[merged["positive"]]
        for keys in (["face"], ["origin", "face"]):
            result = scored.groupby(keys).agg(positives=("anon_id", "size"), **{h[4:]: (h, "sum") for h in hits})
            for column in columns:
                result[f"{column}_rate"] = (result[column] / result["positives"]).round(3)
            print(result.to_string(), "\n")
        negatives = merged[~merged["positive"]]
        twins = negatives["twin_any"]
        print(f"negatives whose face is on a positive: {int(twins.sum())}; flagged: "
              + ", ".join(f"{c} {int(negatives.loc[twins, f'hit_{c}'].sum())}" for c in columns))

    if args.pairs:
        folder = out / "faces" / "pairs"
        folder.mkdir(parents=True, exist_ok=True)
        for old in folder.glob("*.png"):
            old.unlink()
        both = [(float(similar[i, j]), i, j) for i, j in pairs if positive[i] and positive[j]]
        both.sort(reverse=True)
        chosen = both if len(both) <= args.pairs else [both[k] for k in np.linspace(0, len(both) - 1, args.pairs).astype(int)]
        for value, i, j in chosen:
            picture = np.concatenate([crops[i], crops[j]], axis=1)
            cv2.imwrite(str(folder / f"{value:.3f}_{people['anon_id'].iloc[i]}_{people['anon_id'].iloc[j]}.png"), picture)
        print(f"\npictures of {len(chosen)} of the {len(both)} matching pairs of positives -> {folder}")


if __name__ == "__main__":
    main()
