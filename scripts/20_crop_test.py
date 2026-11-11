"""Phase 2 · Measure how well the classic crop works on a sample of real photos.

Takes N front and N back images per tenant from the manifest, runs src/id_fraud_detector/crop.py and writes a
contact sheet per tenant (original with the detected quad + rectified result) so the success
rate can be judged by eye. Decision rule from the plan: >= 90% good crops -> keep OpenCV;
otherwise train a YOLO detector.

Usage:  python scripts/20_crop_test.py [--per-tenant 50] [--seed 42]
Writes: data/<stage>/crop_test/<tenant>_<side>_sheet.jpg and data/<stage>/crop_test/results.csv
"""
import argparse

import cv2
import numpy as np
import pandas as pd

from id_fraud_detector.common import load_config, output_dir
from id_fraud_detector.crop import crop_document

THUMB_W, THUMB_H = 320, 200
COLS = 5


def thumb(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    scale = min(THUMB_W / w, THUMB_H / h)
    resized = cv2.resize(img, (int(w * scale), int(h * scale)))
    canvas = np.full((THUMB_H, THUMB_W, 3), 40, np.uint8)
    y, x = (THUMB_H - resized.shape[0]) // 2, (THUMB_W - resized.shape[1]) // 2
    canvas[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
    return canvas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-tenant", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cfg = load_config()
    out = output_dir(cfg)
    sheets_dir = out / "crop_test"
    sheets_dir.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(out / "manifest.csv")
    manifest = manifest[manifest.has_images & manifest.front_path.notna()]

    results = []
    for tenant in cfg["tenants"]:
        sample = manifest[manifest.tenant == tenant]
        # half positives, half negatives when possible
        parts = [g.sample(min(len(g), args.per_tenant // 2), random_state=args.seed)
                 for _, g in sample.groupby("label")]
        sample = pd.concat(parts) if parts else sample
        for side in ("front", "back"):
            tiles = []
            for r in sample.itertuples(index=False):
                rel = getattr(r, f"{side}_path")
                if not isinstance(rel, str):  # this side was not downloaded
                    continue
                img = cv2.imread(str(out / rel))
                if img is None:
                    continue
                card, quad = crop_document(img)
                drawn = img.copy()
                if quad is not None:
                    cv2.polylines(drawn, [quad.astype(int)], True, (0, 255, 0), max(2, img.shape[1] // 300))
                right = card if card is not None else np.zeros((10, 16, 3), np.uint8)
                pair = np.hstack([thumb(drawn), thumb(right)])
                cv2.putText(pair, f"{r.anon_id[:6]} {r.label[:3]} {'OK' if card is not None else 'NO QUAD'}",
                            (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                tiles.append(pair)
                results.append({"anon_id": r.anon_id, "tenant": tenant, "label": r.label, "side": side,
                                "found": card is not None})
            if not tiles:
                continue
            while len(tiles) % COLS:
                tiles.append(np.zeros_like(tiles[0]))
            rows = [np.hstack(tiles[i:i + COLS]) for i in range(0, len(tiles), COLS)]
            cv2.imwrite(str(sheets_dir / f"{tenant}_{side}_sheet.jpg"), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 80])

    df = pd.DataFrame(results)
    df.to_csv(sheets_dir / "results.csv", index=False)
    print(df.groupby(["tenant", "side"]).found.mean().round(3))
    print(f"\nContact sheets in {sheets_dir}. 'found' only means a quad was detected; judge quality by eye.")


if __name__ == "__main__":
    main()
