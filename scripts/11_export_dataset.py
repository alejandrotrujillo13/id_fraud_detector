"""Phase 1 · Dataset construction.

For each tenant it reads (read-only) users_client, credits_credit and
users_customeridvalidationresult, applies the labeling rules from config.yaml and writes:

  data/<stage>/manifest.csv              one row per customer (positives and ALL negatives), anonymized
  data/<stage>/ocr/<anon_id>.json        Rekognition text lines (front and back) to measure the rules
  data/<stage>/images/<tenant>/<label>/<anon_id>_front.png / _back.png
                                 every positive + a stratified sample of negatives
  data/<stage>/private/id_map.csv        anon_id -> document number (LOCAL ONLY, git-ignored; needed for the backtest)

It can be re-run safely: OCR files and images already on disk are not fetched again.

Usage:  python scripts/11_export_dataset.py [--no-images] [--tenant esmio] [--refresh-ocr] [--workers 8]
"""
import argparse
import hashlib
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from io import BytesIO
from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm import tqdm

from id_fraud_detector.common import anon_id, load_config, output_dir, person_id, pg_connection, s3_client
from id_fraud_detector.dataset import (POSITIVE, fetch_clients, fetch_common_fraud_status, fetch_ocr_lines,
                                       label_clients, sample_negatives, stored_ocr)

SIDES = ("front", "back")


def export_ocr(con, df: pd.DataFrame, ocr_dir: Path, refresh: bool) -> None:
    """Write ocr/<anon_id>.json for every customer in df that does not have one yet."""
    pending = df if refresh else df[[not (ocr_dir / f"{a}.json").exists() for a in df["anon_id"]]]
    if pending.empty:
        print("   OCR: every file already on disk, nothing to fetch")
        return
    anon_by_client = dict(zip(pending["client_id"], pending["anon_id"]))
    written = set()
    with tqdm(total=len(pending), desc="   OCR") as bar:
        for row in fetch_ocr_lines(con, pending["client_id"]):
            payload = stored_ocr(row["front_lines"], row["back_lines"])
            payload["validation_updated_at"] = str(row["validation_updated_at"])
            aid = anon_by_client[row["client_id"]]
            (ocr_dir / f"{aid}.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            written.add(row["client_id"])
            bar.update(1)
        for client_id, aid in anon_by_client.items():  # customers without a validation row
            if client_id not in written:
                payload = stored_ocr(None, None)
                payload["validation_updated_at"] = None
                (ocr_dir / f"{aid}.json").write_text(json.dumps(payload), encoding="utf-8")
                bar.update(1)


def ocr_flags(ocr_dir: Path, anon_ids) -> pd.DataFrame:
    """has_front_ocr / has_back_ocr / validation_updated_at read back from the exported files."""
    rows = []
    for aid in anon_ids:
        data = json.loads((ocr_dir / f"{aid}.json").read_text(encoding="utf-8"))
        rows.append({
            "anon_id": aid,
            "has_front_ocr": bool(data["data_front_id"]["detected_text"]["TextDetections"]),
            "has_back_ocr": bool(data["data_back_id"]["detected_text"]["TextDetections"]),
            "validation_updated_at": data.get("validation_updated_at"),
        })
    return pd.DataFrame(rows, columns=["anon_id", "has_front_ocr", "has_back_ocr", "validation_updated_at"])


def fetch_image(s3, bucket: str, key: str, dest: Path) -> dict:
    """Download one image (unless it is already on disk) and return its file metadata."""
    if dest.exists() and dest.stat().st_size > 0:
        body = dest.read_bytes()
    else:
        body = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
        dest.parent.mkdir(parents=True, exist_ok=True)
        partial = dest.with_name(dest.name + ".part")
        partial.write_bytes(body)
        partial.replace(dest)
    meta = {"bytes": len(body), "md5": hashlib.md5(body).hexdigest(), "width": None, "height": None, "format": None}
    try:
        with Image.open(BytesIO(body)) as im:
            meta.update(width=im.width, height=im.height, format=im.format)
    except Exception:
        pass  # not a readable image: keep the bytes and md5, leave dimensions empty
    return meta


def download_images(s3, bucket: str, df: pd.DataFrame, out: Path, workers: int) -> pd.DataFrame:
    """Download front and back of every customer in df. Returns one row per customer with
    <side>_path, <side>_width, <side>_height, <side>_format, <side>_bytes, <side>_md5, <side>_error."""
    tasks = []
    for r in df.itertuples(index=False):
        for side in SIDES:
            dest = out / "images" / r.tenant / r.label.lower() / f"{r.anon_id}_{side}.png"
            tasks.append((r.anon_id, side, getattr(r, f"{side}_id_image"), dest))

    def work(task):
        aid, side, key, dest = task
        try:
            meta = fetch_image(s3, bucket, key, dest)
            return aid, side, {"path": dest.relative_to(out).as_posix(), **meta}
        except Exception as e:  # deleted object, permissions, etc.
            return aid, side, {"error": str(e)[:120]}

    if not tasks:
        return pd.DataFrame(columns=["anon_id"])
    result: dict[str, dict] = {r.anon_id: {"anon_id": r.anon_id} for r in df.itertuples(index=False)}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for aid, side, meta in tqdm(pool.map(work, tasks), total=len(tasks), desc="   images"):
            result[aid].update({f"{side}_{k}": v for k, v in meta.items()})
    return pd.DataFrame(list(result.values()))


def merge_with_existing(path: Path, new: pd.DataFrame, tenants: list[str]) -> pd.DataFrame:
    """When only some tenants were processed, keep the rows of the other tenants already in the file."""
    if path.exists():
        old = pd.read_csv(path, dtype=str)
        old = old[~old["tenant"].isin(tenants)]
        if not old.empty:
            return pd.concat([old, new.astype(object)], ignore_index=True)
    return new


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-images", action="store_true", help="manifest + OCR only, no S3 downloads")
    ap.add_argument("--tenant", help="process a single tenant")
    ap.add_argument("--refresh-ocr", action="store_true", help="fetch the OCR again even if the file exists")
    ap.add_argument("--workers", type=int, default=8, help="parallel S3 downloads")
    args = ap.parse_args()

    cfg = load_config()
    out = output_dir(cfg)
    ocr_dir = out / "ocr"
    ocr_dir.mkdir(parents=True, exist_ok=True)
    (out / "private").mkdir(parents=True, exist_ok=True)
    s3 = None if args.no_images else s3_client()
    tenants = [t for t in cfg["tenants"] if not args.tenant or t == args.tenant]
    if not tenants:
        sys.exit(f"tenant '{args.tenant}' is not in config.yaml")

    manifests, id_maps, common = [], [], None
    for tenant in tenants:
        print(f"== {tenant}: querying database (read-only)")
        with closing(pg_connection(tenant)) as con:
            if common is None:
                try:
                    common = fetch_common_fraud_status(con)
                except Exception as e:
                    print(f"   WARNING: could not read the cross-brand fraud flag ({str(e).strip()[:100]})")
                    common = pd.DataFrame(columns=["document_number", "common_fraud_status"])
            clients = label_clients(fetch_clients(con, cfg["registered_since"]), tenant, cfg)
            df = clients[clients["label"].notna()].copy()
            df["tenant"] = tenant
            df["anon_id"] = [anon_id(tenant, d) for d in df["document_number"]]
            df["person_id"] = [person_id(d) for d in df["document_number"]]
            df = df.merge(common, on="document_number", how="left")
            with_images = set(df.loc[df["label"] == POSITIVE, "client_id"]) | \
                sample_negatives(df, cfg["negatives_with_images"].get(tenant, 0), cfg["seed"])
            df["has_images"] = df["client_id"].isin(with_images)
            print(f"   positives={int((df['label'] == POSITIVE).sum())}  negatives={int((df['label'] != POSITIVE).sum())}"
                  f"  selected for images={len(with_images)}")
            export_ocr(con, df, ocr_dir, args.refresh_ocr)

        df = df.merge(ocr_flags(ocr_dir, df["anon_id"]), on="anon_id", how="left")
        if s3 is not None:
            images = download_images(s3, cfg["bucket"], df[df["has_images"]], out, args.workers)
            df = df.merge(images, on="anon_id", how="left")
            errors = int(df.filter(regex="_error$").notna().any(axis=1).sum()) if len(images) else 0
            print(f"   images: {len(images)} customers, {errors} with a download error")

        id_maps.append(df[["anon_id", "person_id", "tenant", "client_id", "document_number"]])
        manifests.append(df.drop(columns=["client_id", "document_number", "front_id_image", "back_id_image",
                                          "group", "has_both_images", "fraud_status"]))

    manifest = pd.concat(manifests, ignore_index=True)
    for side in SIDES:  # columns expected by the next scripts, even when no image was downloaded
        if f"{side}_path" not in manifest:
            manifest[f"{side}_path"] = None
    first = ["anon_id", "person_id", "tenant", "label", "label_source", "registered_at", "registered_date",
             "channel", "has_images", "front_path", "back_path"]
    manifest = manifest[first + [c for c in manifest.columns if c not in first]]
    for col in manifest.columns:
        if col.endswith(("_bytes", "_width", "_height")):
            manifest[col] = pd.to_numeric(manifest[col]).astype("Int64")

    manifest = merge_with_existing(out / "manifest.csv", manifest, tenants)
    manifest.to_csv(out / "manifest.csv", index=False)
    merge_with_existing(out / "private" / "id_map.csv", pd.concat(id_maps, ignore_index=True), tenants) \
        .to_csv(out / "private" / "id_map.csv", index=False)

    print("\nSummary:")
    print(manifest.groupby(["tenant", "label", "label_source"]).size().to_string())
    print(f"\nmanifest: {out / 'manifest.csv'}  ({len(manifest)} rows)")
    print("Reminder: data/ is git-ignored. private/id_map.csv holds real document numbers; never share it.")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as e:  # missing STAGE / ANON_SALT, expired SSO session, unreadable secret
        sys.exit(f"ERROR: {e}")
