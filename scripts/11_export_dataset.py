"""Phase 1 · Dataset construction.

For each tenant it reads (read-only) users_client, credits_credit and
users_customeridvalidationresult, applies the labeling rules from config.yaml, keeps one row
per person across tenants and writes the files below. Besides fraud_status, a customer is a
positive when a source of `confirmed_fraud_sources` (config.yaml) matches, or when an analyst
confirmed the ID as forged in the team's verifier (history saved under
data/<stage>/private/verifier_history/casos/, if present); always with no PAID credit.

  data/<stage>/manifest.csv              one row per person (positives and ALL negatives), anonymized
  data/<stage>/ocr/<anon_id>.json        Rekognition text lines (front and back) to measure the rules
                                 (format and origin of the text: id_fraud_detector/ocr_store.py)
  data/<stage>/images/<tenant>/<label>/<anon_id>_front.png / _back.png
                                 every positive + a stratified sample of negatives
  data/<stage>/private/id_map.csv        anon_id -> document number and S3 keys of the photos
                                 (LOCAL ONLY, git-ignored; needed for the backtest and for 14_fresh_ocr.py)

One row per person: a document number labeled in several tenants is kept once, so the same ID
card is not repeated in the dataset (rule: id_fraud_detector.dataset.one_row_per_person). The
photos are the ones of the tenant of the kept row; `labeled_in` lists every tenant.

A customer validated earlier in another brand has copied photos and no text of their own.
For those, the text stored in the other configured tenant is reused, only when the image of
that side is identical in both tenants (checked with a one-byte read of each S3 object).

It can be re-run safely: OCR files and images already on disk are not fetched again.

Usage:  python scripts/11_export_dataset.py [--no-images] [--refresh-ocr] [--no-borrow-ocr] [--workers 8]
"""
import argparse
import hashlib
import sys
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from io import BytesIO
from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm import tqdm

from id_fraud_detector.common import (anon_id, image_path, load_config, output_dir, person_id, pg_connection,
                                      s3_client)
from id_fraud_detector.dataset import (COMMON_COLUMNS, POSITIVE, add_common_info, confirmed_customers,
                                       fetch_clients, fetch_common_clients, label_clients, one_row_per_person,
                                       sample_negatives)
from id_fraud_detector.ocr_store import SIDES, borrow_ocr, export_ocr, ocr_flags
from id_fraud_detector.verifier import forged_by_analyst


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
            dest = image_path(out, r.tenant, r.label, r.anon_id, side)
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


def counts(df: pd.DataFrame) -> str:
    positives = int((df["label"] == POSITIVE).sum())
    return f"positives={positives}  negatives={len(df) - positives}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-images", action="store_true", help="manifest + OCR only, no image downloads")
    ap.add_argument("--refresh-ocr", action="store_true", help="fetch the OCR again even if the file exists")
    ap.add_argument("--no-borrow-ocr", action="store_true",
                    help="do not reuse the text stored in the other tenant for identical images")
    ap.add_argument("--workers", type=int, default=8, help="parallel S3 downloads")
    args = ap.parse_args()

    cfg = load_config()
    out = output_dir(cfg)
    ocr_dir = out / "ocr"
    ocr_dir.mkdir(parents=True, exist_ok=True)
    (out / "private").mkdir(parents=True, exist_ok=True)
    tenants = cfg["tenants"]
    borrow = not args.no_borrow_ocr and len(tenants) > 1
    s3 = s3_client() if borrow or not args.no_images else None

    # 1. label the customers of every tenant
    forged = forged_by_analyst(out)  # IDs an analyst confirmed as forged in the team's verifier (may be empty)
    print(f"verifier history: {len(forged)} ID numbers confirmed as forged by an analyst")
    labeled, common = [], None
    for tenant in tenants:
        print(f"== {tenant}: querying database (read-only)")
        with closing(pg_connection(tenant)) as con:
            if common is None:
                try:
                    common = fetch_common_clients(con)
                except Exception as e:
                    print(f"   WARNING: could not read the cross-brand record ({str(e).strip()[:100]})")
                    common = pd.DataFrame(columns=COMMON_COLUMNS)
            clients = fetch_clients(con, cfg["registered_since"])
            clients = label_clients(clients, tenant, cfg, confirmed_customers(con, tenant, cfg, clients, forged))
        df = add_common_info(clients[clients["label"].notna()], common)
        df["tenant"] = tenant
        print(f"   labeled customers: {counts(df)}")
        labeled.append(df)

    # 2. one row per person: a document number labeled in several tenants is kept once
    everyone = pd.concat(labeled, ignore_index=True)
    people = one_row_per_person(everyone, tenants)
    people["anon_id"] = [anon_id(t, d) for t, d in zip(people["tenant"], people["document_number"])]
    people["person_id"] = [person_id(d) for d in people["document_number"]]
    print(f"== one row per person: {len(everyone)} labeled customers -> {len(people)} people "
          f"({len(everyone) - len(people)} repeated rows removed, "
          f"{int(people['label_conflict'].sum())} people with conflicting labels)")

    # 3. choose who gets images and export the stored OCR of the kept rows
    frames: dict[str, pd.DataFrame] = {}
    for tenant in tenants:
        df = people[people["tenant"] == tenant].copy()
        with_images = set(df.loc[df["label"] == POSITIVE, "client_id"]) | \
            sample_negatives(df, cfg["negatives_with_images"].get(tenant, 0), cfg["seed"])
        df["has_images"] = df["client_id"].isin(with_images)
        print(f"== {tenant}: {counts(df)}  selected for images={len(with_images)}")
        if len(df):
            with closing(pg_connection(tenant)) as con:
                export_ocr(con, tenant, df, ocr_dir, args.refresh_ocr)
        frames[tenant] = df

    # 4. customers without text: reuse the text of the same person in the other tenant, identical images only
    if borrow:
        for tenant, df in frames.items():
            for other in tenants:
                if other == tenant or df.empty:
                    continue
                print(f"== {tenant}: looking in {other} for the text of customers that have none")
                stats = borrow_ocr(tenant, df, other, ocr_dir, pg_connection, s3, cfg["bucket"], args.workers)
                print("   " + "  ".join(f"{k}={v}" for k, v in stats.items()))

    # 5. manifest rows, and images when requested
    manifests, id_maps = [], []
    for tenant, df in frames.items():
        df = df.merge(ocr_flags(ocr_dir, df["anon_id"]), on="anon_id", how="left")
        if not args.no_images:
            images = download_images(s3, cfg["bucket"], df[df["has_images"]], out, args.workers)
            df = df.merge(images, on="anon_id", how="left")
            errors = int(df.filter(regex="_error$").notna().any(axis=1).sum()) if len(images) else 0
            print(f"   images: {len(images)} customers, {errors} with a download error")

        id_maps.append(df[["anon_id", "person_id", "tenant", "client_id", "document_number",
                           "front_id_image", "back_id_image"]])  # S3 keys: used by 14_fresh_ocr.py
        manifests.append(df.drop(columns=["client_id", "document_number", "front_id_image", "back_id_image",
                                          "group", "has_both_images", "fraud_status"]))

    manifest = pd.concat(manifests, ignore_index=True)
    for side in SIDES:  # columns expected by the next scripts, even when no image was downloaded
        if f"{side}_path" not in manifest:
            manifest[f"{side}_path"] = None
    first = ["anon_id", "person_id", "tenant", "label", "label_source", "labeled_in", "registered_at",
             "registered_date", "channel", "has_images", "front_path", "back_path"]
    manifest = manifest[first + [c for c in manifest.columns if c not in first]]
    for col in manifest.columns:
        if col.endswith(("_bytes", "_width", "_height")):
            manifest[col] = pd.to_numeric(manifest[col]).astype("Int64")

    manifest.to_csv(out / "manifest.csv", index=False)
    pd.concat(id_maps, ignore_index=True).to_csv(out / "private" / "id_map.csv", index=False)

    print("\nSummary (tenant = the one whose row and photos were kept):")
    print(manifest.groupby(["tenant", "label", "label_source"]).size().to_string())
    print("\nPeople per label:")
    print(manifest.groupby("label").agg(
        people=("anon_id", "size"),
        labeled_in_several_tenants=("labeled_in", lambda s: int(s.str.contains("|", regex=False).sum()))).to_string())
    text = manifest.assign(back_text=manifest["has_back_ocr"].astype(str) == "True",
                           back_text_borrowed=manifest["back_ocr_from"].notna()
                           & (manifest["back_ocr_from"] != manifest["tenant"]))
    print("\nStored back-side text available (borrowed = taken from the other tenant, identical image):")
    print(text.groupby(["tenant", "label"]).agg(people=("anon_id", "size"), back_text=("back_text", "sum"),
                                                back_text_borrowed=("back_text_borrowed", "sum")).to_string())
    print(f"\nmanifest: {out / 'manifest.csv'}  ({len(manifest)} rows)")
    print("Reminder: data/ is git-ignored. private/id_map.csv holds real document numbers; never share it.")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as e:  # missing STAGE / ANON_SALT, expired SSO session, unreadable secret
        sys.exit(f"ERROR: {e}")
