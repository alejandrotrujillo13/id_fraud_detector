"""Phase 1 · Fresh OCR: Rekognition detect_text on the photos of every person of the manifest.

Run it after 11_export_dataset.py. For each person it gets front and back (from disk when the
photo is already there, otherwise from S3 with the corporate SSO session, GetObject only), sends
them to Rekognition on ANOTHER AWS account and saves the text in
data/<stage>/ocr_fresh/<anon_id>.json. That account has to be named in .env, with no default:
REKOGNITION_PROFILE (its AWS profile) and REKOGNITION_ACCOUNT_ID (its account number). The
script checks it first and stops when either is missing, when the profile belongs to another
account or when it is the account of the SSO session. Every person is processed, with or without
stored text, so the text is known to come from the photo the customer has today.

Photos are handled a few at a time (--workers people in parallel), never all at once. By default
they are not written to disk; with --keep-images each downloaded photo is saved where
11_export_dataset.py keeps its images, so no later step has to download it again.

Without --run it only reports: it checks the two accounts, counts what is pending and estimates
the cost. Nothing is sent to Rekognition.

It can be interrupted and re-run: the text of each person is saved as soon as it arrives, sides
already answered are not sent (or paid) again, and sides that failed are tried again. Positives
go first.

With --negatives N it works only on every positive and N negatives, always the same ones: taken
at random (fixed seed) among the negatives the manifest selects for images.

Usage:  python scripts/14_fresh_ocr.py                      report only, no cost
        python scripts/14_fresh_ocr.py --run --limit 20     trial with 20 people
        python scripts/14_fresh_ocr.py --run --negatives 300 --keep-images   every positive and 300 negatives
        python scripts/14_fresh_ocr.py --run [--keep-images] [--workers 8]   everything that is pending
"""
import argparse
import sys
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
from tqdm import tqdm

from id_fraud_detector.common import image_path, load_config, output_dir, rekognition_client, s3_client
from id_fraud_detector.dataset import POSITIVE
from id_fraud_detector.fresh_ocr import PRICE_PER_IMAGE_USD, flags, negatives_subset, pending_sides, run_person
from id_fraud_detector.ocr_store import SIDES

STOP_AFTER_FAILURES = 25  # people in a row with an error: expired session, wrong permissions, ...


def people_with_keys(out) -> pd.DataFrame:
    """Manifest rows with the S3 keys of their photos (kept in private/id_map.csv)."""
    manifest = pd.read_csv(out / "manifest.csv",
                           usecols=["anon_id", "tenant", "label", "label_source", "has_images"])
    id_map = pd.read_csv(out / "private" / "id_map.csv", dtype=str)
    keys = [f"{side}_id_image" for side in SIDES]
    if not set(keys) <= set(id_map.columns):
        raise RuntimeError("private/id_map.csv has no image keys. Run `python scripts/11_export_dataset.py "
                           "--no-images` again and retry.")
    people = manifest.merge(id_map[["anon_id", *keys]], on="anon_id", how="left", validate="one_to_one")
    if people[keys].isna().any().any():
        raise RuntimeError("manifest.csv and private/id_map.csv do not match. Run 11_export_dataset.py again.")
    return people.sort_values("label", ascending=False, kind="stable").reset_index(drop=True)  # POSITIVE first


def summary(people: pd.DataFrame, ocr_dir) -> pd.DataFrame:
    done = pd.DataFrame([flags(ocr_dir, anon) for anon in people["anon_id"]], index=people.index)
    table = pd.concat([people[["tenant", "label"]], done], axis=1)
    return table.groupby(["tenant", "label"]).agg(
        people=("front_done", "size"), front_done=("front_done", "sum"), back_done=("back_done", "sum"),
        front_with_text=("front_text", "sum"), back_with_text=("back_text", "sum"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true", help="send the pending photos to Rekognition (this has a cost)")
    ap.add_argument("--limit", type=int, help="process at most this many people (positives first)")
    ap.add_argument("--negatives", type=int,
                    help="partial run: every positive and only this many negatives, taken at random (always the "
                         "same ones) among those the manifest selects for images")
    ap.add_argument("--keep-images", action="store_true",
                    help="also save each downloaded photo in data/<stage>/images/, so it is never downloaded again")
    ap.add_argument("--workers", type=int, default=8, help="people processed in parallel")
    args = ap.parse_args()

    # Before anything else: which account pays. It stops here when the paying account was not
    # named in .env, when the profile is of another account or when it is the corporate one.
    print("== accounts")
    rekognition = rekognition_client()

    cfg = load_config()
    out = output_dir(cfg)
    ocr_dir = out / "ocr_fresh"
    ocr_dir.mkdir(parents=True, exist_ok=True)

    people = people_with_keys(out)
    people["pending"] = [len(pending_sides(ocr_dir, anon)) for anon in people["anon_id"]]
    scope = people
    if args.negatives is not None:
        chosen = negatives_subset(people, args.negatives, cfg["seed"])
        scope = people[(people["label"] == POSITIVE) | people["anon_id"].isin(chosen)]
    todo = scope[scope["pending"] > 0]
    if args.limit is not None:
        todo = todo.head(args.limit)
    images = int(todo["pending"].sum())
    print(f"people in the manifest: {len(people)}   photos: {len(people) * len(SIDES)}")
    if args.negatives is not None:
        print(f"partial run: {int((scope['label'] == POSITIVE).sum())} positives and {len(chosen)} negatives "
              f"taken at random among the {int((people['has_images'] & (people['label'] != POSITIVE)).sum())} "
              f"selected for images (seed {cfg['seed']})")
    print(f"already read by Rekognition: {len(people) * len(SIDES) - int(people['pending'].sum())} photos")
    print(f"to send now: {images} photos of {len(todo)} people"
          f"   estimated cost: {images * PRICE_PER_IMAGE_USD:.2f} USD ({PRICE_PER_IMAGE_USD} USD per photo)")

    if not args.run:
        print("\nReport only: nothing was sent. Add --run to send the photos (try --run --limit 20 first).")
        return
    if todo.empty:
        print("\nNothing pending.")
    else:
        s3 = s3_client()
        tasks = [(r.anon_id, {side: getattr(r, f"{side}_id_image") for side in SIDES},
                  {side: image_path(out, r.tenant, r.label, r.anon_id, side) for side in SIDES})
                 for r in todo.itertuples(index=False)]
        sent = failed = in_a_row = total_bytes = 0
        errors: dict[str, int] = {}
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            results = pool.map(lambda task: run_person(s3, rekognition, cfg["bucket"], ocr_dir, *task,
                                                       keep=args.keep_images), tasks)
            for tried in tqdm(results, total=len(tasks), desc="   people"):
                bad = [info["error"] for info in tried.values() if info["error"]]
                sent += len(tried) - len(bad)
                failed += len(bad)
                total_bytes += sum(info["image_bytes"] or 0 for info in tried.values() if not info["error"])
                in_a_row = in_a_row + 1 if bad else 0
                for error in bad:
                    kind = error.split(":", 1)[0]
                    errors[kind] = errors.get(kind, 0) + 1
                if in_a_row >= STOP_AFTER_FAILURES:
                    pool.shutdown(wait=False, cancel_futures=True)
                    print(f"\nStopped: {in_a_row} people in a row failed. Last error: {bad[-1]}")
                    break
        print(f"\nphotos read by Rekognition in this run: {sent}   approximate cost: {sent * PRICE_PER_IMAGE_USD:.2f} USD")
        if sent:
            average = total_bytes / sent
            print(f"average photo size: {average / 1e6:.2f} MB   every photo of the manifest would take about "
                  f"{average * len(people) * len(SIDES) / 1e9:.1f} GB on disk"
                  + ("   (photos kept in data/<stage>/images/)" if args.keep_images else "   (photos were not kept)"))
        if failed:
            print(f"photos with an error (not read; they are tried again on the next run): {failed}")
            for kind, n in sorted(errors.items(), key=lambda item: -item[1]):
                print(f"   {kind}: {n}")

    print("\nFresh OCR so far:")
    print(summary(people, ocr_dir).to_string())
    print(f"\ntext files: {ocr_dir}")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as e:  # missing STAGE / profile, expired SSO session, wrong account
        sys.exit(f"ERROR: {e}")
