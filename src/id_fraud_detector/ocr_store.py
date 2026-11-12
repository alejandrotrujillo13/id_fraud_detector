"""Local store of the Rekognition text of each customer: <output>/ocr/<anon_id>.json.

The file mirrors the production columns and records where each side's text came from:

  {"data_front_id": {"detected_text": {"TextDetections": [...]}},
   "data_back_id":  {"detected_text": {"TextDetections": [...]}},
   "validation_updated_at": "...",       last change of the customer's own validation row
   "front_ocr_from": "esmio" | null,     tenant whose validation produced the text (null = no text)
   "back_ocr_from":  "llevaloya" | null,
   "borrow_checked": ["llevaloya"]}      tenants already searched for a missing side

Why text is borrowed: a person already validated in one brand is not asked for photos again in
another; mikro_back copies the stored photos and Rekognition does not run, so that customer has
the images but no text. The text read from those very images is in the tenant where the photos
were taken. It is only reused when the image of that side is identical in both tenants.
"""
import json
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from .common import s3_fingerprint
from .dataset import fetch_clients_by_document, fetch_ocr_lines, stored_ocr

SIDES = ("front", "back")


def read(ocr_dir: Path, anon_id: str) -> dict:
    return json.loads((ocr_dir / f"{anon_id}.json").read_text(encoding="utf-8"))


def write(ocr_dir: Path, anon_id: str, payload: dict) -> None:
    text = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))  # compact: these files add up
    (ocr_dir / f"{anon_id}.json").write_text(text, encoding="utf-8")


def lines(payload: dict, side: str) -> list:
    return payload[f"data_{side}_id"]["detected_text"]["TextDetections"]


def export_ocr(con, tenant: str, df: pd.DataFrame, ocr_dir: Path, refresh: bool = False) -> None:
    """Write the file of every customer in df that does not have one yet (all of them with
    `refresh`). df needs client_id and anon_id; `con` is a connection to `tenant`."""
    pending = df if refresh else df[[not (ocr_dir / f"{a}.json").exists() for a in df["anon_id"]]]
    if pending.empty:
        print("   OCR: every file already on disk, nothing to fetch")
        return
    anon_by_client = dict(zip(pending["client_id"], pending["anon_id"]))

    def save(client_id, row: dict | None) -> None:  # row None: the customer has no validation row
        payload = stored_ocr(row["front_lines"] if row else None, row["back_lines"] if row else None)
        payload["validation_updated_at"] = str(row["validation_updated_at"]) if row else None
        for side in SIDES:
            payload[f"{side}_ocr_from"] = tenant if lines(payload, side) else None
        payload["borrow_checked"] = []
        write(ocr_dir, anon_by_client[client_id], payload)

    seen = set()
    with tqdm(total=len(pending), desc="   OCR") as bar:
        for row in fetch_ocr_lines(con, pending["client_id"]):  # streamed: one chunk in memory at a time
            save(row["client_id"], row)
            seen.add(row["client_id"])
            bar.update(1)
        for client_id in anon_by_client:
            if client_id not in seen:
                save(client_id, None)
                bar.update(1)


def borrow_ocr(tenant: str, df: pd.DataFrame, other_tenant: str, ocr_dir: Path,
               connect: Callable, s3, bucket: str, workers: int = 8) -> dict:
    """Fill the missing sides of `tenant` customers with the text stored in `other_tenant` for
    the same document number, only where the image of that side is identical in both tenants.

    df needs anon_id, document_number, front_id_image and back_id_image. `connect(tenant)` opens
    a read-only connection. Each customer is searched once per other tenant (borrow_checked).
    Returns counters for the summary.
    """
    stats = dict.fromkeys(("customers_missing_text", "found_in_other_tenant", "front_borrowed", "back_borrowed",
                           "front_different_image", "back_different_image"), 0)
    pending = []
    for r in df.itertuples(index=False):
        payload = read(ocr_dir, r.anon_id)
        missing = [side for side in SIDES if not lines(payload, side)]
        if missing and other_tenant not in payload.get("borrow_checked", []):
            pending.append((r, payload, missing))
    stats["customers_missing_text"] = len(pending)
    if not pending:
        return stats

    with closing(connect(other_tenant)) as con:
        others = fetch_clients_by_document(con, [r.document_number for r, _, _ in pending])
        other_ocr = {row["client_id"]: row for row in fetch_ocr_lines(con, others["client_id"])}
    other_by_document = {o.document_number: o for o in others.itertuples(index=False)}

    candidates = []  # (position in pending, side, own image key, other image key, text lines)
    for position, (r, _, missing) in enumerate(pending):
        other = other_by_document.get(r.document_number)
        if other is None:
            continue
        stats["found_in_other_tenant"] += 1
        row = other_ocr.get(other.client_id)
        for side in missing:
            text = row[f"{side}_lines"] if row else None
            if text:
                candidates.append((position, side, getattr(r, f"{side}_id_image"),
                                   getattr(other, f"{side}_id_image"), text))

    keys = sorted({key for candidate in candidates for key in candidate[2:4]})
    with ThreadPoolExecutor(max_workers=workers) as pool:
        fingerprints = dict(zip(keys, pool.map(lambda key: s3_fingerprint(s3, bucket, key), keys)))

    for position, side, own_key, other_key, text in candidates:
        payload = pending[position][1]
        if fingerprints[own_key] is not None and fingerprints[own_key] == fingerprints[other_key]:
            payload[f"data_{side}_id"]["detected_text"]["TextDetections"] = text
            payload[f"{side}_ocr_from"] = other_tenant
            stats[f"{side}_borrowed"] += 1
        else:
            stats[f"{side}_different_image"] += 1

    for r, payload, _ in pending:
        payload["borrow_checked"] = sorted({*payload.get("borrow_checked", []), other_tenant})
        write(ocr_dir, r.anon_id, payload)
    return stats


def ocr_flags(ocr_dir: Path, anon_ids) -> pd.DataFrame:
    """Per customer: has_<side>_ocr, <side>_ocr_from and validation_updated_at, read from the files."""
    rows = []
    for anon in anon_ids:
        payload = read(ocr_dir, anon)
        row = {"anon_id": anon, "validation_updated_at": payload.get("validation_updated_at")}
        for side in SIDES:
            row[f"has_{side}_ocr"] = bool(lines(payload, side))
            row[f"{side}_ocr_from"] = payload.get(f"{side}_ocr_from")
        rows.append(row)
    return pd.DataFrame(rows, columns=["anon_id", "has_front_ocr", "has_back_ocr", "front_ocr_from",
                                       "back_ocr_from", "validation_updated_at"])
