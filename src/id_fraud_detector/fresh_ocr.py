"""Fresh OCR: Rekognition detect_text run again on the photos of each person.

The text stored in production covers a minority of the customers and cannot always be tied to
the photo the customer has today. Here every photo is read (from disk when it is already there,
from S3 otherwise), sent to Rekognition and the text is kept in <output>/ocr_fresh/<anon_id>.json.
The photo is written to disk only when asked (`keep`), in the same place 11_export_dataset.py
uses, so it is downloaded once for everything. The file uses the nesting of the production
columns, so the rule parser reads it as is, and records which image produced each text.
Everything Rekognition detected is kept as it arrived (LINE and WORD detections, with Id,
ParentId, Confidence and Geometry), so nothing has to be paid for twice:

  {"data_front_id": {"detected_text": {"TextDetections": [every detection of the response]}},
   "data_back_id":  {"detected_text": {"TextDetections": [...]}},
   "front": {"image_md5": "...", "image_bytes": 123456, "sent_as": "original", "ocr_at": "...",
             "model_version": "3.0", "error": null},
   "back":  {...}}

`ocr_at` is set when Rekognition answered (even with no text); a side with an error has no
`ocr_at` and is tried again on the next run.
"""
import hashlib
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path

from PIL import Image

from . import ocr_store
from .dataset import stored_ocr
from .ocr_store import SIDES

MAX_BYTES = 5 * 1024 * 1024  # Rekognition limit for an image sent as bytes
PRICE_PER_IMAGE_USD = 0.001  # detect_text, first million images of the month (checked 2026-11-11)


def fit_for_rekognition(body: bytes) -> tuple[bytes, str]:
    """Bytes to send and how they were sent. Images over the 5 MB limit are re-encoded as JPEG
    in memory, at the highest quality that fits."""
    if len(body) <= MAX_BYTES:
        return body, "original"
    with Image.open(BytesIO(body)) as image:
        rgb = image.convert("RGB")
    for quality in (95, 85, 75, 60):
        buffer = BytesIO()
        rgb.save(buffer, "JPEG", quality=quality)
        if buffer.tell() <= MAX_BYTES:
            return buffer.getvalue(), f"jpeg_{quality}"
    raise ValueError("image over the Rekognition size limit even as JPEG")


def text_detections(response: dict) -> list[dict]:
    """Every detection of a detect_text response (LINE and WORD), exactly as Rekognition sent it."""
    return list(response.get("TextDetections") or [])


def enlarged_text_area(body: bytes, detections: list[dict], margin: float = 0.06, scale: float = 2.0,
                       max_side: int = 4000) -> tuple[bytes, dict] | None:
    """The part of the photo where Rekognition found text (in practice, the card), cut out with
    a margin and enlarged. Used to test whether small text is read better that way. Returns the
    PNG bytes and {"box": [left, top, right, bottom] in pixels, "scale", "size"}, or None when
    the photo has no text to locate."""
    boxes = [d["Geometry"]["BoundingBox"] for d in detections
             if d.get("Type") == "LINE" and (d.get("Geometry") or {}).get("BoundingBox")]
    if not boxes:
        return None
    with Image.open(BytesIO(body)) as image:
        rgb = image.convert("RGB")
    width, height = rgb.size
    box = [round(max(0.0, min(b["Left"] for b in boxes) - margin) * width),
           round(max(0.0, min(b["Top"] for b in boxes) - margin) * height),
           round(min(1.0, max(b["Left"] + b["Width"] for b in boxes) + margin) * width),
           round(min(1.0, max(b["Top"] + b["Height"] for b in boxes) + margin) * height)]
    if box[2] - box[0] < 2 or box[3] - box[1] < 2:
        return None
    area = rgb.crop(box)
    factor = max(1.0, min(scale, max_side / max(area.size)))
    if factor > 1:
        area = area.resize((round(area.width * factor), round(area.height * factor)), Image.LANCZOS)
    buffer = BytesIO()
    area.save(buffer, "PNG")
    return buffer.getvalue(), {"box": box, "scale": round(factor, 3), "size": list(area.size)}


def photo_bytes(s3, bucket: str, key: str, path: Path | None, keep: bool) -> bytes:
    """The photo: from disk when it is already there, from S3 otherwise. With `keep` the
    downloaded photo is also written to `path`, so it never has to be downloaded again."""
    if path is not None and path.exists() and path.stat().st_size > 0:
        return path.read_bytes()
    body = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
    if keep and path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_name(path.name + ".part")
        partial.write_bytes(body)
        partial.replace(path)
    return body


def read_side(s3, rekognition, bucket: str, key: str, path: Path | None = None,
              keep: bool = False) -> tuple[list[dict], dict]:
    """Get one photo (see photo_bytes) and run detect_text on it. Returns (detections, info). It
    never raises: a failure is recorded in info["error"] and leaves info["ocr_at"] empty."""
    info = {"image_md5": None, "image_bytes": None, "sent_as": None, "ocr_at": None,
            "model_version": None, "error": None}
    try:
        body = photo_bytes(s3, bucket, key, path, keep)
        info.update(image_md5=hashlib.md5(body).hexdigest(), image_bytes=len(body))
        payload, info["sent_as"] = fit_for_rekognition(body)
        response = rekognition.detect_text(Image={"Bytes": payload})
        detections = text_detections(response)
        info.update(ocr_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    model_version=response.get("TextModelVersion"))
        return detections, info
    except Exception as e:  # missing object, unreadable image, throttling after the retries, expired session
        info["error"] = f"{type(e).__name__}: {str(e)[:150]}"
        return [], info


def load(ocr_dir: Path, anon_id: str) -> dict | None:
    """The person's file, or None when it does not exist or was cut short by an interrupted run."""
    try:
        return ocr_store.read(ocr_dir, anon_id)
    except (FileNotFoundError, ValueError):
        return None


def pending_sides(ocr_dir: Path, anon_id: str) -> list[str]:
    """Sides Rekognition has not answered for yet."""
    payload = load(ocr_dir, anon_id) or {}
    return [side for side in SIDES if not (payload.get(side) or {}).get("ocr_at")]


def run_person(s3, rekognition, bucket: str, ocr_dir: Path, anon_id: str, keys: dict[str, str],
               paths: dict[str, Path] | None = None, keep: bool = False) -> dict:
    """Run the pending sides of one person and save the file. `keys` maps side -> S3 key and
    `paths` side -> where the photo is (or, with `keep`, will be) on disk.
    Returns {side: info} for the sides that were tried; info["error"] is None when it worked."""
    payload = load(ocr_dir, anon_id) or stored_ocr(None, None)
    tried = {}
    for side in SIDES:
        if (payload.get(side) or {}).get("ocr_at"):
            continue
        detections, info = read_side(s3, rekognition, bucket, keys[side], (paths or {}).get(side), keep)
        payload[f"data_{side}_id"]["detected_text"]["TextDetections"] = detections
        payload[side] = info
        tried[side] = info
    ocr_store.write(ocr_dir, anon_id, payload)
    return tried


def negatives_subset(people, n: int, seed: int) -> set:
    """anon_ids of `n` negatives for a partial run, taken at random among the negatives the
    manifest already selects for images (`has_images`: the stratified sample by month and
    channel). The seed is fixed and the candidates are sorted first, so every run picks the
    same people and an interrupted run can be resumed."""
    candidates = people[(people["label"] == "NEGATIVE") & people["has_images"].astype(bool)].sort_values("anon_id")
    return set(candidates.sample(min(n, len(candidates)), random_state=seed)["anon_id"])


def flags(ocr_dir: Path, anon_id: str) -> dict:
    """Per side: was it read by Rekognition, and did it find any text."""
    payload = load(ocr_dir, anon_id)
    row = {}
    for side in SIDES:
        row[f"{side}_done"] = bool(payload) and bool((payload.get(side) or {}).get("ocr_at"))
        row[f"{side}_text"] = bool(payload) and bool(ocr_store.lines(payload, side))
    return row
