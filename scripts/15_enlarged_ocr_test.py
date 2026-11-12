"""Phase 1 · Test: does Rekognition read the back of the card better when it gets only the card,
enlarged, instead of the whole photo?

Run it after `14_fresh_ocr.py --run --keep-images`. For a sample of people whose back photo is
already on disk it cuts out the area where Rekognition found text the first time (the card),
enlarges it and sends it again. The answer is saved in data/<stage>/ocr_enlarged_test/ and
compared with the first reading of the same photo: is the back code readable now?

Nothing is read from S3: only photos already on disk are used. Rekognition runs on the account
named in .env (REKOGNITION_PROFILE and REKOGNITION_ACCOUNT_ID), checked before anything else.

Without --run it only reports what it would send and compares what was already tested. It can be
interrupted and re-run: a person already answered is not sent (or paid) again.

Usage:  python scripts/15_enlarged_ocr_test.py                    report only, no cost
        python scripts/15_enlarged_ocr_test.py --run [--people 200] [--scale 2]
"""
import argparse
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import pandas as pd
from tqdm import tqdm

from id_fraud_detector import ocr_store
from id_fraud_detector.common import image_path, load_config, output_dir, rekognition_client
from id_fraud_detector.fresh_ocr import (PRICE_PER_IMAGE_USD, enlarged_text_area, fit_for_rekognition, load,
                                         text_detections)
from id_fraud_detector.rules.ocr_parse import ocr_lines, parse_back_code, parse_front_numbers, parse_serial
from id_fraud_detector.rules.registrars import detect_registrar


def candidates(out) -> pd.DataFrame:
    """People with a first reading of the back that found text and with the back photo on disk."""
    manifest = pd.read_csv(out / "manifest.csv", usecols=["anon_id", "tenant", "label", "channel"])
    rows = []
    for r in manifest.itertuples(index=False):
        payload = load(out / "ocr_fresh", r.anon_id)
        photo = image_path(out, r.tenant, r.label, r.anon_id, "back")
        if payload and (payload.get("back") or {}).get("ocr_at") and ocr_store.lines(payload, "back") and photo.exists():
            rows.append({"anon_id": r.anon_id, "label": r.label, "channel": r.channel, "photo": photo})
    return pd.DataFrame(rows, columns=["anon_id", "label", "channel", "photo"])


def run_one(rekognition, fresh_dir, test_dir, anon_id: str, photo, scale: float) -> str | None:
    """Send the enlarged card of one person and save the answer. Returns the error, if any."""
    info = {"sent_as": None, "ocr_at": None, "error": None}
    detections = []
    try:
        enlarged = enlarged_text_area(photo.read_bytes(), ocr_store.lines(ocr_store.read(fresh_dir, anon_id), "back"),
                                      scale=scale)
        if enlarged is None:
            raise ValueError("no text area to cut out")
        body, area = enlarged
        info.update(area)
        payload, info["sent_as"] = fit_for_rekognition(body)
        detections = text_detections(rekognition.detect_text(Image={"Bytes": payload}))
        info["ocr_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    except Exception as e:
        info["error"] = f"{type(e).__name__}: {str(e)[:150]}"
    ocr_store.write(test_dir, anon_id, {"data_back_id": {"detected_text": {"TextDetections": detections}},
                                        "back": info})
    return info["error"]


def reading(back: dict, front_numbers: list[str]) -> dict:
    """What the rules can use from one reading of the back."""
    text = " ".join(ocr_lines(back))
    code = parse_back_code(text)
    return {"code": code is not None,
            "code_equals_front": bool(code and code.number in front_numbers),
            "code_any_confidence": parse_back_code(" ".join(ocr_lines(back, min_confidence=0))) is not None,
            "serial": parse_serial(text) is not None,
            "registrar": detect_registrar(text.upper()) is not None}


def compare(out, people: pd.DataFrame) -> None:
    rows = []
    for r in people.itertuples(index=False):
        test = load(out / "ocr_enlarged_test", r.anon_id)
        if not test or not (test.get("back") or {}).get("ocr_at"):
            continue
        first = ocr_store.read(out / "ocr_fresh", r.anon_id)
        numbers = [n.zfill(10) for n in parse_front_numbers(" ".join(ocr_lines(first["data_front_id"])))]
        whole, enlarged = reading(first["data_back_id"], numbers), reading(test["data_back_id"], numbers)
        rows.append({"label": r.label, "channel": r.channel, "front_read": bool(numbers),
                     **{f"{k}_whole": v for k, v in whole.items()}, **{f"{k}_enlarged": v for k, v in enlarged.items()}})
    if not rows:
        print("\nNobody tested yet.")
        return
    df = pd.DataFrame(rows)
    df["gained"] = df["code_enlarged"] & ~df["code_whole"]
    df["lost"] = df["code_whole"] & ~df["code_enlarged"]

    def table(by) -> pd.DataFrame:
        return df.groupby(by, dropna=False).agg(n=("code_whole", "size"), code_whole_photo=("code_whole", "mean"),
                                  code_enlarged=("code_enlarged", "mean"), gained=("gained", "sum"),
                                  lost=("lost", "sum")).round(3)

    print(f"\n== Back code readable (confidence >= 80): whole photo vs enlarged card   people tested: {len(df)}")
    print(table("channel").to_string())
    print(table("label").to_string())
    print("\n== Everything the rules use from the back (share of the people tested)")
    names = ["code", "code_any_confidence", "serial", "registrar"]
    print(pd.DataFrame({"whole_photo": [df[f"{n}_whole"].mean() for n in names],
                        "enlarged": [df[f"{n}_enlarged"].mean() for n in names]}, index=names).round(3).to_string())
    print("\n== Is the code read right? Code number equal to the number on the front "
          "(people with a front number and a code)")
    for kind in ("whole", "enlarged"):
        with_code = df[df["front_read"] & df[f"code_{kind}"]]
        print(f"   {kind:9s} {int(with_code[f'code_equals_front_{kind}'].sum())} of {len(with_code)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true", help="send the enlarged cards to Rekognition (this has a cost)")
    ap.add_argument("--people", type=int, default=200, help="people to test, taken at random (always the same ones)")
    ap.add_argument("--scale", type=float, default=2.0, help="how many times the card is enlarged")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    print("== accounts")
    rekognition = rekognition_client()  # stops here unless the paying account was named and is not the corporate one

    cfg = load_config()
    out = output_dir(cfg)
    test_dir = out / "ocr_enlarged_test"
    test_dir.mkdir(parents=True, exist_ok=True)

    everyone = candidates(out)
    if everyone.empty:
        sys.exit("ERROR: nobody has both a first reading and the back photo on disk. "
                 "Run `python scripts/14_fresh_ocr.py --run --keep-images` first.")
    people = everyone.sort_values("anon_id").sample(min(args.people, len(everyone)), random_state=cfg["seed"])
    pending = people[[not ((load(test_dir, a) or {}).get("back") or {}).get("ocr_at") for a in people["anon_id"]]]
    print(f"people with the back photo on disk and a first reading: {len(everyone)}   in this test: {len(people)}")
    print(f"to send now: {len(pending)} enlarged cards   estimated cost: {len(pending) * PRICE_PER_IMAGE_USD:.2f} USD")

    if not args.run:
        print("\nReport only: nothing was sent. Add --run to send them.")
    elif not pending.empty:
        errors: dict[str, int] = {}
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            results = pool.map(lambda r: run_one(rekognition, out / "ocr_fresh", test_dir, r.anon_id, r.photo,
                                                 args.scale), pending.itertuples(index=False))
            for error in tqdm(results, total=len(pending), desc="   people"):
                if error:
                    kind = error.split(":", 1)[0]
                    errors[kind] = errors.get(kind, 0) + 1
        sent = len(pending) - sum(errors.values())
        print(f"\nenlarged cards read by Rekognition in this run: {sent}   approximate cost: "
              f"{sent * PRICE_PER_IMAGE_USD:.2f} USD")
        for kind, n in sorted(errors.items(), key=lambda item: -item[1]):
            print(f"   not read ({kind}): {n}")

    compare(out, people)


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as e:  # missing STAGE / profile / account, expired SSO session, wrong account
        sys.exit(f"ERROR: {e}")
