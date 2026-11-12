"""Phase 1 · OCR coverage and first validation of the structural rules.

For every row of the manifest (positives and all negatives) it parses the Rekognition OCR,
runs the rule engine and the cross-customer "repeated back" check, and reports:

  1. legibility of each field per tenant and label (coverage of the rules in practice)
  2. per rule: PASS / FAIL / NOT_EVALUABLE counts by label -> false-positive rate on negatives,
     detection rate on positives (same method as mikro_back/docs/fraud-score/01-problema.html)

Text source (--ocr):
  fresh    data/<stage>/ocr_fresh/, written by 14_fresh_ocr.py from the current photos (default).
           Only the people Rekognition has already read are measured, so it also works on a
           partial run.
  stored   data/<stage>/ocr/, the text kept in production (covers a minority of the customers)

Usage:  python scripts/12_ocr_coverage.py [--ocr fresh|stored]
Writes (under data/<stage>/; with --ocr stored the names end in _stored):
        rules_summary.csv              per rule and label: coverage, FAIL count and rates
        rules_per_customer.csv         per customer: which fields were readable + status of every rule
        private/rules_values.csv       per customer: the values read (ID numbers, serials). LOCAL ONLY
"""
import argparse
import json
import sys

import pandas as pd

from id_fraud_detector.common import load_config, output_dir
from id_fraud_detector.rules.engine import FAIL, NOT_EVALUABLE, PASS, evaluate
from id_fraud_detector.rules.ocr_parse import extract_back_fields, front_text, parse_front_number

SAME_BATCH_SERIAL_GAP = 50_000  # same sequence + same date + serials closer than this = same print batch


def repeated_back_rule(df: pd.DataFrame) -> pd.Series:
    """FAIL when sequence/serial/final appear under a different ID number (clone or template);
    same sequence AND same production date with nearby distinct serials is a normal print batch."""
    status = pd.Series(NOT_EVALUABLE, index=df.index)
    ok = df.code_sequence.notna()
    status[ok] = PASS
    # sequence repeated with a different production date, under different numbers
    g = df[ok].groupby("code_sequence")
    for seq, grp in g:
        if grp.code_number.nunique() <= 1:
            continue
        if grp.code_production.nunique() > 1:
            status[grp.index] = FAIL
            continue
        serials = grp.serial_digits.dropna().astype(float)
        if len(serials) >= 2 and (serials.max() - serials.min()) > SAME_BATCH_SERIAL_GAP:
            status[grp.index] = FAIL
        if grp.serial_digits.notna().sum() >= 2 and grp.serial_digits.nunique() < grp.serial_digits.notna().sum():
            status[grp.index] = FAIL  # identical serial under different numbers
    # identical serial under different numbers, regardless of sequence
    s_ok = df.serial_digits.notna()
    dup = df[s_ok].groupby("serial_digits").code_number.nunique()
    status[df.serial_digits.isin(dup[dup > 1].index)] = FAIL
    return status


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ocr", choices=["fresh", "stored"], default="fresh",
                    help="fresh = text read again from the current photos (14_fresh_ocr.py); "
                         "stored = text kept in production")
    args = ap.parse_args()

    cfg = load_config()
    out = output_dir(cfg)
    manifest = pd.read_csv(out / "manifest.csv")
    ocr_dir = out / ("ocr_fresh" if args.ocr == "fresh" else "ocr")
    suffix = "" if args.ocr == "fresh" else "_stored"
    if not ocr_dir.is_dir():
        sys.exit(f"ERROR: {ocr_dir} does not exist. Run 14_fresh_ocr.py first, or use --ocr stored.")

    rows, without_file, not_read = [], 0, 0
    for r in manifest.itertuples(index=False):
        try:
            with open(ocr_dir / f"{r.anon_id}.json", encoding="utf-8") as f:
                ocr = json.load(f)
        except (FileNotFoundError, ValueError):  # not processed yet: no text, every rule abstains
            ocr, without_file = {}, without_file + 1
        if args.ocr == "fresh" and not all((ocr.get(side) or {}).get("ocr_at") for side in ("front", "back")):
            not_read += 1  # Rekognition has not read both photos: measuring this person would only dilute the rates
            continue
        back = extract_back_fields(ocr.get("data_back_id"))
        ftext = front_text(ocr.get("data_front_id"))
        row = {
            "anon_id": r.anon_id, "tenant": r.tenant, "label": r.label, "channel": r.channel,
            "front_number": parse_front_number(ftext),
            "code_sequence": back.code.sequence if back.code else None,
            "code_number": back.code.number if back.code else None,
            "code_production": back.code.production_date.isoformat() if back.code and back.code.production_date else None,
            "serial_digits": back.serial[0] if back.serial else None,
            "registrar": back.registrar,
            "n_dates": len(back.dates),
            "printed_sex": back.printed_sex,
        }
        for res in evaluate(ftext, back):
            row[res.rule] = res.status
        rows.append(row)
    if not rows:
        sys.exit(f"ERROR: nobody in the manifest has been read yet in {ocr_dir}. Run 14_fresh_ocr.py --run first.")
    df = pd.DataFrame(rows)
    df["R_REPEATED_BACK"] = repeated_back_rule(df)
    if args.ocr == "fresh":
        print(f"text source: fresh ({ocr_dir})   people measured: {len(df)} of {len(manifest)}"
              + (f"   (left out, not read by Rekognition yet: {not_read})" if not_read else ""))
    else:
        print(f"text source: stored ({ocr_dir})   people: {len(df)}"
              + (f"   WITHOUT a text file: {without_file}" if without_file else ""))

    # 1. field legibility
    fields = {"front_number": "front number", "code_sequence": "back code", "serial_digits": "serial",
              "registrar": "registrar surname", "printed_sex": "printed sex"}

    # The values read from the cards are personal data: they stay in data/private/. The shared
    # per-customer file only says whether each field was readable and how each rule ended.
    value_cols = ["front_number", "code_sequence", "code_number", "code_production", "serial_digits"]
    (out / "private").mkdir(parents=True, exist_ok=True)
    df[["anon_id", "tenant", "label"] + value_cols].to_csv(out / "private" / f"rules_values{suffix}.csv", index=False)
    public = df.drop(columns=value_cols)
    for col in ("front_number", "code_sequence", "serial_digits"):
        public.insert(3, f"read_{col}", df[col].notna())
    public.to_csv(out / f"rules_per_customer{suffix}.csv", index=False)

    def legibility(by: list[str]) -> pd.DataFrame:
        return df.groupby(by, dropna=False).agg(**{v: (k, lambda s: s.notna().mean()) for k, v in fields.items()},
                                  two_dates=("n_dates", lambda s: (s >= 2).mean()), n=("anon_id", "size"))

    print("\n== Field legibility (share of customers with a readable value)")
    print(legibility(["tenant", "label"]).round(3).to_string())
    print("\n== Field legibility by channel (the channel decides the camera, so it can explain a difference)")
    print(legibility(["label", "channel"]).round(3).to_string())

    # 2. rule outcomes by label
    rule_cols = [c for c in df.columns if c.startswith("R_")]
    summary = []
    for rule in rule_cols:
        for lbl in ("POSITIVE", "NEGATIVE"):
            s = df.loc[df.label == lbl, rule]
            n_eval = (s != NOT_EVALUABLE).sum()
            summary.append({"rule": rule, "label": lbl, "n": len(s), "evaluable": n_eval,
                            "coverage": n_eval / max(1, len(s)),
                            "fail": (s == FAIL).sum(),
                            "fail_rate_of_evaluable": (s == FAIL).sum() / max(1, n_eval),
                            "fail_rate_of_all": (s == FAIL).sum() / max(1, len(s))})
    summary = pd.DataFrame(summary)
    summary.to_csv(out / f"rules_summary{suffix}.csv", index=False)
    print("\n== Rules by label (fail_rate on NEGATIVE = false positives; on POSITIVE = detection)")
    print(summary.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
