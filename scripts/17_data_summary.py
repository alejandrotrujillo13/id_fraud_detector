"""Phase 1 · Figures of the dataset, for the report and the presentation.

It connects to nothing: it only reads what the other scripts left under data/<stage>/ and
writes one small CSV per table in data/<stage>/summary/ (counts and rates only, no document
numbers, no names, no text of the cards). It also prints every table.

Tables (a table is skipped when its input is not on disk yet):
  01 universe            customers registered in the period, by tenant and group (10_check_database.py)
  02 people              people of the dataset by label, tenant and label source (manifest)
  03 channel             people by label and channel
  04 month               people by label and registration month
  05 stored_ocr          who has text stored in production, by label and channel
  06 photos              photos on disk: how many, size on disk, by label
  07 photo_sizes         most common photo sizes, by channel
  08 fresh_ocr           people and photos read again by Rekognition, and what it cost
  09 legibility          fields read from the fresh text, by label and channel
  10 legibility_width    back code read, by photo width
  11 rules               every rule by label: evaluable, FAIL, rate, and the 95 % ceiling when nobody fails
  12 detection_source    people with at least one FAIL, by label source
  13 detection_channel   people with at least one FAIL, by label and channel
  14 verifier            cases of the team's verifier, by verdict and by who set it
  15 crop_test           how often the OpenCV crop found the card (20_crop_test.py)

Usage:  python scripts/17_data_summary.py
"""
import json

import pandas as pd
from PIL import Image

from id_fraud_detector import verifier
from id_fraud_detector.common import image_path, load_config, output_dir
from id_fraud_detector.fresh_ocr import PRICE_PER_IMAGE_USD
from id_fraud_detector.ocr_store import SIDES

CODE_RULES = ["R_CODE_FORMAT", "R_NUMBER", "R_PRODUCTION", "R_REGISTRAR", "R_REPEATED_BACK"]


def save(summary_dir, name: str, table: pd.DataFrame, title: str) -> None:
    table.to_csv(summary_dir / f"{name}.csv", index=False)
    print(f"\n== {name} · {title}")
    print(table.to_string(index=False))


def photo_table(out, manifest: pd.DataFrame) -> pd.DataFrame:
    """One row per photo on disk: label, channel, side, width, height, megabytes."""
    rows = []
    for r in manifest.itertuples(index=False):
        for side in SIDES:
            path = image_path(out, r.tenant, r.label, r.anon_id, side)
            if path.exists():
                with Image.open(path) as image:
                    width, height = image.size
                rows.append({"anon_id": r.anon_id, "label": r.label, "channel": r.channel, "side": side,
                             "width": width, "height": height, "megabytes": path.stat().st_size / 1e6})
    return pd.DataFrame(rows, columns=["anon_id", "label", "channel", "side", "width", "height", "megabytes"])


def fresh_ocr_table(out, manifest: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for r in manifest.itertuples(index=False):
        path = out / "ocr_fresh" / f"{r.anon_id}.json"
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        done = [side for side in SIDES if (payload.get(side) or {}).get("ocr_at")]
        rows.append({"label": r.label, "photos_read": len(done), "both_sides": len(done) == len(SIDES)})
    return pd.DataFrame(rows, columns=["label", "photos_read", "both_sides"])


def main():
    cfg = load_config()
    out = output_dir(cfg)
    summary_dir = out / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(out / "manifest.csv")
    manifest["channel"] = manifest["channel"].fillna("(none)")
    print(f"manifest: {len(manifest)} people")

    groups = out / "checks" / "groups.csv"
    if groups.exists():
        table = pd.read_csv(groups)[["tenant", "group", "customers", "with_both_images"]]
        save(summary_dir, "01_universe", table,
             f"customers registered since {cfg['registered_since']}, by group (from 10_check_database.py; "
             "labeling by fraud_status only, as it was when that script last ran)")

    people = manifest.groupby(["label", "tenant", "label_source"]).size().rename("people").reset_index()
    save(summary_dir, "02_people", people, "people of the dataset, by label, tenant of the kept row and label source")
    totals = manifest.groupby("label").agg(
        people=("anon_id", "size"), in_several_tenants=("labeled_in", lambda s: s.str.contains("|", regex=False).sum()),
        selected_for_images=("has_images", "sum")).reset_index()
    save(summary_dir, "02_people_totals", totals, "people by label")

    channel = manifest.pivot_table(index="channel", columns="label", values="anon_id", aggfunc="size", fill_value=0)
    channel = channel.assign(**{f"{c}_share": (channel[c] / channel[c].sum()).round(4) for c in channel.columns})
    save(summary_dir, "03_channel", channel.reset_index(), "people by channel and label")

    month = manifest.assign(month=pd.to_datetime(manifest["registered_date"]).dt.strftime("%Y-%m")).pivot_table(
        index="month", columns="label", values="anon_id", aggfunc="size", fill_value=0)
    save(summary_dir, "04_month", month.reset_index(), "people by registration month and label")
    source_month = manifest[manifest["label"] == "POSITIVE"].assign(
        month=pd.to_datetime(manifest["registered_date"]).dt.strftime("%Y-%m")).pivot_table(
        index="month", columns="label_source", values="anon_id", aggfunc="size", fill_value=0)
    save(summary_dir, "04_month_positive_sources", source_month.reset_index(), "positives by month and label source")

    if "has_back_ocr" in manifest:
        stored = manifest.groupby(["label", "channel"]).agg(
            people=("anon_id", "size"), with_stored_front_text=("has_front_ocr", "sum"),
            with_stored_back_text=("has_back_ocr", "sum")).reset_index()
        stored["back_text_share"] = (stored["with_stored_back_text"] / stored["people"]).round(4)
        save(summary_dir, "05_stored_ocr", stored, "text already stored in production, by label and channel")

    photos = photo_table(out, manifest)
    if len(photos):
        on_disk = photos.groupby("label").agg(photos=("side", "size"), people=("anon_id", "nunique"),
                                              gigabytes=("megabytes", lambda s: round(s.sum() / 1e3, 2)),
                                              median_megabytes=("megabytes", "median")).round(2).reset_index()
        save(summary_dir, "06_photos", on_disk, "photos on disk")
        photos["size"] = photos["width"].astype(str) + "x" + photos["height"].astype(str)
        sizes = photos.groupby(["channel", "size"]).size().rename("photos").reset_index()
        sizes["share_of_channel"] = (sizes["photos"] / sizes.groupby("channel")["photos"].transform("sum")).round(4)
        sizes = sizes.sort_values(["channel", "photos"], ascending=[True, False]).groupby("channel").head(6)
        save(summary_dir, "07_photo_sizes", sizes, "most common photo sizes (width x height), by channel")
        widths = photos.assign(width_band=pd.cut(photos["width"], [0, 700, 1400, 100000],
                                                 labels=["up to 700 px", "701 to 1400 px", "over 1400 px"]))
        bands = widths.pivot_table(index="channel", columns="width_band", values="side", aggfunc="size",
                                   fill_value=0, observed=False)
        save(summary_dir, "07_photo_width_bands", bands.reset_index(), "photos by width, by channel")

    fresh = fresh_ocr_table(out, manifest)
    if len(fresh):
        table = fresh.groupby("label").agg(people_with_a_file=("label", "size"), people_fully_read=("both_sides", "sum"),
                                           photos_read=("photos_read", "sum")).reset_index()
        save(summary_dir, "08_fresh_ocr", table, "read again by Rekognition (people of the current manifest)")
        every_file = list((out / "ocr_fresh").glob("*.json"))
        photos_paid = sum(sum(1 for side in SIDES if (json.loads(p.read_text(encoding="utf-8")).get(side) or {})
                              .get("ocr_at")) for p in every_file)
        enlarged = len(list((out / "ocr_enlarged_test").glob("*.json"))) if (out / "ocr_enlarged_test").is_dir() else 0
        cost = pd.DataFrame([{"photos_read": photos_paid, "enlarged_test_photos": enlarged,
                              "usd": round((photos_paid + enlarged) * PRICE_PER_IMAGE_USD, 2)}])
        save(summary_dir, "08_rekognition_cost", cost, "Rekognition calls so far (every file on disk) and their cost")

    rules_path = out / "rules_per_customer.csv"
    if rules_path.exists():
        rules = pd.read_csv(rules_path).merge(manifest[["anon_id", "label_source"]], on="anon_id", how="left")
        rules["channel"] = rules["channel"].fillna("(none)")
        rule_cols = [c for c in rules.columns if c.startswith("R_")]
        fields = {"front_number": "read_front_number", "back_code": "read_code_sequence", "serial": "read_serial_digits"}
        rules["registrar_read"] = rules["registrar"].notna()
        rules["two_dates"] = rules["n_dates"] >= 2

        def legibility(by: list[str]) -> pd.DataFrame:
            return rules.groupby(by, observed=True).agg(
                people=("anon_id", "size"), **{name: (column, "mean") for name, column in fields.items()},
                registrar=("registrar_read", "mean"), two_dates=("two_dates", "mean")).round(4).reset_index()

        save(summary_dir, "09_legibility", legibility(["label", "channel"]),
             "share of people whose field was read from the fresh text (confidence >= 80)")
        save(summary_dir, "09_legibility_by_label", legibility(["label"]), "same, by label only")
        if len(photos):
            back_width = photos[photos["side"] == "back"].set_index("anon_id")["width"]
            rules["width_band"] = pd.cut(rules["anon_id"].map(back_width), [0, 700, 1400, 100000],
                                         labels=["up to 700 px", "701 to 1400 px", "over 1400 px"])
            save(summary_dir, "10_legibility_width", legibility(["width_band"]), "fields read, by width of the back photo")

        summary = []
        for rule in rule_cols:
            for label in ("POSITIVE", "NEGATIVE"):
                status = rules.loc[rules["label"] == label, rule]
                evaluable, fail = int((status != "NOT_EVALUABLE").sum()), int((status == "FAIL").sum())
                summary.append({"rule": rule, "label": label, "people": len(status), "evaluable": evaluable,
                                "coverage": round(evaluable / max(1, len(status)), 4), "fail": fail,
                                "fail_rate_of_evaluable": round(fail / max(1, evaluable), 4),
                                # rule of three: with no FAIL observed, the true rate is below 3/n (95 %)
                                "ceiling_95_when_no_fail": round(3 / evaluable, 4) if evaluable and not fail else None})
        save(summary_dir, "11_rules", pd.DataFrame(summary),
             "rules by label (FAIL on NEGATIVE = false positives; on POSITIVE = detection)")

        rules["any_fail"] = rules[rule_cols].eq("FAIL").any(axis=1)
        rules["code_rule_evaluable"] = rules[CODE_RULES].ne("NOT_EVALUABLE").any(axis=1)

        def detection(by: list[str]) -> pd.DataFrame:
            table = rules.groupby(by).agg(
                people=("anon_id", "size"), code_rule_evaluable=("code_rule_evaluable", "sum"),
                with_a_fail=("any_fail", "sum"),
                **{rule: (rule, lambda s: int((s == "FAIL").sum())) for rule in rule_cols}).reset_index()
            table.insert(len(by) + 3, "fail_share", (table["with_a_fail"] / table["people"]).round(4))
            return table

        save(summary_dir, "12_detection_source", detection(["label", "label_source"]),
             "people with at least one FAIL, by label source")
        save(summary_dir, "13_detection_channel", detection(["label", "channel"]),
             "people with at least one FAIL, by label and channel")

    history = verifier.history_dir(out)
    if history.is_dir():
        cases = [json.loads(p.read_text(encoding="utf-8")) for p in history.glob("*.json")]
        table = pd.DataFrame([{"verdict": c.get("real") or "(none)",
                               "set_by": {"Analista": "analyst", "Claude": "tool, from its score"}.get(
                                   c.get("real_origen"), "not recorded")} for c in cases])
        table = table.groupby(["verdict", "set_by"]).size().rename("cases").reset_index()
        save(summary_dir, "14_verifier", table, f"cases of the team's verifier ({len(cases)} in total)")
        numbers = pd.DataFrame([{"group": name, "id_numbers": len(found)}
                                for name, found in verifier.groups(out).items()])
        save(summary_dir, "14_verifier_people", numbers, "distinct ID numbers of the verifier, by verdict and who set it")

    crop = out / "crop_test" / "results.csv"
    if crop.exists():
        table = pd.read_csv(crop).groupby(["tenant", "side"]).agg(photos=("found", "size"), card_found=("found", "sum"),
                                                                  found_share=("found", "mean")).round(3).reset_index()
        save(summary_dir, "15_crop_test", table, "OpenCV crop: photos where a card outline was found")

    print(f"\ntables saved in {summary_dir}")


if __name__ == "__main__":
    main()
