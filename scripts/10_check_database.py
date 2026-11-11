"""Phase 1 · Pre-flight checks. Run this before 11_export_dataset.py.

Runs the read-only queries that used to be done by hand and prints what the export would
produce, without writing any personal data to disk. It only saves aggregated counts to
data/<stage>/checks/*.csv.

  1. connection        database and schema in use; session is read-only; time zone for dates
  2. groups            customers per group, to compare with docs/context.md §4.4
  3. labels            positives / negatives the export will produce
  4. by month          positives and negatives per registration month (input for the train/validation split)
  5. by channel        positives and negatives per channel (APP, COMMERCE, ...)
  6. NOVATED credits   labeled customers that have a NOVATED credit (not counted as PAID today)
  7. OCR               on a sample: is the stored Rekognition text where the parser expects it?
  8. both brands       labeled customers that also exist in the other tenant (same photos may be copied)
  9. cross-brand flag  labeled customers whose cross-brand record carries a fraud flag
 10. S3                can one image be read with the configured credentials? (skipped with --no-s3)

Usage:  python scripts/10_check_database.py [--ocr-sample 300] [--no-s3]
"""
import argparse
import sys
from contextlib import closing

import pandas as pd

from id_fraud_detector.common import load_config, output_dir, pg_connection, query_df, s3_client, stage
from id_fraud_detector.dataset import (NEGATIVE, POSITIVE, fetch_clients, fetch_common_fraud_status,
                                       fetch_ocr_lines, label_clients)

# Counts obtained by hand on sisteya_live on 2026-11-09 (docs/context.md §4.4 and §4.5):
# (customers, with both images). They drift a little every day (new customers, credits that get
# paid); a large gap is what matters. Only shown when STAGE=live.
LIVE_REFERENCE = {
    "esmio": {"POSITIVE": (296, 281), "NEGATIVE": (16014, 16011), "EXCLUDED_OTHER_STATUS": (13936, 11736),
              "window": (79, 78)},
    "llevaloya": {"POSITIVE": (45, 40), "NEGATIVE": (12077, 12075), "EXCLUDED_OTHER_STATUS": (13300, 12597),
                  "EXCLUDED_FRAUD_WITH_PAID": (1, 1)},
}


def reference(tenant: str) -> dict:
    return LIVE_REFERENCE.get(tenant, {}) if stage() == "live" else {}


def section(title: str) -> None:
    print(f"\n== {title}")


def show(df: pd.DataFrame) -> None:
    print(df.to_string(index=False) if len(df) else "   (no rows)")


def check_connection(con) -> None:
    info = query_df(con, "SELECT current_database() AS database, current_schema() AS schema, "
                         "current_user AS db_user, current_setting('transaction_read_only') AS read_only, "
                         "current_setting('TimeZone') AS time_zone")
    show(info)
    if info.loc[0, "read_only"] != "on":
        sys.exit("The session is NOT read-only. Stopping.")


def groups_table(tenant: str, clients: pd.DataFrame) -> pd.DataFrame:
    g = clients.groupby("group").agg(customers=("client_id", "size"), with_both_images=("has_both_images", "sum"))
    g = g.reset_index()
    ref = reference(tenant)
    if ref:
        g["live_reference_2026_11_09"] = [("{} ({})".format(*ref[x]) if x in ref else "") for x in g["group"]]
    g.insert(0, "tenant", tenant)
    return g


def labels_table(tenant: str, clients: pd.DataFrame) -> pd.DataFrame:
    labeled = clients[clients["label"].notna()]
    t = labeled.groupby(["label", "label_source"]).size().rename("customers").reset_index()
    t.insert(0, "tenant", tenant)
    return t


def month_table(tenant: str, clients: pd.DataFrame) -> pd.DataFrame:
    labeled = clients[clients["label"].notna()].copy()
    labeled["month"] = pd.to_datetime(labeled["registered_date"]).dt.strftime("%Y-%m")
    t = labeled.pivot_table(index="month", columns="label_source", values="client_id", aggfunc="size", fill_value=0)
    t = t.reset_index()
    t.insert(0, "tenant", tenant)
    return t


def channel_table(tenant: str, clients: pd.DataFrame) -> pd.DataFrame:
    labeled = clients[clients["label"].notna()].copy()
    labeled["channel"] = labeled["channel"].fillna("(empty)")
    t = labeled.pivot_table(index="channel", columns="label", values="client_id", aggfunc="size", fill_value=0)
    t = t.reset_index()
    t.insert(0, "tenant", tenant)
    return t


def novated_table(tenant: str, clients: pd.DataFrame) -> pd.DataFrame:
    labeled = clients[clients["label"].notna()]
    t = labeled.groupby("label").agg(customers=("client_id", "size"),
                                     with_novated_credit=("has_novated", "sum")).reset_index()
    t.insert(0, "tenant", tenant)
    return t


def ocr_table(con, tenant: str, clients: pd.DataFrame, sample_size: int, seed: int) -> pd.DataFrame:
    rows = []
    for label in (POSITIVE, NEGATIVE):
        group = clients[clients["label"] == label]
        sample = group.sample(min(len(group), sample_size), random_state=seed)
        found = {r["client_id"]: r for r in fetch_ocr_lines(con, sample["client_id"])}
        n = len(sample)
        front = sum(bool(r["front_lines"]) for r in found.values())
        back = sum(bool(r["back_lines"]) for r in found.values())
        rows.append({"tenant": tenant, "label": label, "sampled": n,
                     "with_validation_row": len(found), "with_front_text": front, "with_back_text": back,
                     "back_text_share": round(back / n, 3) if n else None})
    return pd.DataFrame(rows)


def ocr_structure(con, clients: pd.DataFrame) -> None:
    """Only called when the sample found no text: print the JSON keys actually stored (no content)."""
    ids = [int(i) for i in clients.loc[clients["label"].notna(), "client_id"].head(200)]
    sql = """
        SELECT (SELECT string_agg(k, ', ') FROM jsonb_object_keys(
                    CASE WHEN jsonb_typeof(v.data_back_id) = 'object' THEN v.data_back_id ELSE '{}'::jsonb END) k) AS back_keys,
               (SELECT string_agg(k, ', ') FROM jsonb_object_keys(
                    CASE WHEN jsonb_typeof(v.data_back_id->'detected_text') = 'object'
                         THEN v.data_back_id->'detected_text' ELSE '{}'::jsonb END) k) AS detected_text_keys
        FROM users_customeridvalidationresult v
        WHERE v.customer_id = ANY(%(ids)s)
    """
    keys = query_df(con, sql, {"ids": ids}).fillna("(none)")
    show(keys.groupby(["back_keys", "detected_text_keys"]).size().rename("customers").reset_index())


def both_brands_table(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Labeled customers of one tenant whose document number also exists in another tenant."""
    rows = []
    for tenant, df in frames.items():
        labeled = df[df["label"].notna()]
        for other, odf in frames.items():
            if other == tenant:
                continue
            other_all = set(odf["document_number"])
            other_images = set(odf.loc[odf["has_both_images"].astype(bool), "document_number"])
            other_label = dict(zip(odf.loc[odf["label"].notna(), "document_number"],
                                   odf.loc[odf["label"].notna(), "label"]))
            for label, group in labeled.groupby("label"):
                docs = group["document_number"]
                rows.append({
                    "tenant": tenant, "label": label, "customers": len(group), "other_tenant": other,
                    "also_registered_there": int(docs.isin(other_all).sum()),
                    "with_images_there": int(docs.isin(other_images).sum()),
                    "positive_there": int(sum(other_label.get(d) == POSITIVE for d in docs)),
                    "negative_there": int(sum(other_label.get(d) == NEGATIVE for d in docs)),
                })
    return pd.DataFrame(rows)


def common_flag_table(frames: dict[str, pd.DataFrame], common: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for tenant, df in frames.items():
        labeled = df[df["label"].notna()].merge(common, on="document_number", how="left")
        labeled["common_fraud_status"] = labeled["common_fraud_status"].fillna("(no flag)")
        t = labeled.pivot_table(index="label", columns="common_fraud_status", values="client_id",
                                aggfunc="size", fill_value=0).reset_index()
        t.insert(0, "tenant", tenant)
        parts.append(t)
    return pd.concat(parts, ignore_index=True).fillna(0)


def check_s3(bucket: str, frames: dict[str, pd.DataFrame]) -> None:
    try:
        s3 = s3_client()
    except Exception as e:
        print(f"   could not create the S3 client: {str(e)[:150]}")
        return
    for tenant, df in frames.items():
        labeled = df[df["label"].notna()]
        if labeled.empty:
            continue
        key = labeled.iloc[0]["front_id_image"]
        prefix = "/".join(str(key).split("/")[:4])  # env/tenant/clients/front_ids, no customer id
        try:
            s3.get_object(Bucket=bucket, Key=key, Range="bytes=0-0")
            print(f"   {tenant}: OK, read one object under {prefix}/")
        except Exception as e:
            print(f"   {tenant}: FAILED under {prefix}/ -> {str(e)[:150]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ocr-sample", type=int, default=300, help="customers per tenant and label for the OCR check")
    ap.add_argument("--no-s3", action="store_true", help="skip the S3 access check")
    args = ap.parse_args()

    cfg = load_config()
    print(f"STAGE={stage()}")
    checks_dir = output_dir(cfg) / "checks"
    checks_dir.mkdir(parents=True, exist_ok=True)

    frames: dict[str, pd.DataFrame] = {}
    tables: dict[str, list] = {"groups": [], "labels": [], "by_month": [], "by_channel": [], "novated": [], "ocr_sample": []}
    common, common_error = None, None

    for tenant in cfg["tenants"]:
        print(f"\n################ {tenant} ################")
        with closing(pg_connection(tenant)) as con:
            section("1. connection")
            check_connection(con)

            clients = label_clients(fetch_clients(con, cfg["registered_since"]), tenant, cfg)
            frames[tenant] = clients

            section(f"2. groups (registered since {cfg['registered_since']}); live reference = customers (with both images)")
            tables["groups"].append(groups_table(tenant, clients))
            show(tables["groups"][-1])
            window = clients[(clients["fraud_status"] == "POSSIBLE FRAUD") & ~clients["has_paid"].astype(bool)
                             & clients["label_source"].eq("POSSIBLE_FRAUD_WINDOW")]
            if any(w["tenant"] == tenant for w in cfg.get("possible_fraud_windows", [])):
                ref = reference(tenant).get("window")
                print(f"   POSSIBLE FRAUD inside the configured window, with both images: {len(window)}"
                      + (f"   (live reference: {ref[1]})" if ref else ""))

            section("3. labels the export will produce (customers with both images)")
            tables["labels"].append(labels_table(tenant, clients))
            show(tables["labels"][-1])

            section("4. labeled customers per registration month")
            tables["by_month"].append(month_table(tenant, clients))
            show(tables["by_month"][-1])

            section("5. labeled customers per channel")
            tables["by_channel"].append(channel_table(tenant, clients))
            show(tables["by_channel"][-1])

            section("6. labeled customers with a NOVATED credit (NOVATED is not counted as PAID)")
            tables["novated"].append(novated_table(tenant, clients))
            show(tables["novated"][-1])

            section(f"7. stored OCR on a sample of up to {args.ocr_sample} customers per label")
            try:
                ocr = ocr_table(con, tenant, clients, args.ocr_sample, cfg["seed"])
                tables["ocr_sample"].append(ocr)
                show(ocr)
                if ocr["with_back_text"].sum() == 0 and ocr["with_validation_row"].sum() > 0:
                    print("   No text found where the parser expects it. JSON keys actually stored:")
                    ocr_structure(con, clients)
            except Exception as e:
                print(f"   FAILED: {str(e).strip()[:300]}")

            if common is None and common_error is None:
                try:
                    common = fetch_common_fraud_status(con)
                except Exception as e:
                    common_error = str(e).strip()[:300]

    print("\n################ across tenants ################")
    section("8. labeled customers that also exist in the other tenant (registered since the cut-off date)")
    both = both_brands_table(frames)
    show(both)

    section("9. fraud flag on the cross-brand record (public.customers_commonclient) of labeled customers")
    flags = None
    if common is None:
        print(f"   could not be checked: {common_error}")
    else:
        flags = common_flag_table(frames, common)
        show(flags)

    section("10. S3 access")
    if args.no_s3:
        print("   skipped (--no-s3)")
    else:
        check_s3(cfg["bucket"], frames)

    for name, parts in tables.items():
        if parts:
            pd.concat(parts, ignore_index=True).fillna(0).to_csv(checks_dir / f"{name}.csv", index=False)
    both.to_csv(checks_dir / "both_brands.csv", index=False)
    if flags is not None:
        flags.to_csv(checks_dir / "common_fraud_flag.csv", index=False)
    print(f"\nAggregated tables saved in {checks_dir} (counts only, no personal data).")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as e:  # missing STAGE / ANON_SALT, expired SSO session, unreadable secret
        sys.exit(f"ERROR: {e}")
