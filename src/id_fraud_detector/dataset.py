"""Dataset queries and labeling, shared by scripts/10_check_database.py and
scripts/11_export_dataset.py so both apply exactly the same rules.

Every statement here is a SELECT and runs on a read-only session (see common.pg_connection).
Dates are evaluated in the database session time zone, the one pgAdmin also uses by default,
so the counts match the queries in docs/plan.md (Anexo B).
"""
import hashlib
from collections.abc import Iterable, Iterator

import pandas as pd

from .common import query_df

POSITIVE, NEGATIVE = "POSITIVE", "NEGATIVE"

# One row per customer. No JSON columns: this stays small.
_CLIENTS_SELECT = """
WITH credit_flags AS (
    SELECT customer_id,
           bool_or(status = 'PAID')    AS has_paid,
           bool_or(status = 'NOVATED') AS has_novated
    FROM credits_credit
    GROUP BY customer_id
)
SELECT c.id                                   AS client_id,
       c.document_number,
       NULLIF(c.fraud_status, '')             AS fraud_status,
       c.originated_from                      AS channel,
       c.terms_and_conditions_timestamp       AS registered_at,
       c.terms_and_conditions_timestamp::date AS registered_date,
       (c.deleted_at IS NOT NULL)             AS deleted,
       c.front_id_image,
       c.back_id_image,
       (COALESCE(c.front_id_image, '') <> '' AND COALESCE(c.back_id_image, '') <> '') AS has_both_images,
       COALESCE(f.has_paid, false)            AS has_paid,
       COALESCE(f.has_novated, false)         AS has_novated
FROM users_client c
LEFT JOIN credit_flags f ON f.customer_id = c.id
"""
CLIENTS_SQL = _CLIENTS_SELECT + "WHERE c.terms_and_conditions_timestamp >= %(since)s\n"  # the dataset
CLIENTS_BY_ID_SQL = _CLIENTS_SELECT + "WHERE c.id = ANY(%(ids)s)\n"                      # any registration date
CLIENTS_FULL_BY_DOCUMENT_SQL = _CLIENTS_SELECT + "WHERE c.document_number = ANY(%(documents)s)\n"

# Candidate sources of confirmed fraud that fraud_status does not show yet (config.yaml,
# `confirmed_fraud_sources`). One row per credit, so the commerces that matched can be listed.
#   credit_at_commerce       every credit granted at a commerce inside a date window
#   payout_bank_at_commerce  credits paid out to an account of one bank, requested at a commerce
SOURCE_SQL = {
    "credit_at_commerce": """
SELECT cr.customer_id AS client_id, co.name AS commerce, cr.timestamp::date AS credit_date, cr.status AS credit_status
FROM credits_credit cr
JOIN commerces_commerce co ON co.id = cr.commerce_id
WHERE (co.name ILIKE %(commerce_like)s OR co.commercial_name ILIKE %(commerce_like)s)
  AND cr.customer_id IS NOT NULL
  AND (%(start)s::date IS NULL OR cr.timestamp::date >= %(start)s::date)
  AND (%(end)s::date IS NULL OR cr.timestamp::date <= %(end)s::date)
""",
    "payout_bank_at_commerce": """
SELECT rq.customer_id AS client_id, co.name AS commerce, cr.timestamp::date AS credit_date, cr.status AS credit_status
FROM credits_creditrequest rq
JOIN credits_payout p ON p.id = rq.payout_id
JOIN credits_credit cr ON cr.id = rq.credit_id
JOIN commerces_commerce co ON co.id = rq.commerce_id
WHERE (co.name ILIKE %(commerce_like)s OR co.commercial_name ILIKE %(commerce_like)s)
  AND p.bank_name ILIKE %(bank_like)s
  AND rq.customer_id IS NOT NULL
  AND (%(start)s::date IS NULL OR cr.timestamp::date >= %(start)s::date)
  AND (%(end)s::date IS NULL OR cr.timestamp::date <= %(end)s::date)
""",
}

# Cross-brand record of each person (public schema). Two things matter here:
#   - its fraud flag: mikro_back's Client.get_fraud_status() reads it before the tenant's own flag;
#   - when the person was first validated in ANY brand. A customer whose person was already
#     validated is not asked for photos again: mikro_back copies the photos kept on this record
#     (validate_from_common_client) and Rekognition does not run, so no OCR is stored for them.
#   The photos kept on this record are NOT used: mikro_back stores them only when the person
#   completes a validation (Client.check_user_validation), so customers who uploaded photos but
#   never finished, or were rejected, have none there. The dataset uses each tenant's photos.
COMMON_CLIENTS_SQL = """
SELECT document_number,
       NULLIF(fraud_status, '')                              AS common_fraud_status,
       CASE WHEN is_validated THEN validation_timestamp END  AS common_validated_at
FROM public.customers_commonclient
WHERE deleted_at IS NULL
  AND (NULLIF(fraud_status, '') IS NOT NULL OR is_validated)
"""
COMMON_COLUMNS = ["document_number", "common_fraud_status", "common_validated_at"]


def _lines_expr(column: str) -> str:
    """SQL expression: the LINE detections of one side, reduced to text + confidence, in OCR order.
    The reduction runs in the database so the full Rekognition JSON never leaves it."""
    path = f"v.{column}->'detected_text'->'TextDetections'"
    return f"""(SELECT jsonb_agg(jsonb_build_object('Type', d->'Type', 'DetectedText', d->'DetectedText',
                                                   'Confidence', d->'Confidence') ORDER BY ord)
                FROM jsonb_array_elements(CASE WHEN jsonb_typeof({path}) = 'array' THEN {path}
                                               ELSE '[]'::jsonb END) WITH ORDINALITY AS t(d, ord)
                WHERE d->>'Type' = 'LINE')"""


OCR_LINES_SQL = f"""
SELECT v.customer_id AS client_id,
       v.timestamp   AS validation_updated_at,
       {_lines_expr('data_front_id')} AS front_lines,
       {_lines_expr('data_back_id')}  AS back_lines
FROM users_customeridvalidationresult v
WHERE v.customer_id = ANY(%(ids)s)
"""


# Customers of the connected tenant for a list of document numbers, whenever they registered.
CLIENTS_BY_DOCUMENT_SQL = """
SELECT c.id AS client_id, c.document_number, c.front_id_image, c.back_id_image
FROM users_client c
WHERE c.document_number = ANY(%(documents)s)
  AND COALESCE(c.front_id_image, '') <> '' AND COALESCE(c.back_id_image, '') <> ''
"""


def fetch_clients_by_document(con, documents: Iterable[str], chunk_size: int = 5000) -> pd.DataFrame:
    """Customers with both images whose document number is in `documents`."""
    documents = sorted({str(d) for d in documents})
    parts = [query_df(con, CLIENTS_BY_DOCUMENT_SQL, {"documents": documents[i:i + chunk_size]})
             for i in range(0, len(documents), chunk_size)]
    columns = ["client_id", "document_number", "front_id_image", "back_id_image"]
    df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=columns)
    df["document_number"] = df["document_number"].astype(str)
    return df


def fetch_clients(con, since: str) -> pd.DataFrame:
    df = query_df(con, CLIENTS_SQL, {"since": since})
    df["document_number"] = df["document_number"].astype(str)
    return df


def fetch_clients_by_id(con, client_ids: Iterable[int]) -> pd.DataFrame:
    """Same columns as fetch_clients, for the given customers whatever their registration date."""
    df = query_df(con, CLIENTS_BY_ID_SQL, {"ids": [int(i) for i in client_ids]})
    df["document_number"] = df["document_number"].astype(str)
    return df


def fetch_clients_full_by_document(con, documents: Iterable[str]) -> pd.DataFrame:
    """Same columns as fetch_clients, for the customers with one of these document numbers."""
    df = query_df(con, CLIENTS_FULL_BY_DOCUMENT_SQL, {"documents": [str(d) for d in documents]})
    df["document_number"] = df["document_number"].astype(str)
    return df


def fetch_source_credits(con, source: dict) -> pd.DataFrame:
    """Credits matched by one entry of `confirmed_fraud_sources`: client_id, commerce,
    credit_date and credit_status, one row per credit."""
    params = {name: source.get(name) for name in ("commerce_like", "bank_like", "start", "end")}
    return query_df(con, SOURCE_SQL[source["kind"]], params)


def fetch_common_clients(con) -> pd.DataFrame:
    df = query_df(con, COMMON_CLIENTS_SQL)
    df["document_number"] = df["document_number"].astype(str)
    return df


def add_common_info(df: pd.DataFrame, common: pd.DataFrame | None) -> pd.DataFrame:
    """Attach the cross-brand record to the rows returned by fetch_clients.

    Adds common_fraud_status, common_validated_at and validated_elsewhere_first: True when the
    person was validated (in whatever brand) before registering in this tenant, which means the
    photos stored for this customer are a copy of older ones and were never read by Rekognition
    here. `common=None` (table not readable) leaves the three columns empty / False.
    """
    if common is None:
        common = pd.DataFrame(columns=COMMON_COLUMNS)
    out = df.merge(common[COMMON_COLUMNS], on="document_number", how="left")
    validated = pd.to_datetime(out["common_validated_at"], utc=True, errors="coerce")
    registered = pd.to_datetime(out["registered_at"], utc=True, errors="coerce")
    out["common_validated_at"] = validated
    out["validated_elsewhere_first"] = (validated < registered).fillna(False).astype(bool)
    return out


def fetch_ocr_lines(con, client_ids: Iterable[int], chunk_size: int = 300) -> Iterator[dict]:
    """Yield {client_id, validation_updated_at, front_lines, back_lines} for every customer that
    has a validation row. Queried in small chunks: each stored JSON is large on the server side."""
    ids = [int(i) for i in client_ids]
    for start in range(0, len(ids), chunk_size):
        with con.cursor() as cur:
            cur.execute(OCR_LINES_SQL, {"ids": ids[start:start + chunk_size]})
            columns = [d[0] for d in cur.description]
            for row in cur.fetchall():
                yield dict(zip(columns, row))


def stored_ocr(front_lines: list | None, back_lines: list | None) -> dict:
    """File content of ocr/<anon_id>.json: same nesting as the production columns."""
    return {"data_front_id": {"detected_text": {"TextDetections": front_lines or []}},
            "data_back_id": {"detected_text": {"TextDetections": back_lines or []}}}


VERIFIER_SOURCE = "VERIFIER_ANALYST"  # label_source of an ID an analyst confirmed as forged in the verifier


def confirmed_customers(con, tenant: str, cfg: dict, clients: pd.DataFrame, verifier_numbers=()) -> dict[str, set]:
    """client_ids of `tenant` confirmed as fraud outside fraud_status, by label source:
    CONFIRMED_<name> for each entry of `confirmed_fraud_sources` (config.yaml) of this tenant, and
    VERIFIER_ANALYST for the customers whose document number is in `verifier_numbers` (the IDs an
    analyst confirmed as forged in the team's verifier). Pass the result to label_clients."""
    confirmed: dict[str, set] = {}
    for source in cfg.get("confirmed_fraud_sources") or []:
        if source["tenant"] == tenant:
            confirmed[f"CONFIRMED_{source['name']}"] = set(fetch_source_credits(con, source)["client_id"])
    if verifier_numbers:
        match = clients["document_number"].astype(str).str.lstrip("0").isin(set(verifier_numbers))
        confirmed[VERIFIER_SOURCE] = set(clients.loc[match, "client_id"])
    return confirmed


def label_clients(df: pd.DataFrame, tenant: str, cfg: dict, confirmed: dict[str, set] | None = None) -> pd.DataFrame:
    """Add `group`, `label` and `label_source` to the rows returned by fetch_clients.

    group         same breakdown as the count query of docs/plan.md Anexo B (every customer)
    label         POSITIVE / NEGATIVE / None; only customers with both images get a label
    label_source  FRAUD, POSSIBLE_FRAUD_WINDOW, one of the keys of `confirmed`, or
                  NO_FLAG_WITH_PAID_CREDIT

    `confirmed` (see confirmed_customers) maps a label source to the client_ids confirmed as
    fraud by that source. They are positives under the same condition as the others (no PAID
    credit), whatever their fraud_status; one with a PAID credit gets no label at all, like a
    FRAUD that paid. A customer already positive by fraud_status keeps that source.
    """
    out = df.copy()
    status = out["fraud_status"]
    paid = out["has_paid"].astype(bool)
    registered = pd.to_datetime(out["registered_date"])

    fraud_no_paid = (status == "FRAUD") & ~paid
    in_window = pd.Series(False, index=out.index)
    for w in cfg.get("possible_fraud_windows", []):
        if w["tenant"] == tenant:
            in_window |= registered.between(pd.Timestamp(w["start"]), pd.Timestamp(w["end"]))  # inclusive
    window_no_paid = (status == "POSSIBLE FRAUD") & ~paid & in_window

    positives = [(fraud_no_paid, "FRAUD"), (window_no_paid, "POSSIBLE_FRAUD_WINDOW")]
    taken = fraud_no_paid | window_no_paid
    in_any_source = pd.Series(False, index=out.index)
    for source, client_ids in (confirmed or {}).items():
        in_source = out["client_id"].isin(client_ids)
        in_any_source |= in_source
        positives.append((in_source & ~paid & ~taken, source))
        taken |= in_source & ~paid
    no_flag_paid = status.isna() & paid
    negative = no_flag_paid & ~in_any_source  # confirmed fraud is never a genuine ID, even with a paid credit

    out["group"] = "UNUSED"
    out.loc[no_flag_paid, "group"] = "NEGATIVE"
    out.loc[status.notna(), "group"] = "EXCLUDED_OTHER_STATUS"
    out.loc[(status == "FRAUD") & paid, "group"] = "EXCLUDED_FRAUD_WITH_PAID"
    out.loc[fraud_no_paid, "group"] = "POSITIVE"

    usable = out["has_both_images"].astype(bool)
    out["label"] = None
    out["label_source"] = None
    for mask, label, source in (*((mask, POSITIVE, source) for mask, source in positives),
                                (negative, NEGATIVE, "NO_FLAG_WITH_PAID_CREDIT")):
        out.loc[mask & usable, "label"] = label
        out.loc[mask & usable, "label_source"] = source
    return out


def one_row_per_person(labeled: pd.DataFrame, tenant_order: list[str]) -> pd.DataFrame:
    """Keep one labeled customer per document number, across tenants.

    `labeled` holds the labeled customers of every tenant, with a `tenant` column. The same
    person can be a customer of several brands, usually with the very same photos (mikro_back
    copies them), so keeping every row would repeat the ID card in the dataset.

    Kept row, in this order: POSITIVE before NEGATIVE, the earliest registration (the brand where
    the photos were most likely taken), the order of the tenants in config.yaml, the lowest id.
    Adds `labeled_in` (tenants where the person is labeled, joined with "|") and `label_conflict`
    (True when the person is POSITIVE in one tenant and NEGATIVE in another; POSITIVE is kept).
    """
    df = labeled.copy()
    df["_negative"] = (df["label"] != POSITIVE).astype(int)
    df["_registered"] = pd.to_datetime(df["registered_at"], utc=True, errors="coerce")
    df["_tenant"] = df["tenant"].map({tenant: i for i, tenant in enumerate(tenant_order)})
    df = df.sort_values(["document_number", "_negative", "_registered", "_tenant", "client_id"],
                        na_position="last", kind="stable")
    by_person = df.groupby("document_number")
    df["labeled_in"] = df["document_number"].map(by_person["tenant"].agg(lambda s: "|".join(sorted(set(s)))))
    df["label_conflict"] = df["document_number"].map(by_person["label"].nunique() > 1)
    kept = df.drop_duplicates("document_number", keep="first")
    return kept.drop(columns=["_negative", "_registered", "_tenant"]).reset_index(drop=True)


def sample_negatives(df: pd.DataFrame, n: int, seed: int, keep=frozenset(), always: dict | None = None) -> set:
    """client_ids of the negatives that get images. Three parts, built so the choice does not
    move when the export runs again:

    keep     negatives whose photos are already on disk: always selected.
    always   negatives the classifier needs so the sample resembles the positives, from
             config.yaml: {"channels": [...], "registered_from": "YYYY-MM-DD"}. Every negative
             of those channels, and every negative registered since that date.
    sample   stratified by registration month and channel, proportional to stratum size (at
             least one per stratum, about `n` in total). The kept ones count toward their
             stratum; what is missing is taken in a fixed pseudo-random order that depends only
             on the person and the seed, not on who else is in the dataset.
    """
    neg = df[df["label"] == NEGATIVE]
    if len(neg) <= n:
        return set(neg["client_id"])
    month = pd.to_datetime(neg["registered_date"]).dt.strftime("%Y-%m")
    stratum = month + "|" + neg["channel"].fillna("?").astype(str)
    key = neg["anon_id"] if "anon_id" in neg else neg["client_id"]
    order = key.map(lambda value: hashlib.sha256(f"{seed}:{value}".encode()).hexdigest())
    kept = neg["client_id"].isin(set(keep))
    frac = n / len(neg)
    chosen = set(neg.loc[kept, "client_id"])
    for _, index in neg.groupby(stratum, sort=True).groups.items():
        quota = min(len(index), max(1, round(len(index) * frac)))
        missing = quota - int(kept[index].sum())
        if missing > 0:
            candidates = order[index][~kept[index]].sort_values()
            chosen |= set(neg.loc[candidates.index[:missing], "client_id"])
    if always:
        mask = neg["channel"].isin(always.get("channels") or [])
        if always.get("registered_from"):
            mask |= pd.to_datetime(neg["registered_date"]) >= pd.Timestamp(always["registered_from"])
        chosen |= set(neg.loc[mask, "client_id"])
    return chosen
