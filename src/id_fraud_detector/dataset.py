"""Dataset queries and labeling, shared by scripts/10_check_database.py and
scripts/11_export_dataset.py so both apply exactly the same rules.

Every statement here is a SELECT and runs on a read-only session (see common.pg_connection).
Dates are evaluated in the database session time zone, the one pgAdmin also uses by default,
so the counts match the queries in docs/plan.md (Anexo B).
"""
from collections.abc import Iterable, Iterator

import pandas as pd

from .common import query_df

POSITIVE, NEGATIVE = "POSITIVE", "NEGATIVE"

# One row per customer registered since the cut-off date. No JSON columns: this stays small.
CLIENTS_SQL = """
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
WHERE c.terms_and_conditions_timestamp >= %(since)s
"""

# Fraud flag kept on the cross-brand record (public schema). mikro_back's Client.get_fraud_status()
# reads this one before the tenant's own flag.
COMMON_FRAUD_SQL = """
SELECT document_number, fraud_status AS common_fraud_status
FROM public.customers_commonclient
WHERE NULLIF(fraud_status, '') IS NOT NULL AND deleted_at IS NULL
"""


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


def fetch_clients(con, since: str) -> pd.DataFrame:
    df = query_df(con, CLIENTS_SQL, {"since": since})
    df["document_number"] = df["document_number"].astype(str)
    return df


def fetch_common_fraud_status(con) -> pd.DataFrame:
    df = query_df(con, COMMON_FRAUD_SQL)
    df["document_number"] = df["document_number"].astype(str)
    return df


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


def label_clients(df: pd.DataFrame, tenant: str, cfg: dict) -> pd.DataFrame:
    """Add `group`, `label` and `label_source` to the rows returned by fetch_clients.

    group         same breakdown as the count query of docs/plan.md Anexo B (every customer)
    label         POSITIVE / NEGATIVE / None; only customers with both images get a label
    label_source  FRAUD, POSSIBLE_FRAUD_WINDOW or NO_FLAG_WITH_PAID_CREDIT
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
    no_flag_paid = status.isna() & paid

    out["group"] = "UNUSED"
    out.loc[no_flag_paid, "group"] = "NEGATIVE"
    out.loc[status.notna(), "group"] = "EXCLUDED_OTHER_STATUS"
    out.loc[(status == "FRAUD") & paid, "group"] = "EXCLUDED_FRAUD_WITH_PAID"
    out.loc[fraud_no_paid, "group"] = "POSITIVE"

    usable = out["has_both_images"].astype(bool)
    out["label"] = None
    out["label_source"] = None
    for mask, label, source in ((fraud_no_paid, POSITIVE, "FRAUD"),
                                (window_no_paid, POSITIVE, "POSSIBLE_FRAUD_WINDOW"),
                                (no_flag_paid, NEGATIVE, "NO_FLAG_WITH_PAID_CREDIT")):
        out.loc[mask & usable, "label"] = label
        out.loc[mask & usable, "label_source"] = source
    return out


def sample_negatives(df: pd.DataFrame, n: int, seed: int) -> set:
    """client_ids of a sample of negatives stratified by registration month and channel,
    proportional to stratum size (at least one per stratum)."""
    neg = df[df["label"] == NEGATIVE]
    if len(neg) <= n:
        return set(neg["client_id"])
    month = pd.to_datetime(neg["registered_date"]).dt.strftime("%Y-%m")
    stratum = month + "|" + neg["channel"].fillna("?").astype(str)
    frac = n / len(neg)
    chosen: set = set()
    for _, group in neg.groupby(stratum, sort=True):
        k = min(len(group), max(1, round(len(group) * frac)))
        chosen |= set(group.sample(k, random_state=seed)["client_id"])
    return chosen
