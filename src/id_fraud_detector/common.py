"""Shared helpers: config loading, anonymization and read-only connections.

AWS access always goes through the default credential chain (the SSO session opened with
`aws sso login`); no access keys are stored in this project. Database credentials come from
Secrets Manager, the same secret mikro_back uses, and the database name from STAGE in .env.
"""
import functools
import hashlib
import json
import os
import re
from pathlib import Path

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]  # repository root (editable install)
load_dotenv(ROOT / ".env")

# A variable left blank in .env ("AWS_PROFILE=") must behave as if it were not there; otherwise
# boto3 would look for a profile named "" or for empty static keys instead of the SSO session.
for _name in ("AWS_PROFILE", "AWS_REGION", "AWS_DEFAULT_REGION", "AWS_ACCESS_KEY_ID",
              "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN"):
    if os.environ.get(_name) == "":
        del os.environ[_name]

DEFAULT_REGION = "us-east-1"
DEFAULT_DB_SECRET = "sisteya_db_proxy_credentials"  # the secret mikro_back reads: host, port, username, password


def load_config(path: Path | None = None) -> dict:
    """config.yaml with `tenants` already resolved for the current STAGE: the list under
    `tenants_by_stage` when the stage has one (dev -> demo), the default `tenants` otherwise."""
    with open(path or ROOT / "config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["tenants"] = (cfg.get("tenants_by_stage") or {}).get(stage(), cfg["tenants"])
    return cfg


def output_dir(cfg: dict) -> Path:
    """<output_dir>/<stage>: every environment gets its own folder, so data exported from dev
    can never end up mixed with the live dataset."""
    out = Path(cfg.get("output_dir", "data"))
    return (out if out.is_absolute() else ROOT / out) / stage()


def _salted_hash(*parts: str) -> str:
    salt = os.environ.get("ANON_SALT")
    if not salt:
        raise RuntimeError("ANON_SALT missing in .env (generate a long random string once and never change it)")
    return hashlib.sha256(":".join((salt, *parts)).encode()).hexdigest()[:16]


def anon_id(tenant: str, document_number: str) -> str:
    """Stable anonymous identifier for a customer of one tenant. Requires ANON_SALT in .env."""
    return _salted_hash(tenant, str(document_number))


def person_id(document_number: str) -> str:
    """Stable anonymous identifier for a person, the same in every tenant. It lets the dataset
    detect one person registered in two brands without exposing the document number."""
    return _salted_hash("person", "*", str(document_number))


def stage() -> str:
    """Environment to read from, taken from STAGE in .env (dev, test, live)."""
    value = (os.environ.get("STAGE") or "").strip().lower()
    if not re.fullmatch(r"[a-z0-9_]+", value):
        raise RuntimeError("STAGE missing or invalid in .env (for example STAGE=dev or STAGE=live)")
    return value


def aws_session():
    """boto3 session on the default credential chain: AWS_PROFILE if set, else the default profile."""
    import boto3

    region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION") or DEFAULT_REGION
    return boto3.Session(region_name=region)


def _sso_hint() -> str:
    profile = os.environ.get("AWS_PROFILE")
    return "aws sso login" + (f" --profile {profile}" if profile else "")


def _fetch_secret(secret_name: str) -> dict:
    from botocore.exceptions import BotoCoreError, ClientError

    try:
        response = aws_session().client("secretsmanager").get_secret_value(SecretId=secret_name)
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code", "")
        if code in ("ExpiredTokenException", "UnrecognizedClientException", "InvalidClientTokenId"):
            raise RuntimeError(f"The AWS session expired. Run `{_sso_hint()}` and try again.") from e
        raise RuntimeError(f"Could not read the secret '{secret_name}' from Secrets Manager: {code or e}") from e
    except BotoCoreError as e:  # no credentials, SSO token missing or expired, ...
        raise RuntimeError(f"No valid AWS session ({type(e).__name__}). Run `{_sso_hint()}` and try again.") from e
    return json.loads(response["SecretString"])


@functools.lru_cache(maxsize=1)
def db_settings() -> dict:
    """Connection settings: credentials from Secrets Manager, database name from STAGE.

    Optional overrides in .env (same names as mikro_back): DB_SECRET_NAME, DB_HOST, DB_PORT and
    DB_NAME. The user and password always come from the secret.
    """
    database = os.environ.get("DB_NAME") or f"sisteya_{stage()}"
    secret = _fetch_secret(os.environ.get("DB_SECRET_NAME") or DEFAULT_DB_SECRET)
    host = os.environ.get("DB_HOST") or secret.get("host")
    missing = [name for name, value in (("host", host), ("username", secret.get("username")),
                                        ("password", secret.get("password"))) if not value]
    if missing:
        raise RuntimeError(f"The database secret has no {', '.join(missing)}")
    port = str(os.environ.get("DB_PORT") or secret.get("port") or 5432)
    print(f"   database: {database} @ {host}:{port} (read-only session)")
    return {"host": host, "port": port, "dbname": database,
            "user": secret["username"], "password": secret["password"]}


def pg_connection(tenant: str):
    """Read-only psycopg2 connection with search_path set to the tenant schema.

    Read-only mode and search_path are set with SET statements after connecting (not as startup
    options, which a connection proxy may reject), and read-only is verified before returning.
    """
    import psycopg2
    from psycopg2 import sql

    con = psycopg2.connect(**db_settings(), application_name="id_fraud_detector", connect_timeout=20)
    con.set_session(readonly=True, autocommit=True)
    with con.cursor() as cur:
        cur.execute("SET default_transaction_read_only = on")
        cur.execute(sql.SQL("SET search_path TO {}, public").format(sql.Identifier(tenant)))
        cur.execute("SHOW transaction_read_only")
        if cur.fetchone()[0] != "on":
            con.close()
            raise RuntimeError("Could not open a read-only session; refusing to continue.")
    return con


def query_df(con, sql: str, params: dict | None = None):
    """Run a SELECT and return a DataFrame."""
    import pandas as pd

    with con.cursor() as cur:
        cur.execute(sql, params)
        return pd.DataFrame(cur.fetchall(), columns=[d[0] for d in cur.description])


def s3_client():
    """S3 client on the same SSO session. This project only calls GetObject."""
    return aws_session().client("s3")
