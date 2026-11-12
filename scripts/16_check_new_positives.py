"""Phase 1 · Read-only check of the positives that fraud_status does not show.

It changes nothing: it only counts what each source adds, so the labeling rules (applied by
11_export_dataset.py) can be decided and audited with numbers.

  1. The sources in config.yaml (`confirmed_fraud_sources`): customers with a credit at a
     commerce inside a window, or with a credit paid out to one bank at certain commerces.
  2. The history of the team's ID verifier, when it has been saved under
     data/<stage>/private/verifier_history/casos/ (one JSON per case): the ID numbers marked
     as forged or genuine, split by who set the verdict (an analyst, or the tool by itself).

For each group it prints how many customers exist, how many have both photos, when they
registered, their fraud_status, whether they have a paid credit and which label they have today.
Only counts are printed: no document numbers and no names.

Usage:  python scripts/16_check_new_positives.py
"""
import sys
from contextlib import closing

import pandas as pd

from id_fraud_detector import verifier
from id_fraud_detector.common import load_config, output_dir, pg_connection
from id_fraud_detector.dataset import (fetch_clients_by_id, fetch_clients_full_by_document, fetch_source_credits,
                                       label_clients)


def describe(clients: pd.DataFrame, tenant: str, cfg: dict, forged: bool = True) -> None:
    """Counts for a group of customers of one tenant (rows of fetch_clients_*). `label_today` is
    the label by fraud_status alone, without the sources this script counts."""
    if clients.empty:
        print("   no customers found")
        return
    df = label_clients(clients, tenant, cfg)
    since = pd.Timestamp(cfg["registered_since"])
    df["in_period"] = pd.to_datetime(df["registered_date"]) >= since
    df["label_today"] = df["label"].where(df["in_period"]).fillna("not in the dataset")
    both = df["has_both_images"].astype(bool)
    paid = df["has_paid"].astype(bool)
    usable = both & df["in_period"]
    print(f"   customers: {len(df)}   with both photos: {int(both.sum())}"
          f"   registered since {cfg['registered_since']}: {int(df['in_period'].sum())}"
          f"   with a PAID credit: {int(paid.sum())}")
    print("   fraud_status:      ", df["fraud_status"].fillna("(none)").value_counts().to_dict())
    print("   label by fraud_status alone:", df["label_today"].value_counts().to_dict())
    print("   channel:           ", df["channel"].fillna("(none)").value_counts().to_dict())
    print("   registration month:", pd.to_datetime(df["registered_date"]).dt.strftime("%Y-%m").value_counts()
          .sort_index().to_dict())
    if not forged:  # IDs confirmed as genuine: nothing to add to the positives
        return
    new = usable & ~paid & (df["label_today"] != "POSITIVE")
    print(f"   NEW positives (both photos, registered in the period, no PAID credit, not positive by "
          f"fraud_status): {int(new.sum())}"
          f"   left out for having a PAID credit: {int((usable & paid).sum())}")


def check_sources(cfg: dict) -> None:
    for source in cfg.get("confirmed_fraud_sources") or []:
        print(f"\n== source {source['name']} ({source['tenant']}, {source['kind']})")
        if source["tenant"] not in cfg["tenants"]:
            print(f"   skipped: tenant {source['tenant']} is not read with this STAGE")
            continue
        with closing(pg_connection(source["tenant"])) as con:
            credits = fetch_source_credits(con, source)
            if credits.empty:
                print("   no credit matched")
                continue
            print("   commerces matched: ", credits["commerce"].value_counts().to_dict())
            print("   credits:", len(credits), "  by status:", credits["credit_status"].value_counts().to_dict())
            print("   credits by month:  ", pd.to_datetime(credits["credit_date"]).dt.strftime("%Y-%m").value_counts()
                  .sort_index().to_dict())
            clients = fetch_clients_by_id(con, credits["client_id"].unique())
        describe(clients, source["tenant"], cfg)


def check_verifier(cfg: dict, out) -> None:
    groups = verifier.groups(out)
    if not any(groups.values()):
        print("\n== verifier history: nothing saved under private/verifier_history/casos, skipped")
        return
    for name, numbers in groups.items():
        used = "  (used as positives by 11_export_dataset.py)" if name == verifier.FORGED_BY_ANALYST else ""
        print(f"\n== verifier history · {name}: {len(numbers)} ID numbers{used}")
        for tenant in cfg["tenants"]:
            print(f"   -- {tenant}")
            with closing(pg_connection(tenant)) as con:
                clients = fetch_clients_full_by_document(con, sorted(numbers))
            describe(clients, tenant, cfg, forged=name != verifier.GENUINE_BY_ANALYST)


def main():
    cfg = load_config()
    out = output_dir(cfg)
    print(f"dataset period: registered since {cfg['registered_since']}   tenants: {cfg['tenants']}")
    check_sources(cfg)
    check_verifier(cfg, out)
    print("\nThis script only counts. The labels are applied by 11_export_dataset.py: the sources of "
          "config.yaml and the IDs an analyst confirmed as forged, always without a PAID credit.")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as e:  # missing STAGE, expired SSO session
        sys.exit(f"ERROR: {e}")
