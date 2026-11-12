"""Labeling and sampling tests on a synthetic customer table (no database needed).

Run:  python -m unittest discover tests
"""
import json
import tempfile
import unittest
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

import pandas as pd

from id_fraud_detector import dataset, verifier
from id_fraud_detector.dataset import (add_common_info, confirmed_customers, label_clients, one_row_per_person,
                                       sample_negatives, stored_ocr)
from id_fraud_detector.rules.ocr_parse import ocr_lines

CFG = {"possible_fraud_windows": [{"tenant": "esmio", "start": "2026-09-13", "end": "2026-10-04"}]}


def clients(rows):
    """rows: (fraud_status, has_paid, registered_date, has_both_images)"""
    return pd.DataFrame([{
        "client_id": i, "document_number": str(1000 + i), "fraud_status": status, "channel": "APP",
        "registered_date": registered, "has_both_images": images, "has_paid": paid, "has_novated": False,
    } for i, (status, paid, registered, images) in enumerate(rows)])


class Labeling(unittest.TestCase):
    def setUp(self):
        self.df = clients([
            ("FRAUD", False, date(2025, 5, 1), True),            # 0 positive
            ("FRAUD", True, date(2025, 5, 1), True),             # 1 fraud that paid: excluded
            ("POSSIBLE FRAUD", False, date(2026, 9, 13), True),  # 2 window start (inclusive)
            ("POSSIBLE FRAUD", False, date(2026, 10, 4), True),  # 3 window end (inclusive)
            ("POSSIBLE FRAUD", False, date(2026, 10, 5), True),  # 4 after the window
            ("POSSIBLE FRAUD", True, date(2026, 9, 20), True),   # 5 in the window but paid
            (None, True, date(2025, 8, 1), True),                # 6 negative
            (None, False, date(2025, 8, 1), True),               # 7 no flag, never paid: unused
            ("VERIFICATION REQUIRED", True, date(2025, 8, 1), True),  # 8 other status
            ("FRAUD", False, date(2025, 5, 1), False),           # 9 positive group, but one image missing
        ])

    def test_esmio_labels(self):
        out = label_clients(self.df, "esmio", CFG)
        self.assertEqual(list(out["label"]),
                         ["POSITIVE", None, "POSITIVE", "POSITIVE", None, None, "NEGATIVE", None, None, None])
        self.assertEqual(list(out["label_source"][[0, 2, 6]]),
                         ["FRAUD", "POSSIBLE_FRAUD_WINDOW", "NO_FLAG_WITH_PAID_CREDIT"])
        self.assertEqual(list(out["group"]),
                         ["POSITIVE", "EXCLUDED_FRAUD_WITH_PAID", "EXCLUDED_OTHER_STATUS", "EXCLUDED_OTHER_STATUS",
                          "EXCLUDED_OTHER_STATUS", "EXCLUDED_OTHER_STATUS", "NEGATIVE", "UNUSED",
                          "EXCLUDED_OTHER_STATUS", "POSITIVE"])

    def test_window_only_applies_to_its_tenant(self):
        out = label_clients(self.df, "llevaloya", CFG)
        self.assertEqual(list(out["label"]),
                         ["POSITIVE", None, None, None, None, None, "NEGATIVE", None, None, None])

    def test_empty_table(self):
        out = label_clients(self.df.iloc[0:0], "esmio", CFG)
        self.assertEqual(len(out), 0)

    def test_confirmed_sources_add_positives_under_the_same_conditions(self):
        confirmed = {"CONFIRMED_SHOP": {0, 4, 5, 6, 7, 9}, "VERIFIER_ANALYST": {4, 8, 3}}
        out = label_clients(self.df, "esmio", CFG, confirmed)
        self.assertEqual(list(out["label"]), [
            "POSITIVE",  # 0 already positive by FRAUD
            None,        # 1 fraud that paid
            "POSITIVE",  # 2 window
            "POSITIVE",  # 3 window, also in the verifier
            "POSITIVE",  # 4 POSSIBLE FRAUD after the window: positive now
            None,        # 5 in a source but has a PAID credit
            None,        # 6 was NEGATIVE: confirmed fraud with a PAID credit gets no label
            "POSITIVE",  # 7 no flag, never paid: positive now
            None,        # 8 in the verifier but has a PAID credit
            None,        # 9 one image missing
        ])
        self.assertEqual(list(out["label_source"][[0, 3, 4, 7]]),
                         ["FRAUD", "POSSIBLE_FRAUD_WINDOW", "CONFIRMED_SHOP", "CONFIRMED_SHOP"])  # first source wins
        self.assertEqual(list(out["group"]), list(label_clients(self.df, "esmio", CFG)["group"]))

    def test_confirmed_customers_reads_the_sources_of_the_tenant_and_the_verifier(self):
        cfg = {"confirmed_fraud_sources": [{"name": "SHOP", "tenant": "esmio", "kind": "credit_at_commerce"},
                                           {"name": "OTHER", "tenant": "llevaloya", "kind": "credit_at_commerce"}]}
        credits = pd.DataFrame({"client_id": [4, 4, 7]})
        with mock.patch.object(dataset, "fetch_source_credits", return_value=credits) as fetch:
            found = confirmed_customers(None, "esmio", cfg, self.df, verifier_numbers={"1008", "999"})
        self.assertEqual(found, {"CONFIRMED_SHOP": {4, 7}, "VERIFIER_ANALYST": {8}})
        self.assertEqual(fetch.call_count, 1)  # the other tenant's source is not queried
        self.assertEqual(confirmed_customers(None, "esmio", {}, self.df), {})


class VerifierHistory(unittest.TestCase):
    def test_only_an_analyst_verdict_counts_as_confirmed(self):
        cases = [("0012345", "FALSA", "Analista"), ("222", "FALSA", "Claude"), ("333", "FALSA", None),
                 ("12345", "FALSA", "Claude"),      # same person as the first: stays with the analyst
                 ("444", "AUTÉNTICA", "Analista"), ("555", "AUTÉNTICA", "Claude"), ("666", "Pendiente", "Analista"),
                 ("", "FALSA", "Analista")]
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            self.assertEqual(verifier.forged_by_analyst(out), set())  # no history saved
            verifier.history_dir(out).mkdir(parents=True)
            for i, (cc, verdict, origin) in enumerate(cases):
                (verifier.history_dir(out) / f"{i}.json").write_text(
                    json.dumps({"cc": cc, "real": verdict, "real_origen": origin}), encoding="utf-8")
            self.assertEqual(verifier.groups(out), {
                verifier.FORGED_BY_ANALYST: {"12345"}, verifier.FORGED_UNKNOWN_ORIGIN: {"333"},
                verifier.FORGED_BY_TOOL: {"222"}, verifier.GENUINE_BY_ANALYST: {"444"}})
            self.assertEqual(verifier.forged_by_analyst(out), {"12345"})


class CommonRecord(unittest.TestCase):
    def setUp(self):
        registered = datetime(2025, 6, 1, 15, tzinfo=timezone.utc)
        self.df = pd.DataFrame({"client_id": [1, 2, 3, 4], "document_number": ["10", "20", "30", "40"],
                                "registered_at": [registered] * 4})
        self.common = pd.DataFrame({
            "document_number": ["10", "20", "30"],
            "common_fraud_status": [None, "FRAUD", None],
            "common_validated_at": [registered - timedelta(days=300),   # validated in another brand first
                                    registered + timedelta(minutes=5),  # validated here, after registering
                                    None],                              # flagged or known, never validated
        })

    def test_first_validation_elsewhere(self):
        out = add_common_info(self.df, self.common)
        self.assertEqual(list(out["validated_elsewhere_first"]), [True, False, False, False])
        self.assertEqual(out["common_fraud_status"].fillna("-").tolist(), ["-", "FRAUD", "-", "-"])
        self.assertEqual(len(out), 4)

    def test_table_not_readable(self):
        out = add_common_info(self.df, None)
        self.assertEqual(list(out["validated_elsewhere_first"]), [False] * 4)
        self.assertTrue(out["common_fraud_status"].isna().all())


class OneRowPerPerson(unittest.TestCase):
    """Document numbers are made up."""

    def setUp(self):
        day = lambda d: datetime(2025, 3, d, 12, tzinfo=timezone.utc)
        rows = [
            # tenant, client_id, document, label, registered
            ("esmio", 1, "100", "NEGATIVE", day(10)),      # only in esmio
            ("esmio", 2, "200", "NEGATIVE", day(20)),      # in both, registered first in llevaloya
            ("llevaloya", 12, "200", "NEGATIVE", day(5)),
            ("esmio", 3, "300", "NEGATIVE", day(1)),       # negative here first, positive there later
            ("llevaloya", 13, "300", "POSITIVE", day(25)),
            ("llevaloya", 14, "400", "POSITIVE", day(8)),  # in both, same registration time
            ("esmio", 4, "400", "POSITIVE", day(8)),
            ("esmio", 6, "500", "NEGATIVE", day(9)),       # twice in the same tenant
            ("esmio", 5, "500", "NEGATIVE", day(9)),
        ]
        self.labeled = pd.DataFrame(rows, columns=["tenant", "client_id", "document_number", "label", "registered_at"])
        self.people = one_row_per_person(self.labeled, ["esmio", "llevaloya"]).set_index("document_number")

    def kept(self, document):
        return tuple(self.people.loc[document, ["tenant", "client_id"]])

    def test_one_row_per_document(self):
        self.assertEqual(sorted(self.people.index), ["100", "200", "300", "400", "500"])
        self.assertEqual(len(self.labeled), 9)  # the input is not modified

    def test_which_row_is_kept(self):
        self.assertEqual(self.kept("100"), ("esmio", 1))
        self.assertEqual(self.kept("200"), ("llevaloya", 12))  # earliest registration
        self.assertEqual(self.kept("300"), ("llevaloya", 13))  # positive wins over an earlier negative
        self.assertEqual(self.kept("400"), ("esmio", 4))       # tie: order of the tenants in the config
        self.assertEqual(self.kept("500"), ("esmio", 5))       # tie inside a tenant: lowest id

    def test_where_the_person_is_labeled(self):
        self.assertEqual(self.people["labeled_in"].to_dict(),
                         {"100": "esmio", "200": "esmio|llevaloya", "300": "esmio|llevaloya",
                          "400": "esmio|llevaloya", "500": "esmio"})
        self.assertEqual(self.people["label_conflict"].to_dict(),
                         {"100": False, "200": False, "300": True, "400": False, "500": False})
        self.assertEqual(self.people.loc["300", "label"], "POSITIVE")

    def test_empty_table(self):
        self.assertEqual(len(one_row_per_person(self.labeled.iloc[0:0], ["esmio"])), 0)


class Sampling(unittest.TestCase):
    def setUp(self):
        rows = [(None, True, date(2025, 1 + i % 12, 1), True) for i in range(600)]
        self.df = label_clients(clients(rows), "esmio", CFG)

    def test_size_is_close_to_target_and_reproducible(self):
        first = sample_negatives(self.df, 120, seed=42)
        self.assertEqual(first, sample_negatives(self.df, 120, seed=42))
        self.assertEqual(len(first), 120)

    def test_every_month_is_represented(self):
        chosen = self.df[self.df["client_id"].isin(sample_negatives(self.df, 24, seed=1))]
        self.assertEqual(pd.to_datetime(chosen["registered_date"]).dt.month.nunique(), 12)

    def test_asking_for_more_than_available_returns_all(self):
        self.assertEqual(len(sample_negatives(self.df, 10_000, seed=1)), 600)


class OcrFile(unittest.TestCase):
    def test_exported_file_is_readable_by_the_parser(self):
        lines = [{"Type": "LINE", "DetectedText": "SEXO M", "Confidence": 99.1}]
        self.assertEqual(ocr_lines(stored_ocr(None, lines)["data_back_id"]), ["SEXO M"])
        self.assertEqual(ocr_lines(stored_ocr(None, None)["data_front_id"]), [])


if __name__ == "__main__":
    unittest.main()
